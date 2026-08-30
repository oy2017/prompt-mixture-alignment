"""Level-2 replay validation: take the authors' saved system prompts
(intermediate_results/Algorithmic_stability), regenerate fresh samples with our
own API access, re-fit mixture weights, and compare the simulated distribution
against the human MobLab data (paper Table 2 metrics).

Usage:
  python replay_eval.py --alg EM --game Public_Goods --run 0 \
      [--m-samples 10] [--n-eval 1000] [--workers 16]

Requires OPENROUTER_API_KEY in the environment.
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import kstest, ranksums, wasserstein_distance

from providers import (PROVIDER, client, GEN_MODEL, EXTRACT_MODEL,  # noqa: F401
                       GEN_MAX_TOKENS)

game2inst = {
    'Dictator': "You are paired with another player. Your role is to decide how to divide $100 and the other player simply receives your choice. How would you like to divide the money? Please give only one concrete choice and highlight the amount you give to the other player in [] (such as [$x]).",
    'Proposer': "This is a two-player game. You are the Proposer, and the other player is the Responder. As the proposer, you propose how to divide $100 and the Responder chooses either Accept or Reject. If accepted, the two of you will earn as described by the accepted proposal accordingly. If rejected, then both of you will earn $0. \nHow much would you like to propose to give to the Responder? Please give only one concrete choice, and highlight the amount with [] (such as [$x]).",
    'Responder': "This is a two-player game. You are the Responder, and the other player is the Proposer. The proposer proposes how to divide $100 and you, as the Responder, choose either Accept or Reject. If accepted, the two of you will earn as described by the accepted proposal accordingly. If rejected, then both of you will earn $0. \nAs the Responder, what is the minimal amount in the proposal that you would accept? Please give only one concrete choice, and highlight the amount with [] (such as [$x]).",
    'Investor': "This is a two-player game. You are an Investor and the other player is a Banker. You have $100 to invest and you choose how much of your money to invest with the Banker. The amount you choose to invest will grow by 3x with the Banker. For example, if you invest $10, it will grow to $30 with the Banker. The Banker then decides how much of the money ($0-$30) to return to you, the Investor.\nHow much of the $100 would you like to invest with the Banker? Please give only one concrete choice, and highlight the number with [] (such as [$x]).",
    'Banker': "This is a two-player game. You are a Banker and the other player is an Investor, and the goal for each player is to earn more. The Investor chooses how much of the money (up to $100) to invest with you. The amount the Investor invests will generate a 2x return with you (the current value of investment becomes 3x).  To settle the investment, as the Banker, you get to decide how much of this total amount to return to the Investor and the rest will be kept as your profit.  For example, you can choose to return $0 (therefore the investor will lose their investment), or you can return the entire 3x (initial investment + 2x profit) to the investor, or you can return any amount in between.\nNow, if the investor has invested $50 with you and the current value became $150, how much of the $150 would you like to return to the Investor? Please give only one concrete choice, and highlight the number with [] (such as [$x]).",
    'Public_Goods': "In this public good game, you and 3 others will choose how much to contribute towards a water cleaning project. Each of you gets $20 per round to contribute between $0 and $20. The project has a 50% return rate. Your payoff relies on both your and others' contributions. Everyone benefits from the group's total contribution. Your payoff in each round equals the amount you didn't contribute (endowment - contribution) plus everyone's benefit (total contributions * 50% return rate). Here are two examples to calculate your payoff.\n\nExample one: You contributed $12; total group contributions were $20\n\nYour Payoff = ($20-$12) + $20*50% = $18\n\nExample two: You contributed $12; total group contributions were $30\n\nYour Payoff = ($20-$12) + $30*50% = $23\n\nWe will play a total of 3 rounds, in the first round, how much of the $20 would you like to contribute? Please give a concrete number and highlight it with [] (e.g., [x]).",
    'Bomb': "There are 100 boxes, and one bomb has been randomly placed in 1 of 100 boxes. You can choose to open 0-100 boxes at the same time. If none of the boxes you open has the bomb, you earn points that are equal to the number of boxes you open. If one of the boxes you open has the bomb, you earn zero points.  How many boxes would you open? Please give one concrete number and highlight it with [] (such as [x]).",
}

gamerange = {
    'Dictator': 100, 'Proposer': 100, 'Responder': 100, 'Investor': 100,
    'Banker': 150, 'Public_Goods': 20, 'Bomb': 100,
}

human_col = {
    'Dictator': 'dictator', 'Proposer': 'ultimatum_1', 'Responder': 'ultimatum_2',
    'Investor': 'trust_1', 'Banker': 'trust_3', 'Public_Goods': 'PG', 'Bomb': 'bomb',
}


def _chat(model, messages, max_tokens, retries=5):
    # providers.api_call sleeps through daily-quota exhaustion instead of
    # crashing, so a 1000-sample eval survives a mid-run RESOURCE_EXHAUSTED.
    from providers import api_call
    c = api_call(model, messages, max_tokens=max_tokens, retries=retries)
    return c.choices[0].message.content


def generate_one(game, system_prompt):
    """One simulated data point: decision by GEN_MODEL, integer extracted by
    regex, with EXTRACT_MODEL as fallback (mirrors EM_moblab.play)."""
    for _ in range(4):
        decision = _chat(GEN_MODEL, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": game2inst[game]},
        ], max_tokens=GEN_MAX_TOKENS)
        if decision is None:
            continue
        m = re.findall(r"\[\$?\s*(\d+(?:\.\d+)?)\s*\]", decision)
        if len(m) == 1:
            val = float(m[0])
        else:
            extracted = _chat(EXTRACT_MODEL, [
                {"role": "system", "content": "You are a helpful assistant who helps extract the choice in a conversation. With a conversation between a user and a chatbot provided, please extract the chatbot's choice regarding the user's question. "},
                {"role": "user", "content": game2inst[game]},
                {"role": "assistant", "content": decision},
                {"role": "user", "content": "Please output one single integer number that stands for the choice without anything else:"},
            ], max_tokens=GEN_MAX_TOKENS)
            digits = ''.join(filter(str.isdigit, extracted or ''))
            if not digits:
                continue
            val = float(digits)
        if 0 <= val <= gamerange[game]:
            return val
    return None


def fit_weights(per_prompt_samples, human, n_restarts=10, reg=0.1):
    K = len(per_prompt_samples)
    flat = np.concatenate(per_prompt_samples)
    reps = [len(s) for s in per_prompt_samples]

    def loss(w):
        w_rep = np.concatenate([np.full(n, wi / n) for wi, n in zip(w, reps)])
        return wasserstein_distance(human, flat, v_weights=w_rep) + reg * np.linalg.norm(w)

    best = None
    for _ in range(n_restarts):
        x0 = np.random.dirichlet(np.ones(K))
        res = minimize(loss, x0, method='SLSQP',
                       bounds=Bounds([0.] * K, [1.] * K),
                       constraints=LinearConstraint([[1.] * K], 1, 1), tol=1e-6)
        if best is None or res.fun < best.fun:
            best = res
    w = np.clip(best.x, 0, None)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", choices=["EM", "GB"], required=True)
    ap.add_argument("--game", choices=list(game2inst), required=True)
    ap.add_argument("--run", type=int, default=0, help="0-4: which stability run's prompts")
    ap.add_argument("--m-samples", type=int, default=10, help="samples per prompt for weight fitting")
    ap.add_argument("--n-eval", type=int, default=1000, help="final evaluation sample count")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--prompts-csv", help="evaluate a local fit-from-scratch run: CSV with a 'prompt' column")
    ap.add_argument("--weights-pkl", help="with --prompts-csv: pickle of the run's weights_lst (last entry used)")
    ap.add_argument("--tag", default=None, help="label used in the output filename")
    ap.add_argument("--outdir", default="../reproduction/replay_results")
    args = ap.parse_args()

    human = pd.read_csv("../data/joint.csv")[human_col[args.game]].dropna().values

    if args.prompts_csv:
        # Level-3 mode: the run produced its own prompts AND weights — no refit.
        df = pd.read_csv(args.prompts_csv)
        prompts = df["prompt"].tolist()
        import pickle
        with open(args.weights_pkl, "rb") as f:
            weights = np.asarray(pickle.load(f)[-1], dtype=float)
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()
        assert len(weights) == len(prompts), (len(weights), len(prompts))
        per_prompt = None
        print(f"level-3 eval {args.game}: {len(prompts)} fitted prompts, own weights")
        print("weights:", np.round(weights, 3))
        ex = ThreadPoolExecutor(args.workers)
    else:
        fname = "system_prompts.csv" if args.alg == "EM" else "system_prompt.csv"
        df = pd.read_csv(f"../intermediate_results/Algorithmic_stability/{args.alg}_moblab/{args.game}/{fname}")
        run_cols = [c for c in df.columns if not c.startswith("Unnamed")]
        prompts = df[run_cols[args.run]].dropna().tolist()
        print(f"{args.alg} {args.game} run {args.run}: {len(prompts)} saved prompts")
        ex = ThreadPoolExecutor(args.workers)

        # Stage 1.5: M samples per prompt, to re-fit the (unsaved) weights
        jobs = [(i, ex.submit(generate_one, args.game, p))
                for i, p in enumerate(prompts) for _ in range(args.m_samples)]
        per_prompt = [[] for _ in prompts]
        for i, fut in jobs:
            v = fut.result()
            if v is not None:
                per_prompt[i].append(v)
        keep = [i for i, s in enumerate(per_prompt) if len(s) > 0]
        prompts = [prompts[i] for i in keep]
        per_prompt = [per_prompt[i] for i in keep]
        print(f"fitting weights over {len(prompts)} prompts "
              f"({sum(map(len, per_prompt))} samples)")
        weights = fit_weights(per_prompt, human)
        print("weights:", np.round(weights, 3))

    if True:

        # Stage 2: 1,000 fresh samples from the weighted mixture
        picks = random.choices(range(len(prompts)), weights=weights, k=args.n_eval)
        futs = [ex.submit(generate_one, args.game, prompts[i]) for i in picks]
        pairs = [(i, f.result()) for i, f in zip(picks, futs)]
    attributed = [(i, v) for i, v in pairs if v is not None]
    sim = np.array([v for _, v in attributed])
    print(f"generated {len(sim)}/{args.n_eval} eval samples")

    # Stage 3: compare against the full human distribution
    w = wasserstein_distance(human, sim)
    wil = ranksums(human, sim).pvalue
    ks = kstest(human, sim).pvalue
    human_ref = wasserstein_distance(human, np.random.choice(human, 1000))
    print(f"\nW-dist(simulated, human) = {w:.2f}   "
          f"[reference: 1000 human samples = {human_ref:.2f}]")
    print(f"Wilcoxon rank-sum p = {wil:.3f}  ({'pass' if wil > .05 else 'fail'} at 0.05)")
    print(f"KS-test p = {ks:.3g}  ({'pass' if ks > .05 else 'fail'} at 0.05)")
    print(f"sim mean/std = {sim.mean():.2f}/{sim.std():.2f}   "
          f"human mean/std = {human.mean():.2f}/{human.std():.2f}")

    ex.shutdown(wait=False)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    tag = args.tag or f"{args.alg}_{args.game}_run{args.run}"
    out = f"{outdir}/{tag}.json"
    with open(out, "w") as f:
        json.dump({"alg": args.alg, "game": args.game, "run": args.run,
                   "provider": PROVIDER, "gen_model": GEN_MODEL,
                   "extract_model": EXTRACT_MODEL,
                   "prompts": prompts, "weights": weights.tolist(),
                   "per_prompt_samples": per_prompt,
                   "eval_samples": sim.tolist(),
                   # sample -> index of the mixture component that produced it,
                   # so per-component behaviour stays auditable after the fact
                   "eval_prompt_idx": [int(i) for i, _ in attributed],
                   "wasserstein": w, "wilcoxon_p": wil, "ks_p": ks}, f, indent=1)

    # Also emit the flat one-column-per-game CSV the repo uses for its own
    # cross-backbone artifacts (intermediate_results/*Cross_model_*).
    csv_out = f"{outdir}/{tag}_samples.csv"
    pd.DataFrame({args.game: sim,
                  "prompt_idx": [i for i, _ in attributed]}).to_csv(csv_out, index=False)
    print("saved:", out, "and", csv_out)


if __name__ == "__main__":
    main()
