"""Stage C of the 2D experiment: craft personas against the human JOINT
distribution of two games, kmeans++-style — each round targets the human pair
worst covered by the current persona atoms, and the meta-prompt explicitly
allows context-decoupled traits (the failure mode of 1D-fitted mixtures).

Flash-only per study protocol. Output mirrors the EM/GB artifact layout.
"""

import argparse
import json
import os
import pickle
import statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from providers import api_call, GEN_MODEL
from replay_eval import generate_one, game2inst, human_col, gamerange

ap = argparse.ArgumentParser()
ap.add_argument("--game-a", default="Public_Goods")
ap.add_argument("--game-b", default="Dictator")
ap.add_argument("--rounds", type=int, default=25)
ap.add_argument("--samples", type=int, default=4, help="test pairs per candidate")
ap.add_argument("--improve", type=int, default=2)
ap.add_argument("--outdir", default="../reproduction/stageC_flash_PG_Dictator")
args = ap.parse_args()
GA, GB_ = args.game_a, args.game_b
RNG = np.random.default_rng(1)

H = pd.read_csv("../data/joint.csv")[[human_col[GA], human_col[GB_]]].dropna().values.astype(float)
scale = np.array([gamerange[GA], gamerange[GB_]], dtype=float)

CRAFT_TMPL = """## Goal

A chatbot will act as a single human participant playing TWO separate economic scenarios (in separate conversations). Craft one system prompt so the chatbot behaves like a consistent person who nevertheless may treat the two situations differently — real people often behave differently in group versus one-on-one settings.

## Scenario 1 ({ga})
"{inst_a}"

## Scenario 2 ({gb})
"{inst_b}"

## Desired Behavior

With your system prompt, the chatbot's decision should be approximately {a} in Scenario 1 and approximately {b} in Scenario 2. These two numbers may reflect different dispositions in the two contexts - that is realistic and desired. Do not mention either scenario or any number explicitly; describe the person's dispositions (e.g. attitudes toward groups, strangers, fairness, self-interest) so that the behaviors follow naturally.

## Output Format

Directly output the crafted system prompt starting with "You are ...".
"""

IMPROVE_TMPL = """With your crafted system prompt, the chatbot decided approximately {oa} in Scenario 1 and {ob} in Scenario 2, instead of the desired {a} and {b}. Improve the system prompt. Do not mention the scenarios or numbers explicitly. Directly output the improved system prompt starting with "You are ..."."""


def test_persona(prompt, n):
    with ThreadPoolExecutor(2 * n) as ex:
        fa = [ex.submit(generate_one, GA, prompt) for _ in range(n)]
        fb = [ex.submit(generate_one, GB_, prompt) for _ in range(n)]
        xa = [f.result() for f in fa]
        xb = [f.result() for f in fb]
    xa = [v for v in xa if v is not None]
    xb = [v for v in xb if v is not None]
    if not xa or not xb:
        return None
    return statistics.mode(xa), statistics.mode(xb), xa, xb


def craft(target):
    a, b = target
    messages = [{"role": "user", "content": CRAFT_TMPL.format(
        ga=GA, gb=GB_, inst_a=game2inst[GA], inst_b=game2inst[GB_], a=int(a), b=int(b))}]
    best = None
    for _ in range(1 + args.improve):
        prompt = api_call(GEN_MODEL, messages, max_tokens=2000).choices[0].message.content
        t = test_persona(prompt, args.samples)
        messages.append({"role": "assistant", "content": prompt})
        if t is None:
            continue
        oa, ob, xa, xb = t
        err = abs(oa - a) / scale[0] + abs(ob - b) / scale[1]
        if best is None or err < best[0]:
            best = (err, prompt, oa, ob, xa, xb)
        if err < 0.05:
            break
        messages.append({"role": "user", "content": IMPROVE_TMPL.format(
            oa=oa, ob=ob, a=int(a), b=int(b))})
    return best


pool = []  # (prompt, atom_a, atom_b, samples_a, samples_b, target)
for r in range(args.rounds):
    if pool:
        atoms = np.array([[p[1], p[2]] for p in pool]) / scale
        d = np.linalg.norm(H[:, None, :] / scale - atoms[None, :, :], axis=2).min(axis=1)
        probs = d ** 2
        probs = probs / probs.sum()
        target = H[RNG.choice(len(H), p=probs)]
    else:
        target = H[RNG.integers(len(H))]
    got = craft(target)
    if got is None:
        print(f"round {r}: craft failed for target {target}")
        continue
    err, prompt, oa, ob, xa, xb = got
    pool.append((prompt, oa, ob, xa, xb, target.tolist()))
    print(f"round {r}: target ({int(target[0])},{int(target[1])}) -> atom ({oa},{ob}) err {err:.3f}", flush=True)

# Voronoi weights over the final atoms (stage-B logic)
atoms = np.array([[p[1], p[2]] for p in pool]) / scale
dists = np.linalg.norm(H[:, None, :] / scale - atoms[None, :, :], axis=2)
w = np.bincount(dists.argmin(axis=1), minlength=len(pool)) / len(H)

os.makedirs(args.outdir, exist_ok=True)
pd.DataFrame({"prompt": [p[0] for p in pool]}).to_csv(f"{args.outdir}/prompts_for_eval.csv", index=False)
with open(f"{args.outdir}/weights_for_eval.pkl", "wb") as f:
    pickle.dump([w], f)
with open(f"{args.outdir}/fit_trace.json", "w") as f:
    json.dump([{"prompt": p[0], "atom": [p[1], p[2]], "samples_a": p[3],
                "samples_b": p[4], "target": p[5], "weight": w[i]}
               for i, p in enumerate(pool)], f, indent=1)
print(f"stage C fit done: {len(pool)} personas -> {args.outdir}")
