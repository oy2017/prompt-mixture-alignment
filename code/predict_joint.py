"""Individual-level prediction test (finding 6): can a fitted mixture predict
one answer of a REAL person from their other answer?

For each human pair (a, b): form the posterior over mixture components given
the observed answer a (weight x Gaussian kernel on the normalized distance to
each component's game-A atom), predict b as the posterior-weighted mean of the
components' game-B atoms, and score the mean absolute error in dollars.

Baselines: the constant predictor (population mean of B — what you'd guess
knowing nothing about the person) and a linear regression B~A fitted on the
human data itself (a practical ceiling for "how predictable is B from A").

Uses only recorded artifacts (joint_results/*.json pairs carry per-prompt
attribution) — zero API calls.
"""

import argparse
import json

import numpy as np
import pandas as pd

# duplicated from replay_eval to keep this script import-light (replay_eval
# pulls in the API client, which this zero-API analysis doesn't need)
human_col = {
    'Dictator': 'dictator', 'Proposer': 'ultimatum_1', 'Responder': 'ultimatum_2',
    'Investor': 'trust_1', 'Banker': 'trust_3', 'Public_Goods': 'PG', 'Bomb': 'bomb',
}
gamerange = {
    'Dictator': 100, 'Proposer': 100, 'Responder': 100, 'Investor': 100,
    'Banker': 150, 'Public_Goods': 20, 'Bomb': 100,
}

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+", help="joint_results/*.json files (same game pair)")
ap.add_argument("--bandwidth", type=float, default=0.05,
                help="kernel bandwidth as a fraction of the game range")
args = ap.parse_args()


def load_mixture(path):
    d = json.load(open(path))
    pairs = np.array(d["pairs"], dtype=float)  # [prompt_idx, ans_a, ans_b]
    idx = pairs[:, 0].astype(int)
    atoms = np.full((len(d["prompts"]), 2), np.nan)
    for k in np.unique(idx):
        atoms[k] = np.median(pairs[idx == k, 1:], axis=0)
    w = np.asarray(d["weights"], dtype=float)
    keep = ~np.isnan(atoms).any(axis=1)  # prompts never sampled carry no atom
    return d, atoms[keep], w[keep] / w[keep].sum()


def predict(atoms, w, obs, obs_dim, h):
    # posterior over components given the observed answer, then the
    # posterior-weighted mean of the other coordinate
    pred_dim = 1 - obs_dim
    lik = np.exp(-0.5 * ((obs[:, None] - atoms[None, :, obs_dim]) / h) ** 2)
    post = w[None, :] * lik
    z = post.sum(axis=1)
    # if no component is anywhere near the observation, fall back to the prior
    prior_pred = w @ atoms[:, pred_dim]
    out = np.where(z > 1e-12, (post @ atoms[:, pred_dim]) / np.maximum(z, 1e-12), prior_pred)
    return out


first = json.load(open(args.results[0]))
GA, GB_ = first["game_a"], first["game_b"]
H = pd.read_csv("../data/joint.csv")[[human_col[GA], human_col[GB_]]].dropna().values.astype(float)
print(f"pair: {GA} x {GB_} | humans: {len(H)} | bandwidth {args.bandwidth} x range\n")

for obs_dim, (src, tgt) in enumerate([(GA, GB_), (GB_, GA)]):
    pred_dim = 1 - obs_dim
    actual = H[:, pred_dim]
    const = np.abs(actual - actual.mean()).mean()
    slope, icept = np.polyfit(H[:, obs_dim], actual, 1)
    linreg = np.abs(actual - (slope * H[:, obs_dim] + icept)).mean()
    # oracle: the empirical conditional median among humans with the same
    # observed answer (binned to 5% of range) — the practical MAE ceiling
    bins = np.round(H[:, obs_dim] / (0.05 * gamerange[src])).astype(int)
    med = {b: np.median(actual[bins == b]) for b in np.unique(bins)}
    oracle = np.abs(actual - np.array([med[b] for b in bins])).mean()
    print(f"== predict {tgt} from {src} ==")
    print(f"  constant (population mean, ignores the person): MAE {const:6.2f}")
    print(f"  linear regression fit on the humans themselves: MAE {linreg:6.2f}")
    print(f"  oracle (conditional median of humans, same bin): MAE {oracle:6.2f}")
    for path in args.results:
        d, atoms, w = load_mixture(path)
        assert (d["game_a"], d["game_b"]) == (GA, GB_), f"{path} is a different pair"
        h = args.bandwidth * gamerange[src]
        pred = predict(atoms, w, H[:, obs_dim], obs_dim, h)
        mae = np.abs(actual - pred).mean()
        print(f"  {d['tag']:45s} MAE {mae:6.2f}")
    print()
