"""Stage A of the 2D joint-distribution experiment: sample personas from a
fitted 1D mixture and have each sampled persona play TWO games (independent
calls, same system prompt). Output: paired samples with persona attribution.

Usage:
  joint_eval.py --game-a Public_Goods --game-b Dictator \
      --prompts-csv <...> --weights-pkl <...> --tag <...>
"""

import argparse
import json
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from replay_eval import generate_one, PROVIDER, GEN_MODEL, EXTRACT_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--game-a", required=True)
ap.add_argument("--game-b", required=True)
ap.add_argument("--prompts-csv", required=True)
ap.add_argument("--weights-pkl", required=True)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--tag", required=True)
ap.add_argument("--outdir", default="../reproduction/joint_results")
args = ap.parse_args()

df = pd.read_csv(args.prompts_csv)
prompts = df["prompt"].tolist()
with open(args.weights_pkl, "rb") as f:
    weights = np.asarray(pickle.load(f)[-1], dtype=float)
weights = np.clip(weights, 0, None)
weights = weights / weights.sum()
assert len(weights) == len(prompts)
print(f"{args.tag}: {len(prompts)} personas, {args.game_a} x {args.game_b}, n={args.n}")


def one_pair(i):
    xa = generate_one(args.game_a, prompts[i])
    xb = generate_one(args.game_b, prompts[i])
    return (i, xa, xb)


picks = random.choices(range(len(prompts)), weights=weights, k=args.n)
with ThreadPoolExecutor(args.workers) as ex:
    results = list(ex.map(one_pair, picks))

pairs = [(i, xa, xb) for i, xa, xb in results if xa is not None and xb is not None]
xa = np.array([p[1] for p in pairs])
xb = np.array([p[2] for p in pairs])
rho = spearmanr(xa, xb).statistic
print(f"collected {len(pairs)}/{args.n} pairs | sim spearman({args.game_a},{args.game_b}) = {rho:+.3f}")

os.makedirs(args.outdir, exist_ok=True)
out = f"{args.outdir}/{args.tag}.json"
with open(out, "w") as f:
    json.dump({"tag": args.tag, "provider": PROVIDER, "gen_model": GEN_MODEL,
               "extract_model": EXTRACT_MODEL,
               "game_a": args.game_a, "game_b": args.game_b,
               "prompts": prompts, "weights": weights.tolist(),
               "pairs": pairs, "sim_spearman": rho}, f, indent=1)
print("saved:", out)
