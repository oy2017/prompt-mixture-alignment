"""Stage B of the 2D experiment: analyze Stage A paired samples and retrain
mixture weights against the human joint. No API calls.

For each Stage A JSON:
  1. Induced vs human Spearman correlation.
  2. 2D EMD (POT) of simulated pairs vs human pairs, with two baselines:
     shuffle (destroys within-persona coupling, keeps marginals) and human
     split-half (sampling floor).
  3. Stage B: retrain weights over the fixed per-persona 2D components.
     With per-persona samples in hand, the optimal-weight problem is a
     transportation LP whose solution assigns each human point to its
     nearest persona component (exact for deterministic atoms; for noisy
     components we assign to the component with nearest mean, an upper
     bound). Reports the retrained-weight EMD = the best this persona set
     can do on the joint without new crafting.

Usage: joint_analyze.py <stageA_json> [...]
"""

import json
import sys
from collections import defaultdict

import numpy as np
import ot
import pandas as pd
from scipy.stats import spearmanr

human_col = {'Dictator': 'dictator', 'Proposer': 'ultimatum_1',
             'Responder': 'ultimatum_2', 'Investor': 'trust_1',
             'Banker': 'trust_3', 'Public_Goods': 'PG', 'Bomb': 'bomb'}
RNG = np.random.default_rng(0)


def emd2d(X, Y, scale):
    """EMD between two point clouds (uniform weights), coords scaled to [0,1]."""
    Xs, Ys = X / scale, Y / scale
    M = ot.dist(Xs, Ys, metric='euclidean')
    a = np.full(len(Xs), 1 / len(Xs))
    b = np.full(len(Ys), 1 / len(Ys))
    return ot.emd2(a, b, M, numItermax=1_000_000)


def analyze(path):
    d = json.load(open(path))
    ga, gb = d['game_a'], d['game_b']
    pairs = np.array([(p[1], p[2]) for p in d['pairs']], dtype=float)
    idx = np.array([p[0] for p in d['pairs']])

    df = pd.read_csv('../data/joint.csv')[[human_col[ga], human_col[gb]]].dropna()
    H = df.values.astype(float)
    scale = np.array([H[:, 0].max() - H[:, 0].min() or 1,
                      H[:, 1].max() - H[:, 1].min() or 1])
    n = min(len(pairs), len(H), 1000)
    Hs = H[RNG.choice(len(H), n, replace=False)]
    S = pairs[RNG.choice(len(pairs), n, replace=False)]

    rho_h = spearmanr(H[:, 0], H[:, 1]).statistic
    rho_s = spearmanr(pairs[:, 0], pairs[:, 1]).statistic

    # shuffle baseline: same marginals, coupling destroyed
    Sh = S.copy()
    Sh[:, 1] = RNG.permutation(Sh[:, 1])
    # human floor: split-half
    hperm = RNG.permutation(len(H))
    half = min(n, len(H) // 2)
    Ha, Hb = H[hperm[:half]], H[hperm[half:half * 2]]

    e_sim = emd2d(S, Hs, scale)
    e_shuf = emd2d(Sh, Hs, scale)
    e_floor = emd2d(Ha, Hb, scale)

    # Stage B: per-persona 2D component means; optimal weights = Voronoi mass
    comp = defaultdict(list)
    for i, (a, b) in zip(idx, pairs):
        comp[i].append((a, b))
    kids = sorted(comp)
    means = np.array([np.mean(comp[k], axis=0) for k in kids])
    dists = np.linalg.norm(Hs[:, None, :] / scale - means[None, :, :] / scale, axis=2)
    nearest = dists.argmin(axis=1)
    w_new = np.bincount(nearest, minlength=len(kids)) / len(Hs)
    # retrained cloud: resample personas by w_new, draw from their observed pairs
    draws = RNG.choice(len(kids), n, p=w_new)
    R = np.array([comp[kids[j]][RNG.integers(len(comp[kids[j]]))] for j in draws], dtype=float)
    e_retrained = emd2d(R, Hs, scale)
    rho_r = spearmanr(R[:, 0], R[:, 1]).statistic
    # within-persona determinism in 2D
    stds = [np.std(np.array(comp[k]), axis=0).mean() for k in kids if len(comp[k]) >= 5]
    zerovar = sum(1 for s in stds if s == 0)

    print(f"\n=== {d['tag']} ({ga} x {gb}, {d['gen_model']}) ===")
    print(f"pairs: {len(pairs)} | human complete cases: {len(H)}")
    print(f"spearman: human {rho_h:+.3f} | simulated {rho_s:+.3f} | "
          f"stageB-retrained {rho_r:+.3f}")
    print(f"2D EMD (normalized coords): sim {e_sim:.4f} | shuffle {e_shuf:.4f} | "
          f"stageB-retrained {e_retrained:.4f} | human floor {e_floor:.4f}")
    print(f"2D zero-variance personas: {zerovar}/{len(stds)}")

    d['stageB'] = {"weights_retrained": w_new.tolist(),
                   "component_means": means.tolist(),
                   "spearman_human": rho_h, "spearman_sim": rho_s,
                   "spearman_retrained": rho_r,
                   "emd_sim": e_sim, "emd_shuffle": e_shuf,
                   "emd_retrained": e_retrained, "emd_human_floor": e_floor,
                   "zero_variance_2d": [zerovar, len(stds)]}
    json.dump(d, open(path, 'w'), indent=1)


for p in sys.argv[1:]:
    analyze(p)
