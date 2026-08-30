"""Convert a GB_moblab run's result CSV into the prompts-csv + weights-pkl pair
that replay_eval.py consumes.

GB saves `<n>_result.csv` with one row per pool prompt; the `weights` column of
the LAST row holds the final weight vector (a numpy array repr, space-separated,
possibly multi-line). Usage: prepare_gb_eval.py <gb_run_dir> [n_test]
"""

import pickle
import sys

import numpy as np
import pandas as pd

gb_dir = sys.argv[1]
n_test = sys.argv[2] if len(sys.argv) > 2 else "1"

df = pd.read_csv(f"{gb_dir}/{n_test}_result.csv")
prompts = df["prompts"].tolist()
w_str = df["weights"].iloc[-1]
weights = np.array([float(x) for x in w_str.strip("[]").split()])
assert len(weights) == len(prompts), (len(weights), len(prompts))

pd.DataFrame({"prompt": prompts}).to_csv(f"{gb_dir}/prompts_for_eval.csv", index=False)
with open(f"{gb_dir}/weights_for_eval.pkl", "wb") as f:
    pickle.dump([weights], f)
print(f"{gb_dir}: {len(prompts)} prompts, weights sum {weights.sum():.4f}")
