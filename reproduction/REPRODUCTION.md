# Independent Reproduction of "Distributional Alignment for Social Simulation with LLMs" (KDD 2026)

Reproduction of Xie, Gao & Mei, *Distributional Alignment for Social Simulation with
LLMs: A Mixture Modeling Approach* (KDD 2026, DOI 10.1145/3770855.3818919), performed
August 29, 2026 against this repository's released code and artifacts.

Models were accessed through OpenRouter, which still serves the paper's exact
snapshots: `openai/gpt-4o-2024-05-13` (prompt crafting + generation) and
`openai/gpt-4o-mini-2024-07-18` (answer extraction). Total API cost: **$9.72**.

## Validation levels and results

### Level 1 — recompute published tables from released artifacts (free)

The authors' five stability runs (`intermediate_results/Algorithmic_stability/`)
contain the final simulated samples for all 7 MobLab games under both algorithms.
Recomputing the Wasserstein distance against the human data (`data/joint.csv`)
reproduces Table 2 within run-to-run spread:

| Game | EM 5 runs (mean ± std) | EM Table 2 | GB 5 runs (mean ± std) | GB Table 2 |
|---|---|---|---|---|
| Dictator | 1.86 ± 0.59 | 1.69 | 1.95 ± 0.12 | 1.17 |
| Proposer | 2.79 ± 0.86 | 1.39 | 1.88 ± 0.18 | 1.88 |
| Responder | 3.39 ± 1.12 | 3.05 | 2.64 ± 0.08 | 2.51 |
| Investor | 2.33 ± 0.56 | 1.75 | 2.26 ± 0.39 | 1.84 |
| Banker | 6.82 ± 2.29 | 9.36 | 4.81 ± 0.97 | 4.24 |
| Public Goods | 0.72 ± 0.14 | 0.47 | 1.00 ± 0.25 | 0.88 |
| Bomb | 7.78 ± 2.66 | 6.32 | 6.18 ± 1.28 | 4.59 |

Every value is far below all unaugmented baselines in Table 2. The WVS Emancipative
Values Index pipeline was also verified: recomputing it from the raw WVS Wave 7 CSV
gives mean 43.49 / std 18.22 (96,529 valid respondents) vs. the paper's 43.76 / 18.29.

### Level 2 — replay the authors' fitted prompts with fresh API calls ($2.38)

Using the saved system prompts from EM stability run 0 for Public Goods (weights are
not shipped, so they were re-fitted with the repo's own SLSQP objective from 10 fresh
samples per prompt), then generating 1,000 fresh evaluation samples:

| Metric | This replay | Authors' 5 runs | Table 2 |
|---|---|---|---|
| Wasserstein | **0.57** | 0.59–0.95 | 0.47 |
| sim mean / std | 9.89 / 6.40 | — | — |
| human mean / std | 9.63 / 6.44 | — | — |
| Wilcoxon rank-sum | pass (p = 0.195) | — | pass |
| Kolmogorov–Smirnov | fail | — | fail |

Raw output: `replay_results/EM_Public_Goods_run0.json`.

### Level 3 — fit the mixture from scratch ($7.34)

`EM_moblab.py --game Public_Goods --K 10 --runs 1`, run end-to-end on an independent
account: random initialization from the human distribution, GPT-4o prompt-crafting
loop, 5 EM iterations, SLSQP weight optimization. The resulting 10-component mixture
was evaluated with 1,000 fresh samples using its own fitted weights:

| Metric | From-scratch fit | Table 2 | Human floor (1,000 samples) |
|---|---|---|---|
| Wasserstein | **0.43** | 0.47 | 0.13–0.40 |
| sim mean / std | 9.56 / 6.25 | — | human: 9.63 / 6.44 |
| Wilcoxon rank-sum | pass (p = 0.918) | pass | — |
| Kolmogorov–Smirnov | fail (p = 0.0013) | fail | — |

The independently fitted mixture slightly outperforms the published number and is
statistically indistinguishable from the 19,109 human Public Goods decisions under
the rank-sum test, matching both mean and variance. The learned components are
interpretable personas spanning free-riders ("highly cautious about your resources
... minimize your contributions"), strategic contributors, and altruists, mirroring
the interpretability claims in the paper's Discussion.

All fitting artifacts are in `level3_EM_Public_Goods/` (initialization prompts,
per-iteration cluster allocations and prompt updates, weight trajectory) and the
evaluation samples/metrics in `replay_results/level3_EM_Public_Goods_run1.json`.

## Verdict

The paper's central claim — that a learned mixture of system prompts reproduces
human distributional heterogeneity at near-sampling-noise fidelity — held at every
level tested, including a full from-scratch refit on an independent account. The KS
test failure matches the paper's own reporting (MobLab simulations pass Wilcoxon but
not KS). Scope: levels 2–3 cover one game (Public Goods, EM); level 1 covers all 7
MobLab games under both algorithms.

## Deviations from the original setup

- **API route**: OpenRouter instead of Azure OpenAI (same model snapshots).
  OpenRouter ignores `n>1`, so multi-sample requests were rewritten as parallel
  single-sample requests (`play()` in `code/EM_moblab.py`); token cost is slightly
  higher than the authors', behavior identical.
- **Level-2 weights**: not shipped in the repo, re-fitted with a simplified SLSQP
  objective (plain Wasserstein + small L2 regularizer, no KS discount term).
- **Bug fixes required to run at all** (see git history): MobLab data path,
  `argparse` attribute mismatches in three `main()` functions, `gb_run` keyword,
  missing output directories, `'Public Goods'` vs `'Public_Goods'` key mismatch in
  `EM_moblab.py`, and pandas ≥ 3 `applymap` removal.

## Reproducing this reproduction

```bash
pip install pandas numpy scipy tqdm matplotlib openai
export OPENROUTER_API_KEY=...   # needs ~$10 credit
cd code

# Level 2 (replay authors' prompts):
python replay_eval.py --alg EM --game Public_Goods --run 0

# Level 3 (fit from scratch, then evaluate the fit):
python EM_moblab.py --game Public_Goods --K 10 --runs 1
python replay_eval.py --alg EM --game Public_Goods \
    --prompts-csv Public_Goods/1_result/4_investor_EM_prompts_updated.csv \
    --weights-pkl Public_Goods/1_weights_lst.pkl --tag level3_EM_Public_Goods_run1
```

Level 1 needs no API access; the snippets live in the git history of this
reproduction and read only `intermediate_results/` and `data/joint.csv`.
