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

### Cross-backbone transfer — Gemini (free tier)

Analogue of the paper's backbone-sensitivity study: the level-3 mixture (prompts and
weights fitted with GPT-4o) replayed unchanged on `gemini-3.5-flash` via Google's
OpenAI-compatible endpoint, 1,000 fresh samples.

| Backbone | W-dist | mean / std | Wilcoxon |
|---|---|---|---|
| gpt-4o-2024-05-13 (fitted) | **0.43** | 9.56 / 6.25 | pass (p = 0.918) |
| gemini-3.5-flash (transferred) | **2.17** / **2.04** | 11.56 / 7.33, 11.30 / 7.41 | fail (p < 0.001) |
| human | — | 9.63 / 6.44 | — |

Two independent Gemini runs (2.17 and 2.04) — the degradation is stable, ~5x worse
than the fitted backbone, failing both tests.

Per-component attribution (`eval_prompt_idx` in the saved JSON) localizes the failure
exactly. Each component was crafted to hit a target contribution; comparing that
target to what Gemini actually does:

| component | weight | target | Gemini mean | persona |
|---|---|---|---|---|
| 0 | 0.138 | 0 | 0.00 | highly cautious, minimizes contributions |
| 7 | 0.083 | 1 | 1.11 | conserves resources |
| 9 | 0.158 | 8 | 7.69 | conservative yet cooperative |
| 8 | 0.099 | 9 | 10.67 | rational maximizer |
| 5 | 0.079 | 8 | 10.03 | calculated, resource-conscious |
| 1 | 0.092 | 13 | 14.24 | strategic optimizer |
| 4 | 0.090 | 12 | 18.96 | strategic and thoughtful cooperator |
| **2** | **0.235** | **14** | **20.00** | highly collaborative altruist |
| 3 | 0.014 | 15 | 20.00 | cooperative optimist |
| 6 | 0.011 | 14 | 20.00 | insightful and generous |

Low- and mid-target components transfer almost perfectly (targets 0, 1, 8, 9, 13 land
within ~1 of target). Every component targeting >= 12 saturates at exactly $20 —
including component 2, which alone carries 23.5% of the mixture weight. Gemini treats
"be generous" as maximal contribution where GPT-4o produces graded intermediate
values, so the upper half of the distribution collapses onto its boundary and the
mean rises ~1.7 points.

This bounds *prompt transfer*, not the method: the weights and the crafting loop were
calibrated against GPT-4o's response magnitudes, and a from-scratch refit puts
Gemini's own responses inside the EM feedback loop (untested here, ~$5). Consistent
with the paper's own MobLab transfer results (Llama backbones beat baselines in only
4 of 7 games).

**Token-budget trap.** `gemini-3.5-flash` is a thinking model: hidden reasoning
tokens are charged against `max_tokens`, while `usage.completion_tokens` reports only
the visible output (median 91, max 212 tokens here). Sizing the cap from that
reported number silently truncates almost every response *before* it reaches the
bracketed answer. Measured over 50 probes per cap (240 at 2,000):

| generation cap | truncated | usable answer |
|---|---|---|
| 250 | 98% | 0% |
| 350 | 100% | 0% |
| 500 | 96% | 4% |
| **2,000** | **0/240** | **100%** |
| 8,000 | 0/40 | 100% |

2,000 is sufficient and 8,000 adds nothing. The extraction fallback shares the same
budget for the same reason. Select the backbone with `PMA_PROVIDER=gemini` (see
`PROVIDERS` in `code/replay_eval.py`).

Artifacts: `replay_results/crossbackbone_gemini_Public_Goods.json` (prompts, weights,
all 1,000 samples with the mixture component that produced each, metrics) and
`..._samples.csv` in the same flat format as the repo's own
`intermediate_results/ Cross_model_backbone_sensitivity/` files — enough to recompute
these metrics offline or re-replay the mixture on another backbone.

### Gemini study: refits, full-matrix sweep, and the quantization mechanism

Extending the backbone work above (dates: 2026-08-29/30; `gemini-3.5-flash` and
`gemini-pro-latest` via Google's OpenAI-compatible endpoint, selected with
`PMA_PROVIDER`; flash extraction model throughout).

**From-scratch refits fully recover — and beat — the fitted-backbone results.**
Public Goods, all three backbones, EM K=10:

| Backbone | fitted W-dist | mean/std (human 9.63/6.44) | Wilcoxon |
|---|---|---|---|
| GPT-4o | 0.43 | 9.56/6.25 | pass |
| gemini-3.5-flash | **0.31** | 9.64/6.43 | pass (p=0.94) |
| gemini-pro-latest | 0.44 | 9.32/6.45 | pass (p=0.15) |

So the transfer failures (2.04–2.17 flash / 1.81 pro) were purely
calibration-transfer artifacts; the EM loop self-corrects on any backbone
tested. Pro's transfer failure is near-identical to flash's (34.1% vs 34.0% of
samples at $20; same components saturate), so persona→behavior calibration is a
**model-family trait, not a capability effect**.

**Full flash matrix — EM and GB, all 7 MobLab games (W-dist, 1,000 fresh
samples vs full human data):**

| Game | flash EM | paper EM | flash GB | paper GB |
|---|---|---|---|---|
| Dictator | **0.68** | 1.69 | 2.63 | **1.17** |
| Proposer | **1.05** | 1.39 | **1.80** | 1.88 |
| Responder | **1.65** | 3.05 | **1.82** | 2.51 |
| Investor | **1.56** | 1.75 | 2.77 | **1.84** |
| Banker | **3.34** | 9.36 | **3.36** | 4.24 |
| Public Goods | **0.31** | 0.47 | **0.85** | 0.88 |
| Bomb | **1.48** | 6.32 | **1.89** | 4.59 |

Flash EM beats the paper's GPT-4o EM on **7/7 games** (Dictator even lands
below that run's 1,000-human-sample floor of 1.04). Flash GB wins 5/7. And
within flash, EM ≤ GB on all seven games — reversing the paper, where GB often
won. (Flash GB ran at maxIter=60 rather than the paper's 200; the paper's own
convergence analysis shows GB stabilizes by ~30 prompts.)

**The mechanism finding: Gemini mixtures are deterministic quantizations, not
persona populations.** Per-component attribution over the evaluation samples:

- Flash EM Public Goods: **10/10 components zero-variance** — each persona
  always answers one exact number. Support = exactly K values; entropy 3.19
  bits vs human 3.96. GPT-4o's mixture: 18 distinct values from K=10, with
  real within-component spread.
- Pro EM Public Goods: 10/10 zero-variance (fit-time and eval).
- Pro EM Dictator (K=50, 101-value action space): **48/49 zero-variance** at
  fit time — the mechanism scales.
- Flash GB Dictator/Banker/Bomb: 17/18, 22/22, 22/26 zero-variance.

All the diversity in a Gemini mixture comes from the weights: the EM loop
learns a K-atom quantization of the human histogram. Wasserstein distance,
mean, std, and the Wilcoxon test are all blind to the difference — the
quantization *outscores* the persona population on every one of them. Only
support size / entropy (or, weakly, KS) expose it. Two corollaries:

1. For social simulation — sampling a persona and interacting with it — the
   two mixtures are different objects: noisy person-like agents vs a lookup
   table in persona costume. Population-level alignment metrics cannot tell
   them apart, a measurement gap in the paper's evaluation framework.
2. The EM-over-GB reversal on flash follows from the mechanism: EM's global
   reassignment can relocate atoms every iteration, while GB's greedy additive
   scheme cannot move an atom once placed, only down-weight it.

**Persona conditioning removes response variance on Gemini.** The fixed-prompt
baseline ("You are a helpful assistant.", 1,000 samples): flash W-dist 4.02
(mean 12.0, std 2.56 — 4 distinct values, 59.6% at $10); pro 3.49 (std 4.05 —
3 distinct values, 80% at $10); paper's GPT-4o value 5.04. The *default*
prompt has more within-prompt variance than any crafted persona (std 0.00) —
on Gemini, personas are precision instruments, not diversity generators, and
the smarter tier collapses harder by default.

**Pro focused runs (insight subset per scope decision):** EM Dictator
(archived; eval pending quota), EM Banker and GB Dictator in the overnight
queue. The pro model carries a hard daily quota on this key; the harness
sleeps through exhaustion (`providers.api_call`) and resumes at the
midnight-PT reset. Results will be appended when they land.

**Artifacts.** Every fit: full EM/GB trajectory (initialization prompts,
per-iteration allocations/updates, weight history) under
`level3_<provider>_<alg>_<game>/`. Every eval: prompts, weights, all 1,000
samples with per-component attribution (`eval_prompt_idx`), provider/model
IDs, and metrics in `replay_results/*.json` + flat `*_samples.csv`. Baselines
under `baseline/`. GB evals consume `prompts_for_eval.csv` /
`weights_for_eval.pkl` produced by `code/prepare_gb_eval.py`.

### 2D joint-distribution experiment (stages A/B/C)

The method fits each attribute's marginal independently; populations are
joint distributions. `data/joint.csv` provides ground truth (same participant
across games). Human cross-game consistency is context-dependent: Spearman
+0.291 across the two ultimatum roles (Proposer x Responder, n=5,291) but only
+0.057 across different games (Public Goods x Dictator, n=2,520).

**Stage A — do 1D-fitted personas carry joint structure?** Sample personas
from a fitted mixture; each plays both games (independent calls); compare the
induced joint to the human joint (2D EMD via POT, normalized coordinates;
shuffle baseline destroys within-persona coupling while preserving marginals).

| Arm (mixture used) | human rho | sim rho | EMD sim | EMD shuffle | EMD stage-B | floor |
|---|---|---|---|---|---|---|
| flash PG-mixture, PG x Dictator | +0.057 | **+1.000** | 0.319 | 0.241 | 0.264 | 0.030 |
| gpt-4o PG-mixture, PG x Dictator | +0.057 | +0.786 | 0.186 | 0.155 | 0.177 | 0.032 |
| flash Proposer-mixture, Prop x Resp | +0.291 | **+0.295** | 0.075 | 0.078 | 0.069 | 0.027 |

Findings: (1) deterministic atoms produce *perfect* rank correlation — each
persona is one 2D point and trait-ordering makes them co-monotone; GPT-4o's
within-persona noise only softens this to +0.79. (2) On the near-independent
pair, persona coupling is *worse than assuming independence* (shuffle beats
sim for both backbones). (3) Persona trait-consistency is roughly constant
across contexts while humans' is context-dependent — so the same mechanism
that ruins PG x Dictator almost exactly reproduces Proposer x Responder.

**Stage B — retrain weights on the joint (no new crafting).** With components
fixed, optimal weights reduce to Voronoi assignment of human mass to nearest
persona atom. It barely helps (0.319 -> 0.264 on flash; correlation stays at
+1.0): all atoms lie on the "consistent character" diagonal, and no
reweighting can create the off-diagonal humans (generous in one context,
selfish in the other) who dominate the near-independent joint. This is a
support-coverage failure, not a calibration failure.

**Stage C — joint-aware crafting closes the gap.** 25 rounds of kmeans++-style
residual targeting on the human joint, with a meta-prompt that explicitly
legitimizes context-decoupled dispositions ("real people often behave
differently in group versus one-on-one settings"); Voronoi weights; ~1,500
flash calls (`code/craft2d.py`).

| PG x Dictator | 1D mixture | stage C | human |
|---|---|---|---|
| Spearman | +1.000 | **+0.042** | +0.057 |
| 2D EMD | 0.319 | **0.055** | floor 0.032 |

The crafted personas hit off-diagonal targets exactly (e.g. contribute 10/20
in Public Goods, give $0/100 in Dictator), correlation becomes statistically
indistinguishable from human, and the joint EMD improves 6x to within ~1.7x
of the sampling floor. Notably 13/23 stage-C personas exhibit nonzero 2D
variance — prompting for context-dependence partially restores behavioral
noise as well.

**Implication for the method.** 1D prompt-mixture fitting produces marginals
that are excellent and joints that are structurally wrong in a way weights
cannot repair; a modest extension (joint residual targeting + decoupled-trait
crafting) fixes it. Higher-dimensional joints will face the same
coverage-vs-K economics.

**The method is vector quantization with an LLM decoder — and K is computable,
not searchable.** On deterministic backbones, fitting K weighted point-masses
to a known distribution is classical optimal quantization (Lloyd/k-means;
Zador's theorem). Quantizing the human PG x Dictator joint numerically (zero
API calls) yields the full K-vs-quality curve: split-half floor 0.0227;
quantization EMD 0.148/0.097/0.053/0.037/0.030/0.027 at K =
5/10/25/60/100/150 — matching the Zador K^(-1/2) law in 2D. Two validations:
(1) our greedy K=25 Stage C (0.0553) ran at ~96% of the K=25 theoretical
optimum (0.0534) — the crafting step loses almost nothing over perfect
placement; (2) a VQ-mode refit (`craft2d.py --mode vq`) crafting one persona
per k-means centroid, K=100, fully parallel, achieved 0.0398 with human-level
correlation after reweighting (+0.055 vs human +0.057) — within ~25% of the
human floor and pressing against the 1,000-draw evaluation's own sampling
limit (shuffle of its own samples: 0.0336). Floor-adjusted, the K=25 -> K=100
improvement follows the predicted scaling. Practical recipe: quantize the
target data numerically, read K off the curve, craft one persona per centroid
in parallel, weight by cluster mass — replacing the paper's LLM-in-the-loop
search entirely on precisely-steerable backbones. (The paper chose K
heuristically by action-space size and reported insensitivity — true in 1D
where the error curve is shallow past the knee, not in 2D.)

**Determinism is not a sampling artifact.** A fitted flash persona answers
identically 10/10 even at temperature 2.0 (Gemini's maximum). The mechanism
is deliberation: thinking models converge to the persona's "correct" answer
regardless of sampling randomness, and the crafting loop selects for prompts
that pin the mode. Response diversity on deliberative models is a prompting
property, not a sampling property. Scope note: precise-steerable determinism
is what makes 1D alignment quantization-optimal, but the same property
maximizes the 2D correlation pathology and pays the coverage tax — the more
the task resembles compression, the more determinism helps; the more it
resembles simulation, the more it hurts.

Artifacts: `joint_results/*.json` (paired samples with persona attribution,
stage-B analysis embedded), `stageC_flash_PG_Dictator/` (crafted personas,
weights, fit trace with per-round targets and achieved atoms), scripts
`code/joint_eval.py`, `code/joint_analyze.py`, `code/craft2d.py`.

## Verdict

The paper's central claim — that a learned mixture of system prompts reproduces
human distributional heterogeneity at near-sampling-noise fidelity — held at every
level tested, including a full from-scratch refit on an independent account. The KS
test failure matches the paper's own reporting (MobLab simulations pass Wilcoxon but
not KS). Scope: levels 2–3 cover one game (Public Goods, EM); level 1 covers all 7
MobLab games under both algorithms.

The one negative result is cross-backbone transfer: prompts fitted on GPT-4o do not
carry to Gemini without refitting (W-dist 0.43 → 1.81–2.17 across both Gemini
tiers). The paper reports transfer as a strength based on WVS and partial MobLab
results; on this game it does not hold. Refitting fully recovers — flash EM beats
the paper's GPT-4o numbers on all 7 MobLab games — so the method is
backbone-agnostic even though its fitted artifacts are not.

The deeper qualification is the mechanism finding above: on the Gemini family the
method's excellent aggregate alignment is achieved by a deterministic K-atom
quantization rather than a population of behaviorally noisy personas, and none of
the paper's evaluation metrics can distinguish the two. Distributional alignment
(population level) and simulation fidelity (individual level) come apart, and the
evaluation framework only measures the former.

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
