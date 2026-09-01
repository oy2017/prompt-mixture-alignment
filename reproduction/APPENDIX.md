# Appendix — Operational Details

Supporting material for [REPRODUCTION.md](REPRODUCTION.md): setup deviations,
environment practicalities, methodology notes, commands to re-run everything,
and the artifact index.

## Deviations from the original setup

- **API route**: OpenRouter (GPT-4o snapshots) and Google's OpenAI-compatible
  endpoint (Gemini) instead of Azure OpenAI — same model snapshots for
  GPT-4o. Neither route honors `n>1`, so multi-sample requests were rewritten
  as parallel single-sample requests (`play()` in `code/EM_moblab.py` /
  `GB_moblab.py`); behavior identical.
- **Level-2 replay weights**: the authors' repo does not ship fitted weights,
  so the replay re-fitted them with a simplified objective (plain Wasserstein
  + small L2 regularizer, no KS discount term) from 10 samples per prompt.
- **GB iteration budget**: flash/pro GB runs used maxIter=60 rather than the
  paper's 200; the paper's own convergence analysis shows GB stabilizes by
  ~30 prompts.
- **Backbone selection**: `PMA_PROVIDER` env var chooses the provider/model
  pair (see `PROVIDERS` in `code/providers.py`); `PMA_INIT_WORKERS`
  parallelizes EM initialization across targets.

## Gemini practicalities

**Token-budget trap.** `gemini-3.5-flash` is a thinking model: hidden
reasoning tokens are charged against `max_tokens`, while
`usage.completion_tokens` reports only the visible output (median 91, max 212
here). Sizing the cap from that reported number silently truncates almost
every response *before* it reaches the bracketed answer. Measured over 50
probes per cap (240 at 2,000):

| generation cap | truncated | usable answer |
|---|---|---|
| 250 | 98% | 0% |
| 350 | 100% | 0% |
| 500 | 96% | 4% |
| **2,000** | **0/240** | **100%** |
| 8,000 | 0/40 | 100% |

2,000 is sufficient and 8,000 adds nothing; `gemini-pro-latest` runs with a
4,000 cap. The extraction fallback shares the same budget for the same
reason.

**Quota handling.** The pro model carries a hard daily quota; requests can
also be rate-limited within a day. `providers.api_call` retries transient
errors with exponential backoff and sleeps through daily-quota exhaustion
(resuming at the midnight-PT reset), so multi-hour fits and evaluations run
unattended across quota cycles.

## Methodology notes

**The K-vs-quality curve (finding 5's K choice).** On deterministic
backbones, fitting K weighted point-masses to a known distribution is
classical optimal quantization (Lloyd/k-means; Zador's theorem). Quantizing
the human PG x Dictator joint numerically (zero API calls) yields:
split-half floor 0.0227; quantization EMD 0.148 / 0.097 / 0.053 / 0.037 /
0.030 / 0.027 at K = 5 / 10 / 25 / 60 / 100 / 150 — matching the Zador
K^(-1/2) law in 2D. Validations: the greedy K=25 stage-C fit (0.0553) ran at
~96% of the K=25 theoretical optimum (0.0534), and the VQ-mode K=100 refit
(`craft2d.py --mode vq`: k-means centroids up front, system prompts crafted in
parallel) achieved 0.0398 — within ~25% of the human floor and pressing
against the 1,000-draw evaluation's own sampling limit (shuffle of its own
samples: 0.0336). Practical recipe: quantize the target data numerically,
read K off the curve, craft one system prompt per centroid in parallel, weight by
cluster mass. (The paper chose K heuristically by action-space size and
reported insensitivity — true in 1D where the error curve is shallow past
the knee, not in 2D.)

**Compression vs simulation scope note.** Precise-steerable determinism is
what makes 1D alignment quantization-optimal, but the same property maximizes
the 2D correlation pathology and pays the coverage tax — the more the task
resembles compression, the more determinism helps; the more it resembles
simulation, the more it hurts.

## Reproducing this reproduction

```bash
pip install pandas numpy scipy tqdm matplotlib openai pot
export OPENROUTER_API_KEY=...      # for the GPT-4o arms
export GEMINI_API_KEY=...          # for the Gemini arms
cd code

# Finding 1 — replay authors' prompts, then fit from scratch on GPT-4o:
python replay_eval.py --alg EM --game Public_Goods --run 0
python EM_moblab.py --game Public_Goods --K 10 --runs 1
python replay_eval.py --alg EM --game Public_Goods \
    --prompts-csv Public_Goods/1_result/4_investor_EM_prompts_updated.csv \
    --weights-pkl Public_Goods/1_weights_lst.pkl --tag level3_EM_Public_Goods_run1

# Findings 2–4 — Gemini transfer, full sweep (EM+GB, all games), baselines:
PMA_PROVIDER=gemini python replay_eval.py --alg EM --game Public_Goods \
    --prompts-csv ../reproduction/level3_EM_Public_Goods/1_result/4_investor_EM_prompts_updated.csv \
    --weights-pkl ../reproduction/level3_EM_Public_Goods/1_weights_lst.pkl \
    --tag crossbackbone_gemini_Public_Goods
PMA_PROVIDER=gemini ./run_gemini_sweep.sh
python prepare_gb_eval.py ../reproduction/level3_gemini_GB_<game>   # then replay_eval per game

# Finding 5 — 2D stages A/B/C:
PMA_PROVIDER=gemini python joint_eval.py --game-a Public_Goods --game-b Dictator \
    --prompts-csv <mixture prompts csv> --weights-pkl <weights pkl> --tag <tag>
python joint_analyze.py ../reproduction/joint_results/<tag>.json
PMA_PROVIDER=gemini python craft2d.py --mode vq --rounds 100 --outdir <dir>
```

Recomputing the finding-1 artifact tables needs no API access — the snippets
live in this branch's git history and read only `intermediate_results/` and
`data/joint.csv`.

## Artifact index

- **Fits** — full EM/GB trajectory (initialization prompts, per-iteration
  allocations/updates, weight history) under
  `level3_<provider>_<alg>_<game>/`; 2D fits under
  `stageC_flash_PG_Dictator/` and `stageC_vq100_flash_PG_Dictator/` (crafted
  system prompts, weights, fit trace with per-round targets and achieved atoms).
- **Evaluations** — prompts, weights, all 1,000 samples with per-component
  attribution (`eval_prompt_idx`), provider/model IDs, and metrics in
  `replay_results/*.json` + flat `*_samples.csv`; 2D paired samples with
  embedded stage-B analysis in `joint_results/*.json`.
- **Baselines** — the fixed-prompt baseline as a prompts-csv + weights-pkl
  pair under `baseline/`, runnable through `replay_eval.py` unchanged.
- **Figures** — `figures/2d_diagonal_vs_spread.png`.
- **Scripts** — `code/replay_eval.py` (evaluation harness),
  `code/prepare_gb_eval.py` (GB weight parser), `code/joint_eval.py` /
  `code/joint_analyze.py` (2D stages A/B), `code/craft2d.py` (2D crafting,
  greedy + VQ modes), `code/providers.py` (backbone selection + quota-aware
  retry), `code/run_gemini_sweep.sh` (full-matrix driver).
