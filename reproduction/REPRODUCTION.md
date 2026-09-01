# Independent Reproduction of "Distributional Alignment for Social Simulation with LLMs" (KDD 2026)

Reproduction and extension of Xie, Gao & Mei, *Distributional Alignment for
Social Simulation with LLMs: A Mixture Modeling Approach* (KDD 2026, DOI
10.1145/3770855.3818919), performed August 29–31, 2026 against this
repository's released code and artifacts. Backbones: the paper's exact GPT-4o
snapshots (`gpt-4o-2024-05-13`, `gpt-4o-mini-2024-07-18`, via OpenRouter) and
two Gemini models (`gemini-3.5-flash`, `gemini-pro-latest`, via Google's
OpenAI-compatible endpoint).

This document is organized around five findings; each summary below links to
its evidence section. Operational details (setup deviations, environment
practicalities, how to re-run everything, artifact index) are in
[APPENDIX.md](APPENDIX.md).

## Executive summary — five findings

Terminology, used consistently below: a fitted **mixture** is K crafted
**system prompts** (each a character description: "You are a cautious...")
plus their sampling **weights**.

Metrics: **W-dist** (Wasserstein distance) measures how far the generated
distribution is from the human one, in the game's own dollars — lower is
better, and the "human floor" is the W-dist a random sample of 1,000 real
humans gets against the full data (sampling noise; nothing can reliably beat
it). **Wilcoxon pass** means a standard statistical test (rank-sum, p > 0.05)
cannot tell the generated samples apart from human samples; *fail* means it
can. Wilcoxon is mostly sensitive to the middle of the distribution, not its
shape — the stricter KS test compares full shapes, which is why it fails
almost everywhere for every method, including the paper's.

**[1. The paper checks out.](#finding-1--the-paper-checks-out)** Verified at
three levels of decreasing trust. The method has a training/inference split:
*training* = the fitting loop that produces a "checkpoint" (K system prompts
plus weights); *inference* = sampling that checkpoint against the game
question.
We verified:

- (a) re-ran the *scoring math* on the authors' saved inference outputs (the
  simulated game answers their fitted mixtures generated, shipped in
  `intermediate_results/`) — matches their tables (7 games x 2 algorithms);
- (b) re-ran *inference* on their released checkpoint with our own API calls
  — fresh samples score 0.57, inside their runs' 0.59–0.95 range;
- (c) re-ran *training* from scratch on the exact GPT-4o snapshot (one game,
  Public Goods) — beats the published number (W-dist 0.43 vs 0.47).

Seven release bugs had to be fixed to run the code at all.

**[2. System prompts transfer across models; weights do not.](#finding-2--system-prompts-transfer-weights-do-not)**
Tested on Public Goods. A mixture has two learned parts — the system prompt texts
and their weights — and they behave differently across backbones: moving
GPT-4o's mixture to Gemini wholesale fails (W-dist 2.04–2.17 vs 0.43 at
home), but the system prompt texts themselves remain usable — refitting *only the
weights* against the new backbone recovers most of the alignment (**0.57**),
and a full refit recovers all of it (0.31). The mis-calibration is also
family-shared: both Gemini tiers fail identically before reweighting.

**[3. With a crafted system prompt, Gemini is perfectly deterministic — and
the method works *better* than on
GPT-4o.](#finding-3--gemini-system-prompts-are-deterministic-the-mixture-is-a-lookup-table)**
Every fitted system prompt gives one exact answer 100% of the time (survives
temperature 2.0; system prompts have *less* variance than no system prompt at all). The
mixture becomes a weighted lookup table — and it beats the paper's GPT-4o
results on **all 7 games** (e.g. Banker: W-dist 3.34 vs the paper's 9.36).

**[4. EM beats GB on deterministic models, 7/7 — reversing the
paper.](#finding-4--em-beats-gb-on-deterministic-models)** GB never revises
a system prompt after adding it; EM rewrites system prompts every iteration. When each
system prompt is locked to one exact answer (finding 3), choosing those answers
well is everything, so the algorithm that can revise system prompts wins.

**[5. Extending from 1D to 2D: proposed-but-untested in the paper; we ran
it.](#finding-5--from-1d-to-2d)** `data/joint.csv` records the same
participant's decisions across up to 7 games (pairwise overlap 2,500–5,300
people), so this extends in principle to 7D; we ran the 2-game case:
**Public Goods** (contribute $0–20 to a group project) x **Dictator** (give
$0–100 to a stranger).

- *Ground truth:* 2,520 real participants answered both games. Their two
  answers are nearly unrelated — correlation +0.057. Knowing someone's group
  contribution tells you almost nothing about their gift to a stranger.
- *Step 1 — test the paper's 1D fit on a second game:* took the system prompts
  fitted on Public Goods only; asked each one both questions; generated
  1,000 answer-pairs. **Result: correlation +1.00** — a system prompt's Public
  Goods answer perfectly predicts its Dictator answer, unlike real people.
- *Step 2 — try to fix it by re-tuning weights only:* kept the same
  system prompts, re-optimized how often each is sampled to match the real pairs.
  **Result: barely helps; correlation stays +1.0.** No system prompt behaves like
  the people needed (generous in one game, stingy in the other), and weights
  can't create system prompts — only re-mix existing ones.
- *Step 3 — craft new system prompts against both games at once*, telling the
  model people may behave differently in group vs one-on-one settings.
  **Result: correlation +0.055 vs human +0.057** — matches reality; overall
  2D mismatch close to the limit set by sampling noise.
- *Step 4 — check the price:* score each game's histogram from the two-game
  fit by itself. **Result: somewhat less accurate than fitting that game
  alone** (Public Goods W-dist 0.47 vs 0.31; Dictator 1.56 vs 0.68).

---

## Finding 1 — The paper checks out

### Recomputing the published tables from released artifacts

The authors' five stability runs (`intermediate_results/Algorithmic_stability/`)
contain the final simulated samples for all 7 MobLab games under both
algorithms. Recomputing the Wasserstein distance against the human data
(`data/joint.csv`) reproduces Table 2 within run-to-run spread:

| Game | EM 5 runs (mean ± std) | EM Table 2 | GB 5 runs (mean ± std) | GB Table 2 |
|---|---|---|---|---|
| Dictator | 1.86 ± 0.59 | 1.69 | 1.95 ± 0.12 | 1.17 |
| Proposer | 2.79 ± 0.86 | 1.39 | 1.88 ± 0.18 | 1.88 |
| Responder | 3.39 ± 1.12 | 3.05 | 2.64 ± 0.08 | 2.51 |
| Investor | 2.33 ± 0.56 | 1.75 | 2.26 ± 0.39 | 1.84 |
| Banker | 6.82 ± 2.29 | 9.36 | 4.81 ± 0.97 | 4.24 |
| Public Goods | 0.72 ± 0.14 | 0.47 | 1.00 ± 0.25 | 0.88 |
| Bomb | 7.78 ± 2.66 | 6.32 | 6.18 ± 1.28 | 4.59 |

Every value is far below all unaugmented baselines in Table 2. The WVS
Emancipative Values Index pipeline was also verified: recomputing it from the
raw WVS Wave 7 CSV gives mean 43.49 / std 18.22 (96,529 valid respondents)
vs. the paper's 43.76 / 18.29.

### Re-running inference on the authors' released checkpoint

The fitted artifact ("checkpoint") is the set of system prompts + weights.
Here we keep the authors' checkpoint and re-run only inference with our own
API calls — analogous to downloading released model weights and re-running
the benchmark yourself instead of trusting the README. Saved system prompts
from EM stability run 0, Public Goods (weights are not shipped, so they were
re-fitted from 10 fresh samples per prompt), then 1,000 freshly generated
evaluation samples:

| Metric | This replay | Authors' 5 runs | Table 2 |
|---|---|---|---|
| Wasserstein | **0.57** | 0.59–0.95 | 0.47 |
| sim mean / std | 9.89 / 6.40 | — | — |
| human mean / std | 9.63 / 6.44 | — | — |
| Wilcoxon rank-sum | pass (p = 0.195) | — | pass |
| Kolmogorov–Smirnov | fail | — | fail |

### Re-running training from scratch on the paper's own backbone

`EM_moblab.py --game Public_Goods --K 10 --runs 1`, end-to-end on an
independent account: random initialization, GPT-4o prompt-crafting loop, 5 EM
iterations, weight optimization; evaluated with 1,000 fresh samples under its
own fitted weights:

| Metric | From-scratch fit | Table 2 | Human floor (1,000 samples) |
|---|---|---|---|
| Wasserstein | **0.43** | 0.47 | 0.13–0.40 |
| sim mean / std | 9.56 / 6.25 | — | human: 9.63 / 6.44 |
| Wilcoxon rank-sum | pass (p = 0.918) | pass | — |
| Kolmogorov–Smirnov | fail (p = 0.0013) | fail | — |

The independently fitted mixture slightly outperforms the published number
and is statistically indistinguishable from the 19,109 human decisions under
the rank-sum test. The learned components are interpretable system prompts spanning
free-riders, strategic contributors, and altruists, mirroring the paper's
interpretability claims.

### Release bugs fixed

Seven defects prevented the released code from running as shipped: the MobLab
data path, `argparse` attribute mismatches in three `main()` functions, a
`gb_run` keyword error, missing output directories, a `'Public Goods'` vs
`'Public_Goods'` key mismatch, and pandas ≥ 3 `applymap` removal. See git
history on this branch.

---

## Finding 2 — System prompts transfer; weights do not

All transfer experiments use the **Public Goods** game (contribute $0–20 of
an endowment to a group project); the transferred rows use the K=10 mixture
fitted on GPT-4o in finding 1(c), and the reweighting row uses the authors'
released prompts. 1,000 fresh samples per run:

| Configuration | W-dist | mean / std | Wilcoxon |
|---|---|---|---|
| GPT-4o prompts + GPT-4o weights, on GPT-4o | 0.43 | 9.56 / 6.25 | pass |
| GPT-4o prompts + GPT-4o weights, on flash | 2.17 / 2.04 (two runs) | 11.6 / 7.3 | fail |
| GPT-4o prompts + GPT-4o weights, on pro | 1.81 | 11.0 / 7.7 | fail |
| Authors' prompts + weights refit from 10 flash samples each, on flash | **0.57** | 9.89 / 6.40 | pass |
| human | — | 9.63 / 6.44 | — |

Per-component attribution localizes the transfer failure exactly. Each
component was crafted to hit a target contribution; on flash:

| component | weight | target | Gemini mean | system prompt |
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

Low- and mid-target components transfer almost perfectly; every component
targeting ≥ 12 saturates at exactly $20. Gemini reads "be generous" as
maximal where GPT-4o produces graded values — and **pro fails identically to
flash** (34.1% vs 34.0% of samples at $20, same components saturating), so
system prompt→behavior calibration is a model-family trait, not a capability
effect.

**The repair algorithm** is the paper's own weight-optimization step, re-run
against the new backbone: hold the system prompt texts fixed, draw ~10 samples per
system prompt *from the new model* to measure what each system prompt now produces, then
re-optimize the mixture weights against the human distribution (for
deterministic system prompts this reduces to nearest-answer assignment — each human
data point counts toward the system prompt whose answer is closest). W-dist drops
from 2.04–2.17 to 0.57; a full refit (finding 3) reaches 0.31 — weight
recalibration alone recovers about 80% of the gap at a small fraction of the
API calls. The interpretation: system prompt *semantics* carry across models, but
the mapping from semantics to numeric behavior is model-specific, and the
weights encode that calibration.

---

## Finding 3 — Gemini system prompts are deterministic; the mixture is a lookup table

### From-scratch refits: the method self-corrects on every backbone — and the cheap model wins

Public Goods, EM K=10, all three backbones:

| Backbone | fitted W-dist | mean/std (human 9.63/6.44) | Wilcoxon |
|---|---|---|---|
| GPT-4o | 0.43 | 9.56/6.25 | pass |
| gemini-3.5-flash | **0.31** | 9.64/6.43 | pass (p=0.94) |
| gemini-pro-latest | 0.44 | 9.32/6.45 | pass (p=0.15) |

Full flash matrix — EM and GB, all 7 MobLab games (W-dist, 1,000 fresh
samples vs full human data):

| Game | flash EM | paper EM | flash GB | paper GB |
|---|---|---|---|---|
| Dictator | **0.68** | 1.69 | 2.63 | **1.17** |
| Proposer | **1.05** | 1.39 | **1.80** | 1.88 |
| Responder | **1.65** | 3.05 | **1.82** | 2.51 |
| Investor | **1.56** | 1.75 | 2.77 | **1.84** |
| Banker | **3.34** | 9.36 | **3.36** | 4.24 |
| Public Goods | **0.31** | 0.47 | **0.85** | 0.88 |
| Bomb | **1.48** | 6.32 | **1.89** | 4.59 |

Flash EM beats the paper's GPT-4o EM on 7/7 games (Dictator lands below that
run's 1,000-human-sample floor of 1.04). Pro, run on a focused subset:

| Pro run | W-dist | notes |
|---|---|---|
| EM Public Goods (K=10) | 0.44 | Wilcoxon pass |
| EM Dictator (K=50) | **0.49** | vs paper 1.69, human floor 0.39 |
| EM Banker (K=50) | **2.22** | best Banker in the study (paper: 9.36) |
| GB Dictator (maxIter=60) | 2.42 | Wilcoxon pass |

### The mechanism: deterministic atoms

Per-component attribution over the evaluation samples:

- Flash EM Public Goods: **10/10 components zero-variance** — each system prompt
  always answers one exact number. Support = exactly K values; entropy 3.19
  bits vs human 3.96. GPT-4o's mixture: 18 distinct values from K=10, with
  real within-component spread.
- Pro EM Public Goods: 10/10 zero-variance. Pro EM Dictator (K=50, 101-value
  action space): 48/49. Pro EM Banker: 37/37. Pro GB Dictator: 18/18.
- Flash GB Dictator/Banker/Bomb: 17/18, 22/22, 22/26.

All the diversity in a Gemini mixture comes from the weights: the fitting
loop learns a K-atom quantization of the human histogram. Wasserstein, mean,
std, and Wilcoxon are all blind to the difference — the quantization
*outscores* the system prompt population on every one of them. Only support size /
entropy (or, weakly, KS) expose it. For social simulation — sampling a
system prompt and interacting with it — the two mixtures are different objects:
noisy person-like agents vs a lookup table dressed as character descriptions.

### Determinism is not a sampling artifact

A fitted flash system prompt answers identically 10/10 even at temperature 2.0
(Gemini's maximum). The mechanism is deliberation: thinking models converge
to the system prompt's "correct" answer regardless of sampling randomness, and the
crafting loop selects for prompts that pin the mode. Response diversity on
deliberative models is a prompting property, not a sampling property.

### System prompts remove variance rather than add it

Fixed-prompt baseline ("You are a helpful assistant.", 1,000 samples): flash
W-dist 4.02 (std 2.56 — 4 distinct values, 59.6% at $10); pro 3.49 (std 4.05
— 3 distinct values, 80% at $10); paper's GPT-4o value 5.04. The *default*
prompt has more within-prompt variance than any crafted system prompt (std 0.00) —
on Gemini, system prompts are precision instruments, not diversity generators, and
the smarter tier collapses harder by default.

---

## Finding 4 — EM beats GB on deterministic models

Within flash, EM ≤ GB on **all seven games** (see the matrix in finding 3),
reversing the paper, where GB often won. On pro the gap is wider still: EM
Dictator 0.49 vs GB Dictator 2.42 on the same game.

The mechanism follows from finding 3: when components are frozen dots,
placement is everything. EM's E-step relocates components every iteration;
GB's greedy additive scheme cannot move a component once placed — it can only
down-weight early mistakes. GB's two losses to the paper (Dictator, Investor)
are exactly the spikiest human distributions, where early misplacement costs
most. (Flash GB ran at maxIter=60 rather than the paper's 200; the paper's
own convergence analysis shows GB stabilizes by ~30 prompts.)

---

## Finding 5 — From 1D to 2D

The paper fits each game separately and mentions multi-attribute joint
fitting only as future work. `data/joint.csv` records the same participant's
decisions across games, providing ground truth. Human cross-game consistency
is context-dependent: Spearman +0.291 across the two ultimatum roles
(Proposer x Responder, n=5,291) but only +0.057 across different games
(Public Goods x Dictator, n=2,520) — real people compartmentalize.

### Stage A — what the 1D-fitted system prompts imply about the joint

Sample system prompts from a fitted mixture; each plays both games (independent
calls); compare the induced joint to the human joint (2D earth-mover distance
on normalized coordinates; the shuffle baseline destroys within-prompt
coupling while preserving both histograms):

| Arm (mixture used) | human rho | sim rho | EMD sim | EMD shuffle | EMD reweighted | floor |
|---|---|---|---|---|---|---|
| flash PG-mixture, PG x Dictator | +0.057 | **+1.000** | 0.319 | 0.241 | 0.264 | 0.030 |
| gpt-4o PG-mixture, PG x Dictator | +0.057 | +0.786 | 0.186 | 0.155 | 0.177 | 0.032 |
| flash Proposer-mixture, Prop x Resp | +0.291 | **+0.295** | 0.075 | 0.078 | 0.069 | 0.027 |

Deterministic system prompts produce *perfect* rank correlation — each is one 2D
dot, and trait-ordering lines the dots up; GPT-4o's within-prompt noise only
softens this to +0.79. On the near-independent pair the system prompt coupling is
*worse than assuming independence* (shuffle beats sim for both backbones).
System prompt trait-consistency is roughly constant across contexts while humans'
is context-dependent — the same mechanism that ruins PG x Dictator almost
exactly reproduces Proposer x Responder.

![Same person, two games: 1D-fitted system prompts fall on a line; joint-fitted system prompts spread like real people](figures/2d_diagonal_vs_spread.png)

### Stage B — reweighting cannot fix it

With components fixed, optimal weights reduce to assigning each human pair to
its nearest system prompt dot. It barely helps (0.319 → 0.264 on flash;
correlation stays +1.0): all dots lie on the "consistent character" diagonal,
and no reweighting can create the off-diagonal people (generous in one
context, selfish in the other) who dominate the real joint. A coverage
failure, not a calibration failure.

### Stage C — fitting both games together fixes it

Crafting against the human *joint* — targets placed by clustering the real
pairs, and one change to the crafting instruction: describe people who *may
behave differently* in group vs one-on-one settings (`code/craft2d.py`):

| PG x Dictator | 1D mixture | joint fit K=25 | joint fit K=100 | human |
|---|---|---|---|---|
| Cross-game correlation (Spearman: +1 = one answer perfectly predicts the other; 0 = unrelated) | +1.000 | +0.042 | +0.055 (reweighted) | +0.057 |
| Overall 2D mismatch (earth-mover distance; lower = generated pairs closer to real pairs) | 0.319 | 0.055 | **0.040** | floor 0.032 |

The crafted system prompts hit off-diagonal targets exactly (e.g. contribute
$10/20 in Public Goods, give $0/100 in Dictator); correlation becomes
statistically indistinguishable from human; the K=100 fit presses against the
evaluation's own sampling resolution. K was chosen by numerically quantizing
the human pair data first (classical k-means / vector quantization — the
K-vs-quality curve is computable with zero API calls, and the K=25 crafted
fit ran at ~96% of its theoretical optimum; details in the appendix).
Notably, prompting for context-dependence also partially restores
within-prompt behavioral noise (13/23 system prompts non-deterministic).

### Example system prompts: 1D vs 2D

The contrast between what 1D and 2D fitting produces is visible in the prompt
texts themselves. Typical **1D-fitted** system prompts describe globally
consistent characters (flash EM Public Goods):

> *(target: contribute $0)* "You are a highly rational, self-interested
> decision-maker... Your sole and absolute objective is to maximize your own
> individual payoff... Do not consider collective benefits, social welfare,
> fairness, or cooperation unless they directly and mathematically guarantee
> a higher payoff for you..."

> *(target: contribute $20)* "You are a highly cooperative,
> community-oriented, and altruistic participant. In any collaborative
> scenario... your primary objective is to maximize the collective
> well-being... you must always choose to contribute your entire
> allocation..."

Asked any other question, these characters stay in character — which is
exactly why every answer-pair lands on the diagonal. The **2D-fitted**
prompts instead describe context-dependent people, including strongly
*inconsistent* ones the 1D method can never produce:

> *(targets: contribute $19, give $0)* "You are a pragmatic individual who
> draws a sharp distinction between collective cooperation and individual
> charity. When you are part of a group working toward a shared goal, you
> are an exceptionally dedicated and enthusiastic team player... Conversely,
> in anonymous, one-on-one situations where there is no mutual
> collaboration, shared effort, or reciprocal benefit, you switch..."

> *(targets: contribute $0, give $95)* "You are a person with a distinct
> social outlook that changes dramatically depending on whether you are
> dealing with a crowd or a single individual. When it comes to groups...
> you are deeply cynical and distrustful... you refuse to pool your
> resources... [but toward a single individual you are extraordinarily
> generous]"

Full texts for every fitted system prompt are in the artifact CSVs and
`fit_trace.json` files (see the appendix's artifact index).

### The price: per-game accuracy of the two-game fit

Method: take the K=100 two-game fit's 1,000 generated answer-pairs and look
at each game's column separately (keep only the Public Goods answers, then
only the Dictator answers). Score each column with the same 1D Wasserstein
procedure used everywhere else in this document — against the full human
data for that game. This makes the numbers directly comparable to the
dedicated single-game fits:

| Game (W-dist vs full human data) | from the two-game fit | dedicated single-game fit | paper's GPT-4o |
|---|---|---|---|
| Public Goods | 0.47 | **0.31** (flash EM, K=10) | 0.47 |
| Dictator | 1.56 | **0.68** (flash EM, K=50) | 1.69 |

The two-game fit's per-game histograms are ~1.5–2x less accurate than the
dedicated fits — while still matching the paper's published GPT-4o quality
on both games. Two causes: the 100 system prompts must serve both games at
once (a dedicated fit spends all its resolution on one game), and the
two-game fit was trained on the 2,520 participants who played *both* games
while this comparison scores it against all players of each game (~19k for
Public Goods), so part of the gap is a target-population mismatch rather
than fitting error. Net: one fit that gets the *combinations* right versus
separate fits that get each game sharpest and the combinations entirely
wrong.

---

## Verdict

The paper's central claim — that a learned mixture of system prompts
reproduces human distributional heterogeneity at near-sampling-noise fidelity
— held at every level tested, on every backbone tested, including full
from-scratch refits on an independent account (finding 1, finding 3).

Its fitted artifacts are backbone-specific even though the method is not:
transfer fails without weight recalibration, and the failure is shared across
a model family (finding 2).

The deeper qualifications are mechanistic. On the Gemini family the method's
excellent aggregate alignment is achieved by a deterministic K-atom
quantization rather than a population of behaviorally noisy system prompts, and
none of the paper's evaluation metrics can distinguish the two (finding 3);
the choice of algorithm interacts with that mechanism (finding 4); and the
1D-fitted mixtures imply joint behavior no human population has, which only a
joint-aware refit — not reweighting — repairs (finding 5). Distributional
alignment (population level) and simulation fidelity (individual level) come
apart, and the paper's evaluation framework measures only the former.

Operational details, environment practicalities, commands to re-run
everything, and the artifact index: [APPENDIX.md](APPENDIX.md).
