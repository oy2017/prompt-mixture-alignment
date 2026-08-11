# Prompt Mixture Alignment

This repository contains experiment code, data, appendix material, and intermediate outputs for distributional alignment with LLMs experiments. The central idea is to generate and combine multiple system prompts so model response distributions better match target human-response distributions.

## Repository Index

| Path | Contents |
| --- | --- |
| [`code/`](code/) | Python experiment drivers for EM and gradient boosting algorithms for distributional alignment with LLMs. |
| [`data/`](data/) | Input data used by the experiment scripts. |
| [`appendix/`](appendix/) | Additional methodological details, data sources and processing pipeline, experiment details, and supplementary experiment results PDFs. |
| [`intermediate_results/`](intermediate_results/) | Saved experiment outputs for stability, sensitivity, and backbone-comparison analyses. |

## Code Entry Points

| Script | Purpose |
| --- | --- |
| [`code/EM_bigfive.py`](code/EM_bigfive.py) | EM algorithm for distributional alignment with LLMs for Big Five personality-trait score distributions. |
| [`code/GB_bigfive.py`](code/GB_bigfive.py) | Gradient boosting algorithm for distributional alignment with LLMs for Big Five experiments. |
| [`code/EM_moblab.py`](code/EM_moblab.py) | EM algorithm for distributional alignment with LLMs for MobLab economic-game behavior distributions. |
| [`code/GB_moblab.py`](code/GB_moblab.py) | Gradient boosting algorithm for MobLab economic-game behavior distributions. |
| [`code/EM_wvs.py`](code/EM_wvs.py) | EM algorithm for distributional alignment with LLMs for World Values Survey experiments. |
| [`code/GB_wvs.py`](code/GB_wvs.py) | Gradient boosting algorithm for World Values Survey experiments. |

## Main Data And Artifacts

- [`data/joint.csv`](data/joint.csv) is the MobLab dataset file.
- [`appendix/`](appendix/) contains supplementary materials.
- [`intermediate_results/Algorithmic_stability/`](intermediate_results/Algorithmic_stability/) stores outputs from five independent random-seed runs for algorithmic stability checks.
- [`intermediate_results/Meta_prompt_sensitivity/`](intermediate_results/Meta_prompt_sensitivity/) stores meta-prompt template sensitivity outputs.
- [`intermediate_results/%20Cross_model_backbone_sensitivity/`](intermediate_results/%20Cross_model_backbone_sensitivity/) stores cross-model backbone sensitivity outputs.

