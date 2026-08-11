# Data Index

This directory stores input data for the prompt-mixture alignment experiments.

## Files

| File | Contents |
| --- | --- |
| [`joint.csv`](joint.csv) | Main uploaded data table used by the experiments. |

## External Datasets To Download Manually

The repository does not include every raw source dataset. Download the following datasets manually and place the expected files in this `data/` folder before running the related experiments.

| Dataset | Source | Place in `data/` | Used by |
| --- | --- | --- | --- |
| Big Five / OCEAN personality-test responses | [Kaggle: OCEAN Five Factor Personality Test Responses](https://www.kaggle.com/datasets/lucasgreenwell/ocean-five-factor-personality-test-responses?select=data.csv) | Download `data.csv` and place it at `data/data.csv` unless the script path is changed. | [`../code/EM_bigfive.py`](../code/EM_bigfive.py), [`../code/GB_bigfive.py`](../code/GB_bigfive.py) |
| World Values Survey Wave 7 | [World Values Survey WV7 documentation](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp) | Download the WV7 data file from WVS and place the raw or processed file here using the filename expected by the WVS scripts. | [`../code/EM_wvs.py`](../code/EM_wvs.py), [`../code/GB_wvs.py`](../code/GB_wvs.py) |

