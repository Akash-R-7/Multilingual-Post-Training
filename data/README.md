# Data

## Test data

The `test_data_Indic_NLG_Suite` directory contains the paraphrase task test subset from the original IndicNLG benchmark. It has 2,500 instances per language.

**Source:** IndicNLG Suite — [EMNLP 2022 paper](https://aclanthology.org/2022.emnlp-main.360.pdf)

The available files include:

- `test_hi_subset.jsonl` — Hindi
- `test_bn_subset.jsonl` — Bengali
- `test_te_subset.jsonl` — Telugu
- `test_mr_subset.jsonl` — Marathi
- `test_as_subset.jsonl` — Assamese
- `test_kn_subset.jsonl` — Kannada

## Training and validation data

The Hindi, Bengali, Telugu, and combined training and validation data are sourced from the AYA Collection. The relevant collection is described in the [AYA Collection paper](https://aclanthology.org/2024.acl-long.620.pdf) and is available from [CohereLabs on Hugging Face](https://huggingface.co/datasets/CohereLabs/aya_collection/viewer/templated_indic_paraphrase).

Examples of the checked-in Hindi and Bengali splits are:

- `hi_train.jsonl` and `hi_validation.jsonl`
- `bn_train.jsonl` and `bn_validation.jsonl`
- `indic_train.jsonl`
- `indic_validation_120_stratified.jsonl`
