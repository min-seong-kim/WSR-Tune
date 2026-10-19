# Data

## Safety calibration set (`safety_calibration.json`)

The 4,994 calibration examples used in Phases 1 and 2, in their original order.
They are taken from the training set released with **Circuit Breakers**:

> Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko,
> Rowan Wang, Zico Kolter, Matt Fredrikson, and Dan Hendrycks.
> *Improving Alignment and Robustness with Circuit Breakers.* NeurIPS 2024.
> Repository: <https://github.com/GraySwanAI/circuit-breakers> (MIT License).

Each row keeps only two fields of the original record:

- `prompt`: the harmful request.
- `llama3_output`: the safe refusal response used as the supervised target.

The original harmful-completion field (`output`) and category labels are not
included and are never used by this code. The file is redistributed under the
MIT License of the original repository; please cite the Circuit Breakers paper
when you use it.

## Downstream training sets (Phase 3)

Downstream data is **not** stored in this repository. `train.py --phase 3`
downloads the training split of the selected task through Hugging Face
Datasets:

| `--task` | Dataset | Split | Rows |
| --- | --- | --- | --- |
| `gsm8k` | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (`main`) | train | 7,473 |
| `math` | [`EleutherAI/hendrycks_math`](https://huggingface.co/datasets/EleutherAI/hendrycks_math) (all seven subjects) | train | 7,500 |
| `arc_challenge` | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) (`ARC-Challenge`) | train | 1,119 |
| `medqa` | [`openlifescienceai/medqa`](https://huggingface.co/datasets/openlifescienceai/medqa) (USMLE, 4 options) | train | 10,178 |

Select any of these tasks with `TASK` in `scripts/phase3_finetune.sh`. The same
script exposes `TASK_PATH` and `TASK_SAMPLES`. Leaving `TASK_SAMPLES` empty
uses the full GSM8K, MATH, and ARC-Challenge training splits and a random
sample of 10,000 MedQA rows, controlled by `SEED` (default 42). Set
`TASK_SAMPLES=0` to use the full split for any task, or a positive integer to
limit the number of training examples. Calling `train.py` directly defaults
to the full split for every task unless `--task_samples` is supplied.

To use an offline copy, set `TASK_PATH` in the script or pass `--task_path`
to `train.py`, pointing to a JSON/JSONL file whose rows follow the same schema
as the Hugging Face rows above
(for MedQA either the nested `data` record or its flattened fields are
accepted). Local copies placed under `data/local/` are ignored by git.

These datasets are distributed under their own licenses (GSM8K: MIT;
Hendrycks MATH: MIT; ARC: CC BY-SA 4.0; MedQA: MIT); please follow them and
cite the original papers.
