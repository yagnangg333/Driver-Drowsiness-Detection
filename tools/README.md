# Evaluation tools

Scripts that evaluate and tune the system on the public datasets. They are not part of the installed package and are not needed to run the detector. All of them need the datasets in `data/` (see [`../data/README.md`](../data/README.md)) and are run from the repository root, for example `python tools/run_evaluation.py`.

The full workflow, with the results each step produces, is in [`../EVALUATION.md`](../EVALUATION.md#reproducing-the-results).

| Script | What it does | Reads | Writes |
|---|---|---|---|
| [`run_evaluation.py`](run_evaluation.py) | Runs the full pipeline over every recording; resumable. Options: `--only nthu yawdd uta nitymed`, `--workers N`, `--redo` | the datasets in `data/` | `data/eval/<dataset>/*.csv`, one per recording, every column of `FrameReport.to_row()` plus the labels |
| [`evaluation_report.py`](evaluation_report.py) | Turns those CSVs into the results tables (static vs adaptive) | `data/eval/` | stdout and `data/eval/results.md` |
| [`replay.py`](replay.py) | Offline simulator: re-runs indicators, fusion and alerts on the recorded signals with any settings. Used as a library by the tuning scripts; run directly, it checks that it matches the real pipeline | `data/eval/` | stdout |
| [`condition_features.py`](condition_features.py) | Samples frames and records the raw condition measurements (brightness, glare, contrast, asymmetry) | the datasets in `data/` | `data/condition_features.csv` |
| [`tune_conditions.py`](tune_conditions.py) | Chooses the low-light and occlusion thresholds on half of the videos and reports both halves | `data/condition_features.csv` | stdout |
| [`tune_detection.py`](tune_detection.py) | Scores detection settings on all four datasets with the simulator, leave-one-subject-out on NTHU-DDD | `data/eval/` | stdout |
| [`tune_smile.py`](tune_smile.py) | Chooses the smile margin relative to the resting smile, held out, with a safety check | `data/eval/` | stdout |
| [`learned_fusion.py`](learned_fusion.py) | Trains the logistic-regression fusion; leave-one-subject-out on NTHU-DDD and an external check on the other datasets. `--save PATH` saves the model | `data/eval/` | stdout (and the model) |
| [`learned_variants.py`](learned_variants.py) | Compares learned-fusion variants (features, regularisation, training data) with the rules; `--save models/learned_fusion.json` trains the model the dashboard shows | `data/eval/` | stdout (and the model) |

Shared code between scripts: `tune_detection.py`, `tune_smile.py` and `learned_fusion.py` import `replay.py`; `tune_smile.py` also imports `tune_detection.py`, and `learned_variants.py` imports `learned_fusion.py`. Everything else imports only the `drowsiness` package.
