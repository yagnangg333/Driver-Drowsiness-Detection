# Evaluation

How the system was evaluated, the results, what the evaluation changed in the design, and how to reproduce every number. The system itself is described in [ARCHITECTURE.md](ARCHITECTURE.md).

## Contents

- [Datasets](#datasets)
- [Method](#method)
- [Results](#results)
- [Where each number comes from](#where-each-number-comes-from)
- [Condition thresholds](#condition-thresholds)
- [What the evaluation changed](#what-the-evaluation-changed)
- [Learned fusion](#learned-fusion)
- [Limitations](#limitations)
- [Reproducing the results](#reproducing-the-results)

## Datasets

Four public datasets, 489 recordings and about 630,000 frames. They are not in the repository; [`data/README.md`](data/README.md) explains where to get them and where to put them.

| Dataset | Content used | Ground-truth labels | Used to measure |
|---|---|---|---|
| **NTHU-DDD** (Kaggle copy) | 18,000 infrared frames (7 recordings of 4 subjects; 9,000 drowsy, 9,000 not), acted drowsiness, with and without glasses | drowsy / not drowsy **per frame** | frame-level F1, false alarms, ROC-AUC, detection delay |
| **YawDD** | 320 day-driving "Mirror" clips (105 normal, 100 talking, 115 yawning or talking-and-yawning), with clear glasses and sunglasses; 319 scored (one 10-second yawning clip never finished calibration) | normal / talking / yawning **per clip** | false warnings while talking; yawn detection |
| **UTA-RLDD** (fold 1) | 36 ten-minute videos of 12 people, **real** tiredness | alert (0) / low vigilant (5) / drowsy (10) **per video** | whether drowsy videos score higher than alert ones |
| **NITYMED** | 126 real night-driving videos (19 microsleep, 107 yawning) | yawning / microsleep **per video** | night-time detection |

## Method

- **Identical input for both strategies.** Each recording is processed once by the full pipeline, which writes both the static and the adaptive score for every frame. The alert state machine is then replayed over each score, so the two strategies are compared on exactly the same frames, landmarks and indicator scores.
- **Frame-level metrics (NTHU-DDD).** A frame counts as predicted drowsy when its alert level is ≥ 1 (warning level) or ≥ 2 (alarm level). Reported: precision, recall, F1, the **false-alarm rate** FP / (FP + TN) (the share of alert frames with an alert) and the ROC-AUC of the continuous score (Mann–Whitney).
- **Detection delay.** Time from the start of a labelled drowsy episode to the first alert. Only episodes that start without an alert already on are counted (23 of 41); the others are continuations of an earlier episode.
- **Clip and video level.** YawDD: the share of clips with a warning. UTA-RLDD: AUC of each video's mean score, drowsy vs alert. NITYMED: the share of microsleep videos raising an alarm, and yawns detected per video.
- **Held-out validation.** Every setting was chosen on part of the data and checked on data it was not chosen on: leave-one-subject-out on NTHU-DDD (chosen on 3 subjects, scored on the 4th, predictions pooled), or half of the drivers or videos elsewhere. Splits are always by subject or video, never by frame, because neighbouring frames are nearly identical.
- **Offline replay.** [`tools/replay.py`](tools/replay.py) re-runs everything after face tracking on the recorded signals, in seconds instead of hours. It matches the real pipeline on 98.4–99.98% of frames. Candidate settings were explored with it. The final results come from the real pipeline, except the **held-out** results and the **learned-fusion** comparison: these need different settings or models for each held-out subject, so they are computed with the simulator on the recorded signals (see [where each number comes from](#where-each-number-comes-from)).

## Results

Final settings, full pipeline, static vs adaptive on identical frames:

| Dataset | What it measures | Static | Adaptive |
|---|---|---|---|
| **NTHU-DDD** | F1 at warning / alarm level | 76.6% / 71.8% | **80.9% / 76.4%** |
| | false-alarm rate at warning / alarm level | 34.8% / 22.4% | **23.1% / 10.7%** |
| | recall at warning / alarm level | 83.6% / 68.5% | 83.6% / 68.5% |
| | ROC-AUC of the fused score | 0.823 | **0.883** |
| | **held out:** F1 / false alarms, smile settings chosen without the scored subject | 76.5% / 35.0% | **82.7% / 18.4%** |
| | drowsy episodes detected / median delay | 82.6% / 1.6 s | 82.6% / 1.6 s |
| **YawDD** | talking clips with a (false) warning | 31% | **16%** |
| | normal-driving clips with a (false) warning | 11.4% | **9.5%** |
| | yawning clips with a yawn detected | 68–77% | 68–77% |
| **UTA-RLDD** | AUC drowsy vs alert video (mean score; 12 drowsy vs 12 alert videos) | 0.868 | 0.861 |
| | subjects whose drowsy video scores higher | 12/12 | 12/12 |
| **NITYMED** | microsleep videos raising an alarm | 94.7% (18/19) | 94.7% (18/19) |
| | yawns detected per video (3 performed) | 2.34 | 2.34 |

Adaptive fusion keeps every detection of the static baseline on NTHU-DDD (same recall) while cutting false alarms by a third at warning level and by half at alarm level. The held-out row is the fairest estimate: for each subject the smile settings were chosen on the other three, while the static baseline has nothing tuned on NTHU-DDD. Night detection and the ranking of drowsy vs alert videos are unchanged; the small UTA-RLDD AUC difference (36 videos) changes no subject's ranking.

## Where each number comes from

All results are from the final evaluation run (21 October 2026). Warning-level and alarm-level results use the same frames, the same labels and the same metric code ([`binary_metrics`](src/drowsiness/evaluation.py)); only the cut-off differs: a frame counts as predicted drowsy at alert level ≥ 1 (warning) or ≥ 2 (alarm).

| Number | Dataset and split | Definition | Computed by |
|---|---|---|---|
| Recall 83.6% (warning), 68.5% (alarm) | NTHU-DDD, all 18,000 labelled frames | TP / (TP + FN); TP = 7,527, FN = 1,473 at warning level, for both strategies | real pipeline, [`evaluation_report.py`](tools/evaluation_report.py) |
| False alarms 34.8% → 23.1% (warning), 22.4% → 10.7% (alarm) | NTHU-DDD, all frames (9,000 not drowsy) | FP / (FP + TN); at warning level FP = 3,136 → 2,081 | real pipeline, `evaluation_report.py` |
| F1 76.6% → 80.9% (warning), 71.8% → 76.4% (alarm) | NTHU-DDD, all frames | 2TP / (2TP + FP + FN) | real pipeline, `evaluation_report.py` |
| ROC-AUC 0.823 → 0.883 | NTHU-DDD, all frames | probability that a drowsy frame's fused score exceeds a non-drowsy frame's (Mann–Whitney, ties count half) | real pipeline, `evaluation_report.py` |
| **Held out:** F1 76.5% → 82.7%, false alarms 35.0% → 18.4% (warning) | NTHU-DDD, leave-one-subject-out: the smile settings for each subject are chosen on the other three; the 4 test folds are pooled. Recall 83.6% → 83.5% | same as above | offline simulator, [`tune_smile.py`](tools/tune_smile.py) `held_out()` |
| Median delay 1.6 s, 82.6% of episodes | NTHU-DDD, warning level; 23 drowsy episodes that start without an alert already on; 19 detected | time from the first drowsy frame of an episode to the first alerted frame | real pipeline, `evaluation_report.py` |
| Talking clips warned 31% → 16%; normal clips 11.4% → 9.5% | YawDD, 100 talking and 105 normal clips | share of clips with at least one warning after calibration | real pipeline, `evaluation_report.py` |
| AUC 0.868 / 0.861; 12/12 subjects | UTA-RLDD, 12 subjects × (alert, drowsy) videos | AUC of each video's mean fused score; subjects whose drowsy video has the higher mean | real pipeline, `evaluation_report.py` |
| Night microsleep alarms 94.7% | NITYMED, 19 microsleep videos | videos with at least one alarm (level ≥ 2): 18 of 19 | real pipeline, `evaluation_report.py` |
| Learned-fusion table | as in that table; every NTHU-DDD subject predicted by a model trained without them | same metrics | recorded signals, [`learned_variants.py`](tools/learned_variants.py) |

Every "→" compares static with adaptive on the same frames. The numbers in the first rows were also recounted independently from the alert levels the pipeline logged, with identical confusion counts.

**Sample sizes.** The frame-level results rest on 4 subjects (NTHU-DDD); differences of a few percentage points between strategies on UTA-RLDD (24 videos) or NITYMED (19 microsleep videos) correspond to one or two videos and should not be read as meaningful.

## Condition thresholds

The first thresholds came from artificially darkened or covered test photos and failed on real footage: low light never triggered at real night, and the occlusion check flagged up to 22% of frames with fully visible eyes. They were re-tuned on 15,319 frames from 482 real videos (NITYMED night driving, YawDD day driving with and without glasses or sunglasses, UTA-RLDD indoor webcams) with [`tools/condition_features.py`](tools/condition_features.py) and [`tools/tune_conditions.py`](tools/tune_conditions.py), choosing on half of the videos and checking on the other half. On the held-out half, through the full pipeline:

| Footage | Read as low light | Read as eyes occluded |
|---|---|---|
| Night driving (NITYMED) | 100% | 0.7% |
| Day driving, no glasses (YawDD) | 0% | 0% |
| Day driving, sunglasses (YawDD) | 0% | 1.3% |
| Indoor webcam (UTA-RLDD) | 0% | 0% |

The occlusion check was tuned to flag at most 3% of single frames of uncovered eyes in every dataset, because a false occlusion lowers the eye weight for a driver whose eyes are perfectly visible. The final contrast floor (0.05) flags no frame of any real dataset.

## What the evaluation changed

- **Low light does not weaken the eye cue.** The original plan lowered the eye weight in low light. On NITYMED, eye-measurement noise was 0.8% at night against 0.7% by day, and night "eyes closed" readings were real closures on inspection. Lowering the eye weight only discarded evidence: night microsleep videos raising an alarm fell from 94.7% to 84.2%. Low light now keeps the normal weights.
- **Facial expressions cause most false alarms.** On NTHU-DDD, 28% of eye-closed frames were in non-drowsy footage, and 83% of those came from talking and laughing, where the smile score separated them cleanly from real closures. Tested leave-one-subject-out, ignoring closures while smiling helps every subject, with recall unchanged.
- **Smiles are judged against the driver's resting face.** Some faces and camera angles score above 0.10 at rest. Without a per-driver margin, 21 of 105 YawDD normal-driving clips read as "smiling" most of the time, so those drivers' eye closures would have been ignored; with it, 5. The price is more false warnings on talking clips (9% → 16%), accepted for safety.
- **Yawns are partly read as smiles.** MediaPipe scores 36–54% of wide-open-mouth frames as smiles, which hid the eye closures that come with yawning. A recent wide-open mouth now counts as a yawn, not a smile.
- **Yawn rule.** MAR above 0.45 for 1.0 s replaces 0.5 for 1.5 s. Chosen on half of the YawDD drivers, it caught 69% instead of 56% of yawns on the other half, with one false yawn in about 50 talking clips; on NITYMED, 2.34 instead of 1.76 of 3 yawns per video.
- **Rejected,** because they did not hold on unseen subjects or had bad side effects: other eye-closure thresholds; a shorter or warm-up PERCLOS window; a longer microsleep time; ignoring closures while the mouth stretches; counting backward head tilts; and letting the eye score alone raise alarms ("peak fusion", which raised false warnings on awake drivers from 23% to 41% of clips).

## Learned fusion

A logistic regression on 13 windowed features (eye-closure fractions, closure length, blinks, relative EAR, yawn and head scores, mouth opening, nodding, smile, conditions) was trained instead of hand-set weights ([`fusion/learned/`](src/drowsiness/fusion/learned), [`tools/learned_fusion.py`](tools/learned_fusion.py), [`tools/learned_variants.py`](tools/learned_variants.py)). Every subject was predicted by a model that never saw them. No variant beat the rule-based adaptive fusion overall:

| Variant (held-out subjects) | NTHU F1 | NTHU false alarms | UTA AUC | YawDD awake warned | Night microsleep alarmed |
|---|---|---|---|---|---|
| **Rule-based adaptive (used)** | **82.7%** | **18.4%** | 0.861 | 12.7% | **94.7%** |
| Learned on NTHU-DDD, all features | 76.8% | 27.8% | 0.458 | 3.4% | 84.2% |
| Learned on NTHU-DDD, without smile and conditions | 72.3% | 29.7% | **0.917** | 4.9% | 84.2% |
| Learned on NTHU-DDD + UTA-RLDD, all features | 77.0% | 27.0% | 0.597 | 6.8% | 100% |

With four acted subjects the model learned the training set's habits (its largest weight says "smiling means alert") and did not transfer to real tiredness; each variant won on one dataset and lost on others. The safety rules (microsleep and face loss) stay in the alert manager in every variant.

The dashboard runs the "NTHU-DDD, all features" model ([`models/learned_fusion.json`](models/learned_fusion.json), 0.07 ms per frame) live next to the two rule-based strategies, for comparison only, and shows its learned weights.

## Limitations

- NTHU-DDD (Kaggle copy) has four subjects and acted drowsiness; UTA-RLDD and NITYMED have only per-video labels. UTA-RLDD and NITYMED were also used to check candidate detection settings, so their results are supporting evidence rather than a fully independent test. The condition thresholds were chosen on half of the videos without drowsiness labels.
- No public dataset combines dark sunglasses or hands over the eyes with drowsiness labels, so the occlusion weights are tested only on simulated frames. On real footage it was checked that occlusion is not falsely detected.
- A standard RGB webcam cannot see the face in complete darkness.
- Processing times during evaluation were measured with six recordings in parallel; live, the pipeline needs about 10–15 ms per frame.

## Reproducing the results

Install the project (see the [README](README.md#quick-start)) and put the datasets in `data/` as described in [`data/README.md`](data/README.md). Then, from the repository root:

```bash
# 1. Run the full pipeline over every recording (hours; resumable).
#    Writes one CSV per recording to data/eval/<dataset>/.
python tools/run_evaluation.py

# 2. The results table above -> stdout and data/eval/results.md
python tools/evaluation_report.py

# 3. Check that the offline simulator matches the pipeline
python tools/replay.py

# 4. Condition thresholds (writes data/condition_features.csv)
python tools/condition_features.py
python tools/tune_conditions.py data/condition_features.csv

# 5. Detection settings and the smile margin, held out
python tools/tune_detection.py
python tools/tune_smile.py

# 6. Learned fusion: comparison table, and the dashboard's model
python tools/learned_variants.py
python tools/learned_variants.py --save models/learned_fusion.json
```

What each script does, with its inputs and outputs, is listed in [`tools/README.md`](tools/README.md).
