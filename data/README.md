# Datasets

The evaluation datasets are **not in the repository**: they are large, owned by their authors and contain recordings of people. Only this file is tracked; everything else in `data/` is ignored by Git. The detector itself does not need any of this, only the evaluation tools in [`../tools/`](../tools) do.

Download each dataset from its source, accept its terms (non-commercial academic use), and place it as shown below.

| Dataset | Source | Paper |
|---|---|---|
| NTHU-DDD (Kaggle copy, frames) | [kaggle.com/datasets/ikhlaselhamly/nthu-ddd](https://www.kaggle.com/datasets/ikhlaselhamly/nthu-ddd) | Weng, Lai & Lai, ACCV Workshops 2016 |
| YawDD | [kaggle.com/datasets/enider/yawdd-dataset](https://www.kaggle.com/datasets/enider/yawdd-dataset) | Abtahi et al., ACM MMSys 2014 |
| UTA-RLDD | [sites.google.com/view/utarldd](https://sites.google.com/view/utarldd/home) | Ghoddoosian, Galib & Athitsos, CVPR Workshops 2019 |
| NITYMED | [IEEE DataPort, doi:10.21227/85xe-3f88](https://dx.doi.org/10.21227/85xe-3f88) | Petrellis et al., 2022 |
| UL-DD (restricted: request access and accept its data usage agreement) | [zenodo.org/records/17978727](https://zenodo.org/records/17978727) | Bodaghi et al., Scientific Data 13, 506 (2026), [doi:10.1038/s41597-025-06540-1](https://doi.org/10.1038/s41597-025-06540-1) |

## Expected layout

```
data/
├── NTHU-DDD/
│   ├── drowsy/        001_glasses_sleepyCombination_1003_drowsy.jpg, ...
│   └── notdrowsy/     ..._notdrowsy.jpg
├── YawDD dataset/
│   └── Mirror/        (only the Mirror videos are used)
│       ├── Female_mirror/            1-FemaleNoGlasses-Normal.avi, ...-Talking.avi, ...-Yawning.avi
│       └── Male_mirror Avi Videos/   ...
├── uta-rldd/
│   ├── Fold1_part1/   01/ 02/ ...   each with 0.mov (alert), 5.mov (low vigilant), 10.mov (drowsy)
│   └── Fold1_part2/   ...
├── DSM_Dataset-HDTV720/              (NITYMED)
│   ├── Microsleep/{Female,Male}/HDTV720/*.mp4
│   └── Yawning/{Female,Male}/HDTV720/*.mp4
├── UL-DD/
│   ├── Video_Data/<A-S>/<A|D>/   <id>_IR_<A|D>.mp4 (infrared face, 240x220, 60 fps), ...
│   ├── CSV_Files/ Extracted_Features/ Codes/
│   └── Labels.csv                KSS (1-9) every 4 min, 10 per session; A = awake, D = drowsy
│
│   Written by the tools:
├── eval/
│   ├── nthu/ yawdd/ uta/ nitymed/    one CSV of per-frame results per recording
│   └── results.md                    the results tables
└── condition_features.csv            condition measurements for threshold tuning
```

Folder names must match exactly (including spaces and capitals): the tools find recordings and their labels from these paths and file names. NTHU-DDD labels come from the file names (`drowsy` / `notdrowsy`), YawDD labels from the clip names (`Normal`, `Talking`, `Yawning`), UTA-RLDD labels from the video names (0, 5, 10) and NITYMED labels from the folder (`Microsleep`, `Yawning`). UL-DD labels are the self-reported KSS ratings in `Labels.csv`.

## Privacy

The per-frame CSVs in `eval/` say when a recorded person appeared drowsy. Keep them local, like the datasets themselves.
