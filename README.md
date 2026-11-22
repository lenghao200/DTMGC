# DTMGC

## Install

From the repository root, install the dependencies with Python 3.10.

```sh
python -m pip install -r requirements.txt
```

## Prepare the data

Put the label files and pre-extracted features under `data/`, or set
`DTMGC_DATA_ROOT` to an existing data directory with this layout.

```text
data/
├── CMUMOSI/
│   ├── CMUMOSI_features_raw_2way.pkl
│   └── features/
└── IEMOCAP/
    ├── IEMOCAP_features_raw_4way.pkl
    └── features/
```

Each `features/` directory contains `wav2vec-large-c-UTT`,
`deberta-large-4-UTT`, and `manet_UTT` when using the default features.

## Train

```sh
# CMU-MOSI
python -u DTMGC/train_mr.py --dataset CMUMOSI --mask_rate 0.7 --seed 66

# IEMOCAP (four classes)
python -u DTMGC/train_mr.py --dataset IEMOCAPFour --mask_rate 0.7 --seed 66
```

At `--mask_rate 0.7`, the masking routine keeps one of the three modalities per
utterance. Training defaults to 300 epochs, with 150 epochs per stage. Logs are
saved under `saved/log/main_result/` and result summaries are written to the
current directory.
