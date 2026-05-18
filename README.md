# CV II Final Project: Image-to-Image Translation (Maps)

## 1. Project Overview

This project addresses image-to-image translation using the **Maps** dataset, where the model learns to generate realistic map images from corresponding label maps.

Main goals:

1. Train a conditional GAN model for paired image translation.
2. Evaluate model performance on the test split.
3. Run inference on single images or full folders for qualitative analysis.

## 2. Dataset

- Dataset: **Maps**
- Expected extracted location: `data/raw/maps`

Prepare the dataset with:

```bash
python scripts/prepare_data.py --zip maps.zip --out data/raw --dataset-name maps
```

## 3. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Reproducible Workflow

1. **Prepare data**

```bash
python scripts/prepare_data.py --zip maps.zip --out data/raw --dataset-name maps
```

This creates explicit `train/`, `val/`, and `test/` folders under `data/raw/maps`.

2. **Train**

```bash
python scripts/train.py --config configs/base.yaml
```

3. **Evaluate**

```bash
python scripts/evaluate.py --config configs/base.yaml --checkpoint checkpoints/best.pt
```

4. **Task 2 improvement (augmentation)**

```bash
python scripts/train.py --config configs/improved_aug.yaml
python scripts/evaluate.py --config configs/improved_aug.yaml --checkpoint checkpoints_aug/best.pt
```

5. **Inference demo**

```bash
python scripts/infer.py --checkpoint checkpoints/best.pt --input data/raw/maps/test --output outputs/demo_test --paired-input
```

## 5. Inference Examples


## Notebook Demo

Run the notebook demo:

- Open `demo/colab_runner.ipynb`
- Set paths in the configuration cell (checkpoint, test image/folder)
- Run all cells

The notebook also generates side-by-side outputs for easier interpretation.

## 6. Deliverables

This repository is organized to support the following deliverables:

1. Training and evaluation scripts.
2. Checkpoints and generated outputs. (In a separated link https://nubeusc-my.sharepoint.com/:f:/g/personal/patriciaguadalupe_alvarenga_rai_usc_es/IgB-HfKqCkyHQqaXK8g01KiZASDtHARgc-Z_Pf1a3iqxBlE?e=fSadey)
3. Quantitative and qualitative results for final analysis.

## 7. Notes for Evaluation

- Uses the same configuration file (`configs/base.yaml`) for reproducibility.
- Runs evaluation with the target checkpoint to report final metrics.
- Uses test-split inference outputs as qualitative evidence of model performance.
