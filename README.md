# Dry Waste Classification Dataset (Metal, Paper, Plastic)

Single-item images of dry waste, captured using a purpose-built automated
sorting device in 2020, in a fixed top-down orientation against a
consistent background.

The image dataset itself is hosted externally (see below) due to size.
This repository contains the dataset documentation, baseline training
code, and baseline results.

## Dataset access

- **Data (Kaggle):** [add Kaggle dataset URL here]
- **Data + citable DOI (Zenodo):10.5281/zenodo.21849964

## Contents of this repository

```
docs/      Full dataset documentation (LaTeX source + compiled PDF), figures
code/      Baseline PyTorch training script (MobileNetV2 transfer learning)
results/   Baseline metrics, class distribution, classification report
```

## Dataset summary

- **Classes:** metal, paper, plastic
- **Total images:** 2,043 (1,636 train / 407 val)

| Class   | Train | Val | Total |
|---------|-------|-----|-------|
| Metal   | 604   | 151 | 755   |
| Paper   | 464   | 115 | 579   |
| Plastic | 568   | 141 | 709   |

**Note:** A small number of glass images (22 total) were also collected but
excluded due to insufficient sample count for meaningful training or
evaluation.

## Baseline results

A MobileNetV2 transfer-learning baseline (partial fine-tuning) achieves
**82.3% validation accuracy**.

| Class   | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| Metal   | 0.771     | 0.914  | 0.836    |
| Paper   | 0.928     | 0.783  | 0.849    |
| Plastic | 0.817     | 0.759  | 0.787    |

See `docs/dataset_documentation.pdf` for the full write-up, methodology,
and discussion of results.

## Quick start

```bash
pip install torch torchvision scikit-learn matplotlib pillow --break-system-packages

# Arrange downloaded data as:
#   data/train/<class>/*.jpg
#   data/val/<class>/*.jpg

python code/train_baseline_classifier.py
```

## Known limitations

- Images come from a single capture device/setup — models trained here may
  not generalize to other cameras, lighting, or backgrounds without
  fine-tuning.
- Some images show the same physical object from multiple angles. If
  building your own train/val split, consider grouping by object to avoid
  data leakage between splits.
- Moderate class imbalance (paper underrepresented).
- Dataset size is modest (~2k images) — suitable as a lightweight benchmark
  or transfer-learning starting point, not a large-scale training set.

## License

- **Dataset:** CC BY 4.0 (see the Kaggle/Zenodo listings)
- **Code in this repository:** MIT — see `LICENSE-CODE`

## Citation

If you use this dataset or code, please cite:

```
[Author Name]. (2026). Dry Waste Classification Dataset
(Metal, Paper, Plastic) [Data set]. Zenodo. https://doi.org/[DOI]
```

See `CITATION.cff` for machine-readable citation metadata.
