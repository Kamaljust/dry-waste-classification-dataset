# Dataset Summary (auto-generated)

- Classes: metal, paper, plastic
- Total images: 2043 (1636 train / 407 val)

## Class distribution

| Class | Train | Val | Total |
|---|---|---|---|
| metal | 604 | 151 | 755 |
| paper | 464 | 115 | 579 |
| plastic | 568 | 141 | 709 |

## Baseline model

- Architecture: MobileNetV2 (ImageNet-pretrained, frozen backbone, fine-tuned classifier head)
- Best validation accuracy: 0.8231

```
              precision    recall  f1-score   support

       metal      0.771     0.914     0.836       151
       paper      0.928     0.783     0.849       115
     plastic      0.817     0.759     0.787       141

    accuracy                          0.823       407
   macro avg      0.839     0.818     0.824       407
weighted avg      0.831     0.823     0.823       407

```

## Figures

- `class_distribution.png` — per-class image counts (train/val)
- `sample_grid.png` — example images per class
- `training_curves.png` — loss/accuracy over epochs
- `confusion_matrix.png` — validation confusion matrix