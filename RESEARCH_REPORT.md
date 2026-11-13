# Nabd AI 2.0 — Research Summary

Research-only ECG image multi-label classifier, not clinically validated. PTB-XL, 6000 records, patient-disjoint split: 4793 train, 602 validation, 605 test. EfficientNetB0, five outputs: NORM, MI, STTC, CD, HYP. Fine-tuned test macro AUROC 0.7385, macro F1 0.5195. The test set informed development; metrics are exploratory. Model weights and complete training pipeline are not yet published. See forthcoming documentation for details.
