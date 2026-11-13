# Nabd AI 2.0 — Research Report / التقرير البحثي

## 1. Objective / الهدف

Develop a research-only multi-label ECG image classifier using uniform renderings of ECG signals. This prototype is **not** clinically validated and must not be used for patient diagnosis or triage.

## 2. Dataset and partitions

- Dataset: **PTB-XL**, from PhysioNet (see references below).
- Selected records: **6,000**.
- Train: **4,793**; validation: **602**; test: **605**.
- Patient IDs were checked for **zero overlap** between partitions.
- Signals were rendered to uniform **384×384 RGB PNG** images.
- Fixed label order: **NORM, MI, STTC, CD, HYP**.

**Open issue:** a sample of working annotations included `NORM` together with a disease superclass. The label mapping must be audited against PTB-XL source metadata before medical interpretation.

## 3. Architecture and fine-tuning

- Backbone: **EfficientNetB0**.
- Head: global average pooling, dropout, 5-output multi-label sigmoid layer.
- Fine-tuning: last **20** backbone layers eligible for training, excluding BatchNormalization; Adam learning rate **3e-6**; **3 epochs**; binary cross-entropy; best validation-loss checkpoint saved.
- Model expects **384×384 RGB float32** images in **0–255** pixel range, matching the Colab pipeline.
- Each class uses a separate decision threshold selected by validation F1.

### Fine-tuned thresholds

| NORM | MI | STTC | CD | HYP |
|---:|---:|---:|---:|---:|
| 0.52 | 0.20 | 0.18 | 0.28 | 0.11 |

## 4. Evaluation

| Metric | Before fine-tuning | After fine-tuning |
|---|---:|---:|
| Validation macro AUROC | 0.7574 | 0.7668 |
| Validation macro F1 | 0.5450 | 0.5468 |
| Test macro AUROC | 0.7305 | 0.7385 |
| Test macro F1 | 0.5114 | 0.5195 |

The test partition contains **605** records. Results were recorded in the original Google Colab session, not reproduced by a public CI workflow. **The test set had already been examined during development**; consequently this is an exploratory comparison, not an untouched independent test. Validation F1 is measured on the same data used to optimize thresholds. No confidence intervals, external evaluation, calibration or prospective clinical validation are available.

## 5. Scientific limitations

1. **Not a clinical model.** Scores are not calibrated probabilities of disease; threshold crossings cannot confirm or exclude diagnoses.
2. **Input-domain restriction.** Only ECG images rendered exactly like the training data are within the demonstrated workflow. The interface cannot verify compatibility of uploaded images beyond PNG dimensions.
3. **Label audit.** Review the possible overlap between NORM and pathological labels.
4. **Selection and generalization.** A subset of PTB-XL and a previously consulted test set do not establish generalizability to new hospitals, devices, patients or populations.
5. **Privacy.** Do not publish or upload patient-identifiable ECG records to a public Gradio endpoint.
6. **Reproducibility.** This repository currently provides inference code and documentation only. Training notebook, dataset manifests, signal-to-image rendering pipeline and model weights are not included.

## 6. Supervisor review checklist

- Verify label construction and PTB-XL diagnostic superclass aggregation.
- Audit patient-wise partitioning and any leakage.
- Reproduce the signal rendering and preprocessing exactly.
- Recompute class-specific precision, recall, F1 and uncertainty intervals.
- Evaluate on genuinely independent, previously untouched external data.
- Review privacy, licensing, ethics and any claims about medical use.

## References

- [PTB-XL on PhysioNet](https://physionet.org/content/ptb-xl/)
- Wagner et al., [PTB-XL, a large publicly available electrocardiography dataset](https://doi.org/10.1038/s41597-020-0495-6)

**Status:** Research prototype for academic discussion; not a finished, validated medical system.
