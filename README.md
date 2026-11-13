# Nabd AI 2.0 | نبض AI

**Arabic / English ECG research prototype — مشروع بحثي لتحليل صور تخطيط القلب**

> **Research only — للأبحاث فقط.** Not a medical device, not validated for diagnosis, clinical triage, or treatment. Outputs are uncalibrated model scores, not clinical disease probabilities.

## Academic overview / ملخص للمشرف

Nabd AI 2.0 is a multi-label classifier for standardized ECG signal renderings. It uses a fine-tuned EfficientNetB0 backbone and five outputs: `NORM`, `MI`, `STTC`, `CD`, `HYP`.

The development subset contained 6,000 PTB-XL records with patient-disjoint groups: 4,793 training, 602 validation, and 605 test. Signals were rendered to uniform 384×384 PNG images.

| Metric | Original | Fine-tuned |
|---|---:|---:|
| Validation macro AUROC | 0.7574 | 0.7668 |
| Validation macro F1 | 0.5450 | 0.5468 |
| Test macro AUROC | 0.7305 | 0.7385 |
| Test macro F1 | 0.5114 | 0.5195 |

**Critical caveat:** test data informed prior model-development decisions. These figures are exploratory, **not an independent unbiased estimate** of generalization or clinical performance. Thresholds were selected on validation, which also makes validation F1 optimistic.

See **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)** for methodology, class thresholds, scientific limitations and supervisor review checklist.

## Repository contents

- `app.py`: bilingual Gradio research interface, ECG PNG format check and model inference.
- `requirements.txt`: Python dependencies.
- `model/README.md`: instructions for supplying model weights privately.
- `RESEARCH_REPORT.md`: research report and limitations.
- `.gitignore`: excludes model weights and private assets.

**Model weights are not in this public repository.** The interface cannot produce predictions until the owner supplies `model.keras` and `thresholds.json`. This GitHub URL is a code repository, **not** a hosted working demo.

## Run in Google Colab

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/khuloodakram/Nabd-AI-2.0.git
%cd /content/Nabd-AI-2.0
!pip install -q -r requirements.txt
import os
os.environ["NABD_MODEL_DIR"] = "/content/drive/MyDrive/Nabd_AI_2_0/Nabd_AI_2_0_Release"
!python app.py
```

The Drive release folder should contain `model.keras` and `thresholds.json`. The app accepts **384×384 PNGs rendered using the same procedure as training**. A dimension check cannot prove rendering compatibility. Do not upload identifiable patient data to a public demo.

## Dataset attribution

Dataset: [PTB-XL on PhysioNet](https://physionet.org/content/ptb-xl/). Publication: Wagner et al., [PTB-XL, a large publicly available electrocardiography dataset](https://doi.org/10.1038/s41597-020-0495-6).

## Current status

**Academic review prototype, not production-ready.** The full training notebook, split manifest, and image renderer are not yet in the repository. Independent external evaluation and a label-mapping audit are required before scientific or clinical claims.
