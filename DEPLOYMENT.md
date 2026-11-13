# Deployment plan / خطة تشغيل العرض البحثي

## What is ready?

- Bilingual Gradio application code.
- Documented research metrics and limitations.
- Model-loading support through `NABD_MODEL_DIR`.
- GitHub automated **syntax and file-presence** checks (these are NOT runtime or medical validation tests).

## What is still needed?

The model weights are stored in the project owner's Google Drive, **not** in this GitHub repository. The assistant cannot retrieve that private Drive file from GitHub. The owner must supply the release files through an authorized channel before a hosted inference demo can run.

Required release assets:

- `model.keras`
- `thresholds.json`
- `metadata.json` (recommended)

Original Drive location:

`MyDrive/Nabd_AI_2_0/Nabd_AI_2_0_Release/`

## Local / Google Colab preview

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

Run with a **synthetic or permitted research example** rendered by the exact original ECG image-generation pipeline. A valid 384×384 PNG alone does not establish input compatibility.

## Hosted demonstration for academic review

Prefer a **private, access-controlled** deployment rather than an unrestricted public endpoint. The hosting provider needs sufficient RAM for TensorFlow and the saved EfficientNet model. Supply the model files privately through the hosting provider's secure storage mechanism; do not commit them to a public GitHub repository or expose private Drive links.

For an initial supervisor review, the GitHub repository can be shared immediately, but it is **not a running hosted demo**. Only claim a working deployment after a real startup, model-loading, and inference smoke test has succeeded on that host.

## Data governance

- Never upload patient-identifiable ECGs to public hosting.
- Do not expose training datasets, private credentials, or Drive tokens.
- Do not imply the outputs are clinical diagnoses or calibrated probabilities.
- Independent external evaluation is required before claims about clinical performance.
