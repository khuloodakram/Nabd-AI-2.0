"""Nabd AI 2.0: research-only bilingual ECG image demo."""
import json
import os
from pathlib import Path
import gradio as gr
import numpy as np
from PIL import Image

MODEL_DIR = Path(os.environ.get("NABD_MODEL_DIR", Path(__file__).parent / "model"))
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
TRANSLATIONS = {
    "العربية": {
        "title": "نبض AI 2.0 — أداة بحثية لتخطيط القلب",
        "description": "نموذج بحثي لصور ECG الموحدة. غير مخصص للتشخيص أو الفرز الطبي.",
        "upload": "صورة ECG بحثية (PNG، 384×384)",
        "button": "تحليل بحثي",
        "results": "نتائج الفئات — درجات نموذج غير معايرة",
        "status": "حالة التحليل",
        "warning": "⚠️ نتائج بحثية فقط؛ لا تثبت أو تنفي أي مرض. استشر مختصًا عند وجود أعراض.",
        "invalid": "الصورة يجب أن تكون PNG بقياس 384×384. لا يمكن التحقق آليًا من مطابقة طريقة الرسم لبيانات التدريب.",
        "missing": "ملفات النموذج غير متاحة. أضف model.keras وthresholds.json أو اضبط NABD_MODEL_DIR.",
        "done": "نجح الاستدلال البحثي. تجاوز العتبة ليس تشخيصًا.",
        "above": "تجاوز العتبة",
        "below": "لم يتجاوز",
    },
    "English": {
        "title": "Nabd AI 2.0 — ECG Research Demo",
        "description": "Research prototype for standardized ECG renderings. Not for diagnosis or triage.",
        "upload": "Research ECG PNG (384×384)",
        "button": "Research analysis",
        "results": "Class scores — uncalibrated model outputs",
        "status": "Analysis status",
        "warning": "⚠️ Research only. A score cannot confirm or exclude disease. Seek qualified care for symptoms.",
        "invalid": "Upload a 384×384 PNG. Rendering compatibility cannot be automatically verified.",
        "missing": "Model files unavailable. Supply model.keras and thresholds.json or set NABD_MODEL_DIR.",
        "done": "Research inference completed. A threshold crossing is not a diagnosis.",
        "above": "Above threshold",
        "below": "Below threshold",
    },
}
_model = None
_thresholds = None

def load_model():
    global _model, _thresholds
    if _model is None:
        import tensorflow as tf
        with open(MODEL_DIR / "thresholds.json", encoding="utf-8") as f:
            thresholds = json.load(f)
        if set(thresholds) != set(LABELS):
            raise ValueError("Threshold labels mismatch")
        net = tf.keras.models.load_model(str(MODEL_DIR / "model.keras"), compile=False)
        if tuple(net.input_shape[1:]) != (384, 384, 3) or net.output_shape[-1] != 5:
            raise ValueError("Unexpected model shape")
        _model, _thresholds = net, thresholds
    return _model, _thresholds

def infer(image_path, language):
    t = TRANSLATIONS[language]
    if not image_path:
        return [], t["invalid"]
    try:
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.size != (384, 384):
                return [], t["invalid"]
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        if not (MODEL_DIR / "model.keras").exists() or not (MODEL_DIR / "thresholds.json").exists():
            return [], t["missing"]
        model, thresholds = load_model()
        scores = np.asarray(model(pixels[None, ...], training=False)).reshape(-1)
        if scores.shape != (5,) or not np.isfinite(scores).all():
            raise ValueError("Invalid model output")
        rows = []
        for label, score in zip(LABELS, scores):
            threshold = float(thresholds[label])
            rows.append([label, round(float(score), 4), threshold, t["above"] if score >= threshold else t["below"]])
        return rows, t["done"]
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"

def change_language(lang):
    t = TRANSLATIONS[lang]
    return (
        gr.update(value=f"## {t['title']}\n\n{t['description']}"),
        gr.update(label=t["upload"]),
        gr.update(value=t["button"]),
        gr.update(label=t["results"]),
        gr.update(label=t["status"]),
        gr.update(value=t["warning"]),
    )

with gr.Blocks(title="Nabd AI 2.0 | Research") as demo:
    language = gr.Radio(["العربية", "English"], value="العربية", label="اللغة / Language")
    heading = gr.Markdown(f"## {TRANSLATIONS['العربية']['title']}\n\n{TRANSLATIONS['العربية']['description']}")
    image = gr.Image(type="filepath", sources=["upload"], label=TRANSLATIONS["العربية"]["upload"])
    analyze = gr.Button(TRANSLATIONS["العربية"]["button"], variant="primary")
    results = gr.Dataframe(headers=["Class", "Score", "Threshold", "Decision"], datatype=["str", "number", "number", "str"], interactive=False, label=TRANSLATIONS["العربية"]["results"])
    status = gr.Textbox(label=TRANSLATIONS["العربية"]["status"], interactive=False)
    warning = gr.Markdown(TRANSLATIONS["العربية"]["warning"])
    analyze.click(infer, inputs=[image, language], outputs=[results, status])
    language.change(change_language, inputs=language, outputs=[heading, image, analyze, results, status, warning])

if __name__ == "__main__":
    demo.launch()
