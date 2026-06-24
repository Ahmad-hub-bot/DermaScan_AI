# 🔬 DermaScan AI

**Explainable AI-powered skin lesion classification — built with Streamlit, EfficientNet-B3, and Grad-CAM.**

DermaScan AI lets you upload a dermoscopy image (or take a live photo) and get an instant, explainable prediction across 8 dermatological lesion categories, including melanoma. Each prediction comes with a confidence score, risk level, and a Grad-CAM heatmap showing exactly which regions of the image the model focused on.

> ⚠️ **Research / educational use only.** This tool does **not** replace professional medical diagnosis. Always consult a qualified dermatologist for any skin concern.

---

## ✨ Features

- 📁 **Upload an image** or 📷 **use your live camera** to capture a lesion photo
- 🧠 **EfficientNet-B3** classifier trained on the **ISIC 2019** dataset (Test AUC: **0.8973**)
- 🔍 **Grad-CAM** explainability overlay — see what the model "looked at"
- 🩺 Classifies **8 lesion types**:
  | Code | Condition |
  |------|-----------|
  | MEL | Melanoma |
  | NV | Melanocytic Nevus |
  | BCC | Basal Cell Carcinoma |
  | AK | Actinic Keratosis |
  | BKL | Benign Keratosis |
  | DF | Dermatofibroma |
  | VASC | Vascular Lesion |
  | SCC | Squamous Cell Carcinoma |
- 🚦 Automatic **risk level tagging** (High / Moderate / Low) with referral guidance
- 📊 Full probability breakdown across all classes
- 👤 Optional patient detail form (age, gender, lesion location, duration, recent changes) for added context
- ⚠️ **Confidence-aware warnings** — flags low-quality or non-dermoscopic images instead of giving false confidence

---

## 🖥️ Demo Preview

Upload an image →  get a diagnosis card, Grad-CAM heatmap, risk badge, and full class probability breakdown — all in one clean dashboard.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/Ahmad-hub-bot/DermaScan_AI.git
cd DermaScan_AI

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> ℹ️ On first run, the trained model weights (`best_model.pth`) are automatically downloaded from Google Drive via `gdown` — this may take a minute depending on your connection.

---

## 🧰 Tech Stack

| Component | Library |
|---|---|
| Web app framework | [Streamlit](https://streamlit.io/) |
| Model | [`timm`](https://github.com/huggingface/pytorch-image-models) — EfficientNet-B3 |
| Deep learning | PyTorch |
| Explainability | [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) |
| Image preprocessing | Albumentations, OpenCV, Pillow |
| Model hosting | Google Drive (via `gdown`) |

---

## 📁 Project Structure

```
DermaScan_AI/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── packages.txt         # System-level packages (for deployment, e.g. Streamlit Cloud)
└── README.md
```

---

## 📌 Tips for Best Results

- ✅ Use a **close-up photo** of the lesion, filling most of the frame
- ✅ Shoot in **good, even lighting**
- ✅ Keep the camera **steady and in focus**
- ❌ Avoid flash glare, shadows, filters, or zoom/portrait mode
- 🔬 Best accuracy comes from real **dermoscopy images** — the model was trained on clinical dermoscope photos, so standard phone photos may show lower confidence

---

## ⚠️ Disclaimer

DermaScan AI is a research/educational project and is **not a certified medical device**. Predictions should never be used as a substitute for professional dermatological evaluation. If you notice a new, changing, or concerning skin lesion, please see a licensed healthcare provider.

---

## 🙌 Acknowledgements

- [ISIC 2019 Challenge Dataset](https://challenge.isic-archive.com/)
- [EfficientNet](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)
- [Grad-CAM](https://arxiv.org/abs/1610.02391) for visual explainability
