import streamlit as st
import numpy as np
import torch
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import gdown

CLASS_NAMES = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]
FULL_NAMES  = {
    "MEL":"Melanoma","NV":"Melanocytic Nevus",
    "BCC":"Basal Cell Carcinoma","AK":"Actinic Keratosis",
    "BKL":"Benign Keratosis","DF":"Dermatofibroma",
    "VASC":"Vascular Lesion","SCC":"Squamous Cell Carcinoma"
}
RISK = {
    "MEL":("HIGH RISK","Urgent dermatologist referral","#A32D2D","#FCEBEB"),
    "BCC":("HIGH RISK","Dermatologist referral recommended","#A32D2D","#FCEBEB"),
    "SCC":("HIGH RISK","Dermatologist referral recommended","#A32D2D","#FCEBEB"),
    "AK" :("MODERATE RISK","Consult a dermatologist","#854F0B","#FAEEDA"),
    "VASC":("MODERATE RISK","Clinical evaluation recommended","#854F0B","#FAEEDA"),
    "NV" :("LOW RISK","Routine monitoring advised","#3B6D11","#EAF3DE"),
    "BKL":("LOW RISK","Routine monitoring advised","#3B6D11","#EAF3DE"),
    "DF" :("LOW RISK","Routine monitoring advised","#3B6D11","#EAF3DE"),
}
RISK_ICON = {"HIGH RISK":"⛔","MODERATE RISK":"⚠️","LOW RISK":"✅"}

MODEL_PATH = "best_model.pth"
GDRIVE_ID  = "1eNwqylZ_xQbc74Xx44j5WoRI7H1j9AVY"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Downloading model on first run... (this may take a minute)"):
            gdown.download(id=GDRIVE_ID, output=MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")

st.set_page_config(page_title="DermaScan AI", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}
.hero {
    background: linear-gradient(135deg, #085041 0%, #1D9E75 60%, #5DCAA5 100%);
    border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 2rem; color: white;
}
.hero h1 {
    font-family: 'DM Serif Display', serif; font-size: 2.8rem; font-weight: 400;
    margin: 0 0 0.4rem 0; color: white; letter-spacing: -0.5px;
}
.hero p { font-size: 1rem; opacity: 0.88; margin: 0; font-weight: 300; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35); border-radius: 100px;
    padding: 4px 14px; font-size: 0.75rem; font-weight: 500;
    letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 1rem;
}
.warn-bar {
    background: #FAEEDA; border-left: 4px solid #BA7517;
    border-radius: 0 10px 10px 0; padding: 0.75rem 1.25rem;
    font-size: 0.88rem; color: #633806; margin-bottom: 1.5rem;
}
.result-card {
    background: white; border-radius: 16px; padding: 1.5rem;
    border: 1px solid #e2e8f0; margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.pred-name {
    font-family: 'DM Serif Display', serif; font-size: 2.2rem;
    font-weight: 400; color: #085041; margin: 0;
}
.pred-full { font-size: 1rem; color: #6b7280; margin: 0.2rem 0 0 0; }
.confidence-chip {
    display: inline-block; background: #E1F5EE; color: #0F6E56;
    border-radius: 100px; padding: 6px 18px; font-size: 1.1rem;
    font-weight: 600; margin-top: 0.75rem;
}
.risk-badge { border-radius: 12px; padding: 1rem 1.25rem; display: flex; align-items: center; gap: 12px; margin-top: 1rem; }
.risk-label { font-weight: 600; font-size: 0.95rem; letter-spacing: 0.3px; }
.risk-sub { font-size: 0.85rem; opacity: 0.8; margin-top: 2px; }
.prob-row { margin-bottom: 0.65rem; }
.prob-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: #374151; margin-bottom: 4px; }
.prob-bar-bg { background: #f1f5f9; border-radius: 100px; height: 8px; overflow: hidden; }
.prob-bar-green { height:100%; border-radius:100px; background: linear-gradient(90deg,#1D9E75,#5DCAA5); }
.prob-bar-red   { height:100%; border-radius:100px; background: linear-gradient(90deg,#E24B4A,#F09595); }
.prob-bar-amber { height:100%; border-radius:100px; background: linear-gradient(90deg,#BA7517,#FAC775); }
.img-label {
    background: #f8fafc; border-top: 1px solid #e2e8f0;
    padding: 0.5rem 1rem; font-size: 0.82rem; color: #6b7280;
    text-align: center; font-weight: 500;
}
.section-head {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #9ca3af; margin: 1.5rem 0 0.75rem 0;
}
.camera-tip {
    background: #E1F5EE; border-radius: 12px; padding: 1rem 1.25rem;
    margin-bottom: 1.5rem; font-size: 0.9rem; color: #085041;
}
.footer-strip {
    background: #f8fafc; border-radius: 12px; padding: 0.75rem 1.5rem;
    font-size: 0.78rem; color: #9ca3af; display: flex; gap: 24px;
    flex-wrap: wrap; margin-top: 2rem; border: 1px solid #e2e8f0;
}
div[data-testid="stButton"] > button {
    background: #1D9E75 !important; color: white !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.65rem 2rem !important; font-size: 1rem !important;
    font-weight: 500 !important; width: 100%; transition: background 0.2s !important;
}
div[data-testid="stButton"] > button:hover { background: #0F6E56 !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI · EfficientNet-B3 · Grad-CAM XAI</div>
    <h1>🔬 DermaScan AI</h1>
    <p>Explainable skin lesion classification across 8 dermatological categories.<br>
    Upload a dermoscopy image or use your live camera for instant AI analysis.</p>
</div>
<div class="warn-bar">
    ⚠️ <strong>Research use only.</strong> This tool does not replace professional medical diagnosis. Always consult a qualified dermatologist.
</div>
""", unsafe_allow_html=True)

# ── Download model if missing ─────────────────────────────────────────────────
download_model()

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = timm.create_model("efficientnet_b3", pretrained=False, num_classes=9)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    cam = GradCAM(model=m, target_layers=[m.conv_head])
    return m, cam

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

def run_inference(img_np):
    import cv2 as cv
    model, cam = load_model()
    # Resize to 224x224 for both the model input and Grad-CAM overlay
    img_resized = cv.resize(img_np, (224, 224))
    img_float = img_resized.astype(np.float32) / 255.0
    tensor = transform(image=img_np)["image"].unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0].numpy()
    probs_8 = probs[:8] / probs[:8].sum()
    pred_idx  = int(probs_8.argmax())
    pred_name = CLASS_NAMES[pred_idx]
    pred_conf = probs_8[pred_idx]
    grayscale_cam = cam(input_tensor=tensor)
    # grayscale_cam[0] is (224,224), img_float is now also (224,224,3) — sizes match
    heatmap = show_cam_on_image(img_float, grayscale_cam[0], use_rgb=True)
    return pred_name, pred_conf, probs_8, heatmap

def show_results(pred_name, pred_conf, probs_8, heatmap, original_img):
    risk_level, risk_msg, risk_color, risk_bg = RISK[pred_name]
    risk_icon = RISK_ICON[risk_level]

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(original_img, use_column_width=True)
        st.markdown('<div class="img-label">📷 Input Image</div>', unsafe_allow_html=True)
    with col2:
        st.image(heatmap, use_column_width=True)
        st.markdown('<div class="img-label">🔥 Grad-CAM · AI attention regions</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2, 1], gap="medium")
    with left:
        st.markdown(f"""
        <div class="result-card">
            <div class="section-head">Diagnosis</div>
            <p class="pred-name">{pred_name}</p>
            <p class="pred-full">{FULL_NAMES[pred_name]}</p>
            <div class="confidence-chip">Confidence: {pred_conf:.1%}</div>
            <div class="risk-badge" style="background:{risk_bg};">
                <span style="font-size:1.6rem">{risk_icon}</span>
                <div>
                    <div class="risk-label" style="color:{risk_color}">{risk_level}</div>
                    <div class="risk-sub" style="color:{risk_color}">{risk_msg}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="result-card" style="height:100%">
            <div class="section-head">Model Info</div>
            <div style="font-size:0.85rem; color:#374151; line-height:2.2">
                <div>🧠 <strong>EfficientNet-B3</strong></div>
                <div>📊 ISIC 2019 Dataset</div>
                <div>📈 Test AUC: <strong>0.8973</strong></div>
                <div>🔍 XAI: Grad-CAM</div>
                <div>🏷 8 lesion classes</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="result-card"><div class="section-head">All Class Probabilities</div>', unsafe_allow_html=True)
    for i in np.argsort(probs_8)[::-1]:
        p = float(probs_8[i])
        name = CLASS_NAMES[i]
        r_level = RISK[name][0]
        bar_cls = "prob-bar-red" if r_level == "HIGH RISK" else ("prob-bar-amber" if r_level == "MODERATE RISK" else "prob-bar-green")
        bold = "font-weight:600;" if i == int(probs_8.argmax()) else ""
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">
                <span style="{bold}">{name} — {FULL_NAMES[name]}</span>
                <span style="{bold}">{p:.1%}</span>
            </div>
            <div class="prob-bar-bg">
                <div class="{bar_cls}" style="width:{p*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-strip">
        <span>🧬 EfficientNet-B3</span>
        <span>📂 ISIC 2019</span>
        <span>📈 AUC 0.8973</span>
        <span>🔍 Grad-CAM XAI</span>
        <span>⚠️ Not for clinical use</span>
    </div>
    """, unsafe_allow_html=True)

# ── Mode selector ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Choose Input Method</div>', unsafe_allow_html=True)
mode = st.radio("", ["📁  Upload Image", "📷  Live Camera"], horizontal=True, label_visibility="collapsed")
st.markdown("---")

if mode == "📁  Upload Image":
    uploaded = st.file_uploader("Drag & drop a dermoscopy image (JPG, JPEG, PNG)", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)
        with st.spinner("🔬 Analysing image..."):
            pred_name, pred_conf, probs_8, heatmap = run_inference(img_np)
        show_results(pred_name, pred_conf, probs_8, heatmap, image)

else:
    st.markdown("""
    <div class="camera-tip">
        📷 <strong>Live camera</strong> — Allow camera access when prompted, point at the skin lesion,
        then click <strong>Take Photo</strong>. The AI will analyse it instantly.
    </div>
    """, unsafe_allow_html=True)

    photo = st.camera_input("Take a photo of the lesion")
    if photo is not None:
        image = Image.open(photo).convert("RGB")
        img_np = np.array(image)
        with st.spinner("🔬 Analysing captured photo..."):
            pred_name, pred_conf, probs_8, heatmap = run_inference(img_np)
        show_results(pred_name, pred_conf, probs_8, heatmap, image)
