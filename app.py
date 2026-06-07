
import streamlit as st
import numpy as np
import torch
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

CLASS_NAMES = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]
FULL_NAMES  = {
    "MEL":"Melanoma","NV":"Melanocytic Nevus",
    "BCC":"Basal Cell Carcinoma","AK":"Actinic Keratosis",
    "BKL":"Benign Keratosis","DF":"Dermatofibroma",
    "VASC":"Vascular Lesion","SCC":"Squamous Cell Carcinoma"
}
RISK = {
    "MEL":"🔴 HIGH RISK — Urgent dermatologist referral",
    "BCC":"🔴 HIGH RISK — Dermatologist referral recommended",
    "SCC":"🔴 HIGH RISK — Dermatologist referral recommended",
    "AK" :"🟡 MODERATE RISK — Consult dermatologist",
    "VASC":"🟡 MODERATE RISK — Clinical evaluation recommended",
    "NV" :"🟢 LOW RISK — Routine monitoring",
    "BKL":"🟢 LOW RISK — Routine monitoring",
    "DF" :"🟢 LOW RISK — Routine monitoring",
}

@st.cache_resource
def load_model():
    m = timm.create_model("efficientnet_b3", pretrained=False, num_classes=9)
    ckpt = torch.load("best_model.pth", map_location="cpu")
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    cam = GradCAM(model=m, target_layers=[m.conv_head])
    return m, cam

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]),
    ToTensorV2()
])

st.set_page_config(page_title="DermaScan AI", page_icon="🔬", layout="wide")
st.title("🔬 DermaScan AI")
st.markdown("### Explainable Skin Lesion Classification — 8 Categories")
st.warning("⚠️ Research tool only — not a substitute for medical diagnosis")

uploaded = st.file_uploader("Upload a dermoscopy image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analysing..."):
        model, cam = load_model()
        img_np    = np.array(image)
        img_float = img_np.astype(np.float32) / 255.0
        tensor    = transform(image=img_np)["image"].unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)[0].numpy()
        probs_8   = probs[:8]
        probs_8   = probs_8 / probs_8.sum()
        pred_idx  = int(probs_8.argmax())
        pred_name = CLASS_NAMES[pred_idx]
        pred_conf = probs_8[pred_idx]
        grayscale_cam = cam(input_tensor=tensor)
        heatmap = show_cam_on_image(img_float, grayscale_cam[0], use_rgb=True)

    with col2:
        st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)

    st.markdown("---")
    st.markdown(f"## Prediction: **{pred_name} — {FULL_NAMES[pred_name]}**")
    st.markdown(f"### Confidence: **{pred_conf:.1%}**")
    st.markdown(f"### Risk: {RISK[pred_name]}")
    st.markdown("---")
    st.markdown("### Confidence Scores — All Classes")
    for i in np.argsort(probs_8)[::-1]:
        st.progress(float(probs_8[i]),
            text=f"{CLASS_NAMES[i]} — {FULL_NAMES[CLASS_NAMES[i]]}: {probs_8[i]:.1%}")
    st.markdown("---")
    st.caption("Model: EfficientNetB3 | Dataset: ISIC 2019 | Test AUC: 0.8973 | Explainability: Grad-CAM")
