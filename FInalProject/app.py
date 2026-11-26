import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================
# CONFIGURATION
# ============================================================

class CFG:
    IMG_SIZE = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EMBED_DIM = 256
    NUM_HEADS = 8
    NUM_TRANS_LAYERS = 4


# ============================================================
# VESSEL EXTRACTION FUNCTION
# ============================================================

def extract_vessels(img_rgb, img_size=512):
    green = img_rgb[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    vessel_mask = cv2.resize(cleaned, (img_size, img_size))
    return vessel_mask


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class VesselEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, CFG.EMBED_DIM, 3, 2, 1),
            nn.BatchNorm2d(CFG.EMBED_DIM),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b3(weights=None)
        self.features = backbone.features
        self.proj = nn.Conv2d(1536, CFG.EMBED_DIM, 1)

    def forward(self, x):
        return self.proj(self.features(x))


class DVATNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.ves_enc = VesselEncoder()

        self.attn = nn.MultiheadAttention(
            CFG.EMBED_DIM,
            CFG.NUM_HEADS,
            batch_first=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            CFG.EMBED_DIM,
            CFG.NUM_HEADS,
            batch_first=True
        )

        self.trans = nn.TransformerEncoder(
            encoder_layer,
            CFG.NUM_TRANS_LAYERS
        )

        self.cls = nn.Linear(CFG.EMBED_DIM, 2)

    def forward(self, img, mask):
        fi = self.img_enc(img)
        fv = self.ves_enc(mask)

        ti = fi.flatten(2).transpose(1, 2)
        tv = fv.flatten(2).transpose(1, 2)

        fused, _ = self.attn(ti, tv, tv)
        out = self.trans(fused).mean(1)

        return self.cls(out)


# ============================================================
# LOAD MODEL
# ============================================================

model = DVATNet().to(CFG.DEVICE)
model.load_state_dict(torch.load("weights/dvatnet_fives.pth", map_location=CFG.DEVICE))
model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="DVAT-Net Risk Prediction", layout="centered")

st.title("🫀 Retinal Cardiovascular Risk Prediction")
st.markdown("Upload a retinal fundus image to assess cardiovascular risk.")

uploaded = st.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])

if uploaded:

    # Read image
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (CFG.IMG_SIZE, CFG.IMG_SIZE))

    # Extract vessels
    vessel_mask = extract_vessels(img_rgb, CFG.IMG_SIZE)

    # Prepare tensors
    mask_tensor = torch.tensor(
        vessel_mask / 255.0,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(CFG.DEVICE)

    img_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(CFG.DEVICE)

    st.markdown("---")
    st.subheader("🔎 Preprocessed Inputs")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img_rgb, caption="Original Fundus Image", width=350)

    with col2:
        st.image(vessel_mask, caption="Extracted Vessel Map", width=350)

    st.markdown("---")

    if st.button("🔍 Analyze Risk", use_container_width=True):

        with st.spinner("Analyzing retinal structures..."):
            with torch.no_grad():
                logits = model(img_tensor, mask_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                risk = float(probs[1])

        # Generate Attention Map
        fmap = model.img_enc(img_tensor)
        cam = fmap.mean(1)[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (CFG.IMG_SIZE, CFG.IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_resized, 0.6, heat, 0.4, 0)

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        if risk > 0.7:
            level = "🔴 HIGH RISK"
            color = "red"
        elif risk > 0.4:
            level = "🟡 MODERATE RISK"
            color = "orange"
        else:
            level = "🟢 LOW RISK"
            color = "green"

        st.markdown(
            f"<h3 style='text-align:center; color:{color};'>{level}</h3>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<h4 style='text-align:center;'>Risk Probability: {risk*100:.2f}%</h4>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("🗺️ Attention Heatmap")

        center_col = st.columns([1, 2, 1])
        with center_col[1]:
            st.image(
                overlay,
                caption="Model Attention Visualization",
                width=400
            )

        st.info(
            "⚠️ This is a research prototype and should not replace professional medical diagnosis."
        )