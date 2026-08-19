import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import VGG16_UNet


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = VGG16_UNet(
        out_channels=1
    )

    checkpoint = torch.load(
        "vgg16_unet_best.pth",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

])


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "🧠 Brain Tumor Segmentation Using VGG16 + U-Net"
)

st.write(
    "Deep-learning-based brain MRI tumor segmentation "
    "using a pretrained VGG16 encoder and U-Net decoder."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.info(
    f"Model: VGG16 + U-Net | "
    f"Device: {DEVICE}"
)


# ============================================================
# MRI UPLOAD
# ============================================================

st.header("📤 Upload Brain MRI")

uploaded_file = st.file_uploader(

    "Choose an MRI image",

    type=[
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff"
    ]

)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Load image

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Display original image

    st.subheader("Original MRI")

    st.image(
        image,
        use_container_width=True
    )


    # Preprocess

    input_tensor = transform(
        image
    )

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    # Predict

    if st.button(
        "🔍 Segment Tumor"
    ):

        with st.spinner(
            "Segmenting tumor..."
        ):

            with torch.no_grad():

                output = model(
                    input_tensor
                )

                probability = torch.sigmoid(
                    output
                )

                prediction = (
                    probability > 0.5
                ).float()


        # Convert mask to NumPy

        predicted_mask = (

            prediction
            .squeeze()
            .cpu()
            .numpy()

        )


        # Resize mask to original image size

        predicted_mask_image = Image.fromarray(

            (
                predicted_mask * 255
            ).astype(
                np.uint8
            )

        ).resize(

            image.size,

            Image.NEAREST

        )


        predicted_mask = (

            np.array(
                predicted_mask_image
            ) / 255.0

        )


        # ====================================================
        # CREATE OVERLAY
        # ====================================================

        original_array = np.array(
            image
        ).copy()


        tumor_region = (
            predicted_mask > 0.5
        )


        overlay = original_array.copy()


        overlay[tumor_region] = (
            255,
            0,
            0
        )


        overlay_image = Image.fromarray(
            overlay
        )


        # ====================================================
        # TUMOR AREA
        # ====================================================

        tumor_pixels = int(
            np.sum(tumor_region)
        )


        total_pixels = (
            predicted_mask.shape[0]
            *
            predicted_mask.shape[1]
        )


        tumor_percentage = (

            tumor_pixels
            /
            total_pixels
            *
            100

        )


        # ====================================================
        # RESULTS
        # ====================================================

        st.success(
            "Tumor segmentation completed."
        )


        st.header(
            "🧠 Segmentation Results"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.image(

                image,

                caption="Original MRI",

                use_container_width=True

            )


        with col2:

            st.image(

                predicted_mask,

                caption="Predicted Tumor Mask",

                use_container_width=True

            )


        with col3:

            st.image(

                overlay_image,

                caption="Tumor Overlay",

                use_container_width=True

            )


        # ====================================================
        # TUMOR INFORMATION
        # ====================================================

        st.header(
            "📊 Prediction Information"
        )


        metric1, metric2 = st.columns(2)


        with metric1:

            st.metric(
                "Tumor Pixels",
                f"{tumor_pixels:,}"
            )


        with metric2:

            st.metric(
                "Tumor Percentage",
                f"{tumor_percentage:.2f}%"
            )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.warning(
    "⚠️ Research use only. "
    "This system is not a clinical diagnostic tool."
)