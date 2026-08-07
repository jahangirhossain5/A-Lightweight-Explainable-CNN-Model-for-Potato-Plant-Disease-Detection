import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="centered"
)

# =========================
# Class Names
# =========================
CLASS_NAMES = [
    Potato___Early_blight,
    Potato___Late_blight,
    Potato___healthy
]

DISPLAY_NAMES = {
    Potato___Early_blight Early Blight,
    Potato___Late_blight Late Blight,
    Potato___healthy Healthy
}

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model()
    model = tf.keras.models.load_model(
        explainable_cnn_model.keras
    )
    return model


model = load_model()

# =========================
# Title
# =========================
st.title(🥔 Potato Plant Disease Detection)

st.write(
    Upload a potato leaf image and the trained CNN model 
    will predict the disease.
)

st.divider()

# =========================
# Image Upload
# =========================
uploaded_file = st.file_uploader(
    Upload a potato leaf image,
    type=[jpg, jpeg, png]
)

# =========================
# Prediction
# =========================
if uploaded_file is not None

    # Open image
    image = Image.open(uploaded_file).convert(RGB)

    st.subheader(Uploaded Image)

    st.image(
        image,
        caption=Uploaded Potato Leaf,
        use_container_width=True
    )

    # -------------------------
    # Preprocessing
    # -------------------------
    img = image.resize((224, 224))

    img_array = np.array(img).astype(float32)

    # Same preprocessing used during CNN training
    img_array = img_array  255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------
    # Prediction
    # -------------------------
    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[0][predicted_index]
    )  100

    # -------------------------
    # Result
    # -------------------------
    st.divider()

    st.subheader(🔍 Prediction)

    st.success(
        fPrediction {DISPLAY_NAMES[predicted_class]}
    )

    st.metric(
        Confidence,
        f{confidence.2f}%
    )

    # -------------------------
    # Probability
    # -------------------------
    st.subheader(📊 Class Probabilities)

    for i, class_name in enumerate(CLASS_NAMES)

        probability = float(
            predictions[0][i]
        )  100

        st.write(
            f{DISPLAY_NAMES[class_name]} 
            f{probability.2f}%
        )

        st.progress(
            min(probability  100, 1.0)
        )
