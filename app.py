import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# =========================================================
# 1. App Configuration
# =========================================================

st.set_page_config(
    page_title="Potato Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# =========================================================
# 2. Load Trained Custom CNN Model
# =========================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "explainable_cnn_model.keras"
    )
    return model


model = load_model()


# =========================================================
# 3. Class Names
# =========================================================

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]


# User-friendly class names
display_names = {
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Potato___healthy": "Healthy Potato Leaf"
}


# =========================================================
# 4. Image Preprocessing
# =========================================================

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize according to model input
    image = image.resize((224, 224))

    # Convert to NumPy array
    img_array = np.array(image).astype("float32")

    # Same normalization used during training
    img_array = img_array / 255.0

    # Add batch dimension
    # Shape: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# =========================================================
# 5. Main UI
# =========================================================

st.title("🌿 Potato Plant Disease Detection System")

st.markdown(
    """
    Upload an image of a potato leaf and the trained
    Custom CNN model will predict the disease.
    """
)

st.divider()


# =========================================================
# 6. Image Upload
# =========================================================

uploaded_file = st.file_uploader(
    "Choose a potato leaf image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# 7. Prediction
# =========================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.subheader("Uploaded Leaf Image")

    st.image(
        image,
        caption="Uploaded Potato Leaf",
        use_container_width=True
    )

    st.divider()

    # Analyze button
    if st.button(
        "🔍 Analyze Image",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            try:

                # -----------------------------------------
                # Preprocess Image
                # -----------------------------------------

                processed_image = preprocess_image(image)


                # -----------------------------------------
                # Model Prediction
                # -----------------------------------------

                predictions = model.predict(
                    processed_image,
                    verbose=0
                )


                # -----------------------------------------
                # Get Predicted Class
                # -----------------------------------------

                predicted_index = np.argmax(
                    predictions[0]
                )

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = (
                    float(
                        predictions[0][predicted_index]
                    ) * 100
                )


                # -----------------------------------------
                # Display Prediction
                # -----------------------------------------

                st.subheader("Prediction Result")

                st.success(
                    f"Prediction: "
                    f"{display_names[predicted_class]}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


                # -----------------------------------------
                # Health Status
                # -----------------------------------------

                if predicted_class == "Potato___healthy":

                    st.success(
                        "✅ The potato leaf appears healthy."
                    )

                elif predicted_class == "Potato___Early_blight":

                    st.warning(
                        "⚠️ Early Blight detected."
                    )

                elif predicted_class == "Potato___Late_blight":

                    st.warning(
                        "⚠️ Late Blight detected."
                    )


                # -----------------------------------------
                # Class Probabilities
                # -----------------------------------------

                st.subheader("Class Probabilities")

                for i, class_name in enumerate(
                    class_names
                ):

                    probability = (
                        float(predictions[0][i]) * 100
                    )

                    st.write(
                        f"{display_names[class_name]}: "
                        f"{probability:.2f}%"
                    )

                    st.progress(
                        min(probability / 100, 1.0)
                    )


            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )


# =========================================================
# 8. Footer
# =========================================================

st.divider()

st.caption(
    "Developed for Thesis/Capstone Research | "
    "Custom Lightweight CNN"
)
