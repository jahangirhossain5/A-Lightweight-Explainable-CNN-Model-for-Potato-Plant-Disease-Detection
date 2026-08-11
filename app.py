import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from io import BytesIO


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_PATH = "potato_gradcam_model.keras"

IMG_SIZE = (224, 224)

# Your PlantVillage 38 classes
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# 3. DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "cause": "Usually associated with the fungal pathogen Alternaria solani.",
        "symptoms": [
            "Dark brown circular or target-like spots",
            "Older leaves are commonly affected first",
            "Yellowing may appear around lesions",
            "Severe infection can cause leaf loss"
        ],
        "solution": [
            "Remove heavily infected plant material",
            "Avoid prolonged leaf wetness",
            "Maintain good field sanitation",
            "Use locally recommended fungicide management when appropriate"
        ],
        "prevention": [
            "Use healthy planting material",
            "Maintain proper spacing",
            "Avoid unnecessary overhead irrigation",
            "Remove infected crop debris"
        ]
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "cause": "Caused by the oomycete pathogen Phytophthora infestans.",
        "symptoms": [
            "Irregular dark lesions",
            "Rapid browning of leaves",
            "Water-soaked appearance may occur",
            "Disease can spread rapidly under favorable conditions"
        ],
        "solution": [
            "Remove severely infected plant material where practical",
            "Improve field ventilation",
            "Avoid prolonged leaf wetness",
            "Follow locally recommended late-blight management practices"
        ],
        "prevention": [
            "Use healthy seed tubers",
            "Monitor crops frequently",
            "Avoid unnecessary overhead irrigation",
            "Follow local disease-management recommendations"
        ]
    },

    "Potato___healthy": {
        "name": "Healthy Potato Plant",
        "cause": "No disease pattern was detected by the model.",
        "symptoms": [
            "No major disease pattern detected",
            "Leaf appearance is consistent with the healthy class"
        ],
        "solution": [
            "Continue regular monitoring",
            "Maintain appropriate irrigation",
            "Maintain balanced plant nutrition"
        ],
        "prevention": [
            "Use healthy planting material",
            "Maintain field sanitation",
            "Monitor plants regularly"
        ]
    }
}


# ============================================================
# 4. GENERIC INFORMATION FOR OTHER PLANT DISEASES
# ============================================================

def get_disease_info(class_name):

    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]

    display_name = class_name.replace("___", " - ").replace("_", " ")

    return {
        "name": display_name,
        "cause": "The model classified the uploaded image into this disease class.",
        "symptoms": [
            "The prediction is based on visual features learned by the CNN.",
            "Check the plant carefully for matching visible symptoms."
        ],
        "solution": [
            "Remove severely affected plant material where appropriate.",
            "Maintain good crop hygiene.",
            "Consult a local agricultural expert for treatment decisions."
        ],
        "prevention": [
            "Use healthy planting material.",
            "Maintain field sanitation.",
            "Monitor plants regularly."
        ]
    }


# ============================================================
# 5. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        return None

    try:

        loaded_model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return loaded_model

    except Exception as e:

        st.error(
            "❌ Model could not be loaded."
        )

        st.code(str(e))

        return None


model = load_model()


# ============================================================
# 6. PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    img_array = np.asarray(
        image
    ).astype("float32")

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# 7. PREDICTION
# ============================================================

def predict_image(image):

    if model is None:
        return None, None, None

    img_array = preprocess_image(
        image
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predictions = predictions[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# ============================================================
# 8. FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer():

    if model is None:
        return None

    conv_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        )
    ]

    if len(conv_layers) == 0:
        return None

    return conv_layers[-1].name


# ============================================================
# 9. GRAD-CAM
# ============================================================

def make_gradcam_heatmap(
    img_array,
    pred_index
):

    if model is None:
        return None

    last_conv_layer_name = (
        find_last_conv_layer()
    )

    if last_conv_layer_name is None:
        return None

    try:

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(
                    last_conv_layer_name
                ).output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = (
                grad_model(
                    img_array
                )
            )

            class_channel = predictions[
                :,
                pred_index
            ]

        grads = tape.gradient(
            class_channel,
            conv_outputs
        )

        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = (
            conv_outputs
            @ pooled_grads[..., tf.newaxis]
        )

        heatmap = tf.squeeze(
            heatmap
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        if float(max_value) > 0:

            heatmap = (
                heatmap / max_value
            )

        return heatmap.numpy()

    except Exception as e:

        st.session_state[
            "gradcam_error"
        ] = str(e)

        return None


# ============================================================
# 10. CREATE GRAD-CAM IMAGE
# ============================================================

def create_gradcam_image(
    original_image,
    heatmap,
    alpha=0.45
):

    if heatmap is None:
        return None

    try:

        # Resize heatmap
        heatmap_img = Image.fromarray(
            np.uint8(
                heatmap * 255
            )
        )

        heatmap_img = heatmap_img.resize(
            original_image.size
        )

        heatmap_array = np.asarray(
            heatmap_img
        )

        # Simple red/yellow style heatmap
        heatmap_rgb = np.zeros(
            (
                heatmap_array.shape[0],
                heatmap_array.shape[1],
                3
            ),
            dtype=np.uint8
        )

        heatmap_rgb[:, :, 0] = 255

        heatmap_rgb[:, :, 1] = heatmap_array

        heatmap_rgb[:, :, 2] = 0

        heatmap_rgb = Image.fromarray(
            heatmap_rgb
        ).convert("RGBA")

        original = (
            original_image
            .convert("RGBA")
        )

        overlay = Image.blend(
            original,
            heatmap_rgb,
            alpha
        )

        return overlay.convert("RGB")

    except Exception:
        return None


# ============================================================
# 11. SESSION STATE
# ============================================================

if "uploaded_image" not in st.session_state:

    st.session_state[
        "uploaded_image"
    ] = None


if "prediction_done" not in st.session_state:

    st.session_state[
        "prediction_done"
    ] = False


if "prediction_data" not in st.session_state:

    st.session_state[
        "prediction_data"
    ] = None


# ============================================================
# 12. HEADER
# ============================================================

st.title(
    "🌿 Plant Disease Detection & Explainable AI"
)

st.markdown(
    """
    Upload **one plant leaf image** below.

    The same image is automatically used for:
    **Disease Detection, Explainable AI, Disease Information,
    Causes & Solutions, and Performance analysis.**
    """
)


# ============================================================
# 13. SINGLE IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.session_state[
        "uploaded_image"
    ] = image

    st.session_state[
        "prediction_done"
    ] = False


# ============================================================
# 14. PROCESS IMAGE ONCE
# ============================================================

if st.session_state[
    "uploaded_image"
] is not None:

    image = st.session_state[
        "uploaded_image"
    ]

    st.divider()

    st.subheader(
        "📷 Uploaded Image"
    )

    st.image(
        image,
        use_container_width=True
    )

    if st.button(
        "🔍 Analyze Image",
        type="primary"
    ):

        if model is None:

            st.error(
                "Model file was not found. "
                f"Please upload {MODEL_PATH} to GitHub."
            )

        else:

            with st.spinner(
                "Analyzing image..."
            ):

                result = predict_image(
                    image
                )

                st.session_state[
                    "prediction_data"
                ] = result

                st.session_state[
                    "prediction_done"
                ] = True


# ============================================================
# 15. RESULTS
# ============================================================

if (
    st.session_state[
        "prediction_done"
    ]
    and
    st.session_state[
        "prediction_data"
    ] is not None
):

    (
        predicted_class,
        confidence,
        predictions
    ) = st.session_state[
        "prediction_data"
    ]

    disease_info = get_disease_info(
        predicted_class
    )

    display_name = disease_info[
        "name"
    ]


    # ========================================================
    # MAIN RESULT
    # ========================================================

    st.divider()

    st.header(
        "🔍 Disease Detection"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"**Prediction:** {display_name}"
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ========================================================
    # TABS
    # ========================================================

    tabs = st.tabs(
        [
            "🔍 Disease Detection",
            "🧠 Explainable AI",
            "📋 Disease Information",
            "⚠️ Causes & Solutions",
            "📊 Performance",
            "🤖 Model Comparison",
            "📦 Batch Analysis"
        ]
    )


    # ========================================================
    # TAB 1 - DISEASE DETECTION
    # ========================================================

    with tabs[0]:

        st.subheader(
            "Prediction Result"
        )

        st.write(
            f"### {display_name}"
        )

        st.write(
            f"Model confidence: "
            f"**{confidence * 100:.2f}%**"
        )

        st.subheader(
            "Class Probabilities"
        )

        # Show only classes with meaningful probabilities
        top_indices = np.argsort(
            predictions
        )[::-1][:10]

        for index in top_indices:

            probability = float(
                predictions[index]
            )

            if probability > 0.001:

                class_display = (
                    CLASS_NAMES[index]
                    .replace("___", " - ")
                    .replace("_", " ")
                )

                st.write(
                    f"**{class_display}** "
                    f"— {probability * 100:.2f}%"
                )

                st.progress(
                    min(
                        probability,
                        1.0
                    )
                )


    # ========================================================
    # TAB 2 - EXPLAINABLE AI
    # ========================================================

    with tabs[1]:

        st.subheader(
            "🧠 Explainable AI — Grad-CAM"
        )

        st.write(
            """
            Grad-CAM highlights image regions that contributed
            to the CNN prediction.
            """
        )

        img_array = preprocess_image(
            image
        )

        heatmap = make_gradcam_heatmap(
            img_array,
            np.argmax(predictions)
        )

        if heatmap is not None:

            gradcam_image = (
                create_gradcam_image(
                    image,
                    heatmap
                )
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:

                st.image(
                    gradcam_image,
                    caption="Grad-CAM Explanation",
                    use_container_width=True
                )

            st.success(
                "✅ Grad-CAM successfully generated."
            )

            last_layer = (
                find_last_conv_layer()
            )

            if last_layer:

                st.info(
                    f"Grad-CAM target layer: "
                    f"`{last_layer}`"
                )

        else:

            st.error(
                "❌ Grad-CAM could not be generated."
            )

            st.warning(
                """
                The prediction model is working, but the
                loaded model does not expose a compatible
                convolutional graph for Grad-CAM.

                Make sure the GitHub model is the Functional
                model saved from the Kaggle Grad-CAM notebook:
                `potato_gradcam_model.keras`
                """
            )

            if "gradcam_error" in st.session_state:

                with st.expander(
                    "Technical error"
                ):

                    st.code(
                        st.session_state[
                            "gradcam_error"
                        ]
                    )


    # ========================================================
    # TAB 3 - DISEASE INFORMATION
    # ========================================================

    with tabs[2]:

        st.subheader(
            "📋 Disease Information"
        )

        st.write(
            f"## {display_name}"
        )

        st.markdown(
            "### 🦠 Cause"
        )

        st.write(
            disease_info[
                "cause"
            ]
        )

        st.markdown(
            "### 🔎 Common Symptoms"
        )

        for symptom in disease_info[
            "symptoms"
        ]:

            st.write(
                f"• {symptom}"
            )


    # ========================================================
    # TAB 4 - CAUSES & SOLUTIONS
    # ========================================================

    with tabs[3]:

        st.subheader(
            "⚠️ Why does this problem happen?"
        )

        st.write(
            disease_info[
                "cause"
            ]
        )

        st.markdown(
            "### 🚑 What should be done?"
        )

        for solution in disease_info[
            "solution"
        ]:

            st.write(
                f"✅ {solution}"
            )

        st.markdown(
            "### 🛡️ How to prevent it?"
        )

        for prevention in disease_info[
            "prevention"
        ]:

            st.write(
                f"🛡️ {prevention}"
            )

        st.info(
            """
            Treatment decisions should be based on local
            agricultural recommendations and the actual
            field condition. The model prediction should
            be treated as decision-support information.
            """
        )


    # ========================================================
    # TAB 5 - PERFORMANCE
    # ========================================================

    with tabs[4]:

        st.subheader(
            "📊 Model Performance"
        )

        st.info(
            """
            Accuracy, Precision, Recall, F1 Score and
            Confusion Matrix cannot be calculated from
            a single uploaded image.

            They must be calculated using the test dataset.
            """
        )

        st.markdown(
            "### Evaluation Components"
        )

        st.write(
            "✓ Accuracy"
        )

        st.write(
            "✓ Precision"
        )

        st.write(
            "✓ Recall"
        )

        st.write(
            "✓ F1 Score"
        )

        st.write(
            "✓ Confusion Matrix"
        )

        st.write(
            "✓ Training / Validation Performance"
        )

        st.warning(
            """
            To display your real Kaggle evaluation values
            here, export those results from Kaggle and add
            them to GitHub. The app does not invent
            performance numbers.
            """
        )


    # ========================================================
    # TAB 6 - MODEL COMPARISON
    # ========================================================

    with tabs[5]:

        st.subheader(
            "🤖 Model Comparison"
        )

        st.write(
            """
            Your project compares:

            • Custom Lightweight CNN

            • ResNet50

            • MobileNetV2
            """
        )

        st.info(
            """
            Model comparison values should come from the
            actual Kaggle test results. They are not generated
            from the uploaded image.
            """
        )

        st.markdown(
            "### Recommended Comparison Table"
        )

        comparison_data = {
            "Model": [
                "Custom CNN",
                "ResNet50",
                "MobileNetV2"
            ],
            "Test Accuracy": [
                "From Kaggle",
                "From Kaggle",
                "From Kaggle"
            ],
            "Test Loss": [
                "From Kaggle",
                "From Kaggle",
                "From Kaggle"
            ],
            "Parameters": [
                "From model summary",
                "From model summary",
                "From model summary"
            ]
        }

        st.table(
            comparison_data
        )


    # ========================================================
    # TAB 7 - BATCH ANALYSIS
    # ========================================================

    with tabs[6]:

        st.subheader(
            "📦 Batch Analysis"
        )

        st.write(
            """
            Upload multiple leaf images to analyze them
            together.
            """
        )

        batch_files = st.file_uploader(
            "Upload multiple images",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            accept_multiple_files=True,
            key="batch_uploader"
        )

        if batch_files:

            rows = []

            progress = st.progress(
                0
            )

            total = len(
                batch_files
            )

            for i, batch_file in enumerate(
                batch_files
            ):

                try:

                    batch_image = Image.open(
                        batch_file
                    ).convert("RGB")

                    (
                        batch_class,
                        batch_confidence,
                        _
                    ) = predict_image(
                        batch_image
                    )

                    batch_info = (
                        get_disease_info(
                            batch_class
                        )
                    )

                    rows.append(
                        {
                            "Image": batch_file.name,
                            "Prediction": batch_info[
                                "name"
                            ],
                            "Confidence": (
                                f"{batch_confidence * 100:.2f}%"
                            )
                        }
                    )

                except Exception as e:

                    rows.append(
                        {
                            "Image": batch_file.name,
                            "Prediction": "Error",
                            "Confidence": str(e)
                        }
                    )

                progress.progress(
                    (i + 1) / total
                )

            st.dataframe(
                rows,
                use_container_width=True
            )


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.caption(
    "Potato / Plant Disease Detection System | "
    "Custom Lightweight CNN | Explainable AI"
)
