import os
import json
import time
import csv
import io

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Potato Plant Disease AI",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_PATH = "potato_gradcam_model.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

DISPLAY_NAMES = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy"
}


# ============================================================
# 3. DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Potato___Early_blight": {

        "name": "Early Blight",

        "description":
            "A common potato leaf disease that can reduce "
            "plant growth and yield.",

        "symptoms": [
            "Brown or dark spots on leaves",
            "Circular or target-like patterns may appear",
            "Older leaves are often affected first",
            "Severely affected leaves can yellow and die"
        ],

        "causes": [
            "Fungal infection",
            "Warm and humid conditions",
            "Poor air circulation",
            "Infected plant debris"
        ],

        "actions": [
            "Remove severely affected leaves when practical",
            "Improve plant spacing and air circulation",
            "Avoid unnecessary overhead irrigation",
            "Remove infected plant debris",
            "Use locally recommended fungicide practices when appropriate"
        ],

        "prevention": [
            "Use healthy planting material",
            "Maintain field sanitation",
            "Avoid prolonged leaf wetness",
            "Monitor plants regularly"
        ]
    },


    "Potato___Late_blight": {

        "name": "Late Blight",

        "description":
            "A potentially fast-spreading potato disease, "
            "especially under cool and humid conditions.",

        "symptoms": [
            "Dark brown or black irregular lesions",
            "Water-soaked appearance may occur",
            "Lesions can expand quickly",
            "Leaves can collapse when infection becomes severe"
        ],

        "causes": [
            "Pathogen infection",
            "Cool and humid weather",
            "Extended leaf wetness",
            "Poor field sanitation"
        ],

        "actions": [
            "Remove severely infected plant material where practical",
            "Improve air circulation",
            "Avoid prolonged leaf wetness",
            "Monitor nearby plants carefully",
            "Follow locally recommended disease-management guidance"
        ],

        "prevention": [
            "Use disease-free planting material",
            "Inspect plants frequently",
            "Maintain field sanitation",
            "Avoid excessive irrigation",
            "Act early when symptoms appear"
        ]
    },


    "Potato___healthy": {

        "name": "Healthy",

        "description":
            "The model classified the uploaded potato leaf as healthy.",

        "symptoms": [
            "No strong disease pattern detected by the model",
            "Leaf appearance is generally consistent with a healthy class"
        ],

        "causes": [
            "No disease pattern was detected by the model"
        ],

        "actions": [
            "Continue regular monitoring",
            "Maintain proper irrigation",
            "Maintain balanced plant nutrition",
            "Keep the field clean"
        ],

        "prevention": [
            "Use healthy planting material",
            "Monitor leaves regularly",
            "Maintain appropriate plant spacing",
            "Control pests and weeds",
            "Maintain field hygiene"
        ]
    }
}


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        return None, (
            f"Model file not found: {MODEL_PATH}"
        )

    try:

        loaded_model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return loaded_model, None

    except Exception as e:

        return None, str(e)


model, model_error = load_model()


# ============================================================
# 5. SESSION STATE
# ============================================================

defaults = {
    "image": None,
    "prediction": None,
    "probabilities": None,
    "confidence": None,
    "filename": None,
    "gradcam": None,
    "gradcam_layer": None
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# 6. PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = array / 255.0

    return np.expand_dims(
        array,
        axis=0
    )


# ============================================================
# 7. PREDICTION
# ============================================================

def predict_image(image):

    if model is None:

        raise RuntimeError(
            model_error or "Model unavailable."
        )

    processed = preprocess_image(
        image
    )

    output = model.predict(
        processed,
        verbose=0
    )

    output = np.asarray(
        output
    )

    if output.ndim == 2:

        probabilities = output[0]

    else:

        probabilities = output.reshape(-1)


    # Safety for logits
    if (
        np.min(probabilities) < 0
        or np.max(probabilities) > 1
        or not np.isclose(
            np.sum(probabilities),
            1.0,
            atol=1e-3
        )
    ):

        shifted = (
            probabilities
            - np.max(probabilities)
        )

        exp_values = np.exp(
            shifted
        )

        probabilities = (
            exp_values
            / np.sum(exp_values)
        )


    if len(probabilities) != len(CLASS_NAMES):

        raise ValueError(
            f"Model returned "
            f"{len(probabilities)} outputs, "
            f"but this application expects "
            f"{len(CLASS_NAMES)} classes."
        )


    index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        index
    ]

    confidence = float(
        probabilities[index]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# 8. CONFIDENCE LEVEL
# ============================================================

def confidence_level(confidence):

    if confidence >= 0.85:

        return "High", "🟢"

    elif confidence >= 0.60:

        return "Moderate", "🟡"

    else:

        return "Low", "🔴"


# ============================================================
# 9. TOP PREDICTIONS
# ============================================================

def get_top_predictions(
    probabilities,
    number=3
):

    indices = np.argsort(
        probabilities
    )[::-1][:number]

    results = []

    for index in indices:

        results.append(
            (
                CLASS_NAMES[
                    int(index)
                ],
                float(
                    probabilities[
                        int(index)
                    ]
                )
            )
        )

    return results


# ============================================================
# 10. FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer():

    if model is None:

        return None

    for layer in reversed(
        model.layers
    ):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            return layer

    return None


# ============================================================
# 11. BUILD GRAD-CAM MODEL
# ============================================================

@st.cache_resource
def get_gradcam_model():

    if model is None:

        return None, None

    conv_layer = find_last_conv_layer()

    if conv_layer is None:

        return None, None

    try:

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                conv_layer.output,
                model.output
            ]
        )

        return (
            grad_model,
            conv_layer.name
        )

    except Exception:

        return None, None


# ============================================================
# 12. GENERATE GRAD-CAM
# ============================================================

def make_gradcam(
    image,
    class_index
):

    grad_model, layer_name = (
        get_gradcam_model()
    )

    if grad_model is None:

        raise ValueError(
            "The convolutional layer could not "
            "be connected to the model output. "
            "Use the Functional API model saved "
            "from the Kaggle Grad-CAM notebook."
        )


    input_tensor = preprocess_image(
        image
    )

    with tf.GradientTape() as tape:

        conv_output, predictions = (
            grad_model(
                input_tensor
            )
        )

        score = predictions[
            :,
            class_index
        ]


    gradients = tape.gradient(
        score,
        conv_output
    )

    if gradients is None:

        raise ValueError(
            "Gradients were not produced."
        )


    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_output = conv_output[0]

    weights = weights[0]


    heatmap = tf.reduce_sum(
        conv_output * weights,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(
        heatmap
    )

    if float(maximum) > 0:

        heatmap = (
            heatmap / maximum
        )


    return (
        heatmap.numpy(),
        layer_name
    )


# ============================================================
# 13. CREATE HEATMAP
# ============================================================

def create_heatmap_image(
    heatmap
):

    heat = Image.fromarray(
        np.uint8(
            np.clip(
                heatmap * 255,
                0,
                255
            )
        )
    )

    heat = heat.resize(
        IMG_SIZE
    )

    array = (
        np.asarray(
            heat,
            dtype=np.float32
        )
        / 255.0
    )


    red = np.clip(
        2 * array,
        0,
        1
    )

    green = np.clip(
        2 * (
            1 - np.abs(
                array - 0.5
            ) * 2
        ),
        0,
        1
    )

    blue = np.clip(
        2 * (1 - array),
        0,
        1
    )


    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )


    return Image.fromarray(
        np.uint8(
            rgb * 255
        )
    )


# ============================================================
# 14. GRAD-CAM OVERLAY
# ============================================================

def create_overlay(
    image,
    heatmap
):

    original = (
        image
        .convert("RGB")
        .resize(IMG_SIZE)
    )

    heatmap_image = (
        create_heatmap_image(
            heatmap
        )
    )

    return Image.blend(
        original,
        heatmap_image,
        0.45
    )


# ============================================================
# 15. EXPERIMENTAL ATTENTION ESTIMATION
# ============================================================

def estimate_attention(
    heatmap
):

    if heatmap is None:

        return None, None

    threshold = 0.55

    mask = (
        heatmap >= threshold
    )

    percentage = (
        float(
            np.mean(mask)
        )
        * 100
    )


    if percentage < 10:

        level = "Low"

    elif percentage < 30:

        level = "Moderate"

    else:

        level = "High"


    return (
        percentage,
        level
    )


# ============================================================
# 16. DISEASE INFO
# ============================================================

def get_info(
    predicted_class
):

    return DISEASE_INFO[
        predicted_class
    ]


# ============================================================
# 17. TEST METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    n_classes
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64
    )


    matrix = np.zeros(
        (
            n_classes,
            n_classes
        ),
        dtype=np.int64
    )


    for actual, predicted in zip(
        y_true,
        y_pred
    ):

        if (
            0 <= actual < n_classes
            and
            0 <= predicted < n_classes
        ):

            matrix[
                actual,
                predicted
            ] += 1


    if len(y_true) == 0:

        return (
            0,
            0,
            0,
            0,
            matrix
        )


    accuracy = float(
        np.mean(
            y_true == y_pred
        )
    )


    precisions = []
    recalls = []
    f1_scores = []


    for i in range(
        n_classes
    ):

        tp = matrix[
            i,
            i
        ]

        fp = (
            np.sum(
                matrix[:, i]
            )
            - tp
        )

        fn = (
            np.sum(
                matrix[i, :]
            )
            - tp
        )


        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0
        )


        if (
            precision
            + recall
        ):

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0


        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1_scores.append(
            f1
        )


    return (
        accuracy,
        float(
            np.mean(
                precisions
            )
        ),
        float(
            np.mean(
                recalls
            )
        ),
        float(
            np.mean(
                f1_scores
            )
        ),
        matrix
    )


# ============================================================
# 18. READ TRAINING HISTORY
# ============================================================

def read_history():

    path = (
        "training_history.json"
    )

    if not os.path.exists(
        path
    ):

        return None


    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return None


# ============================================================
# 19. READ MODEL COMPARISON
# ============================================================

def read_model_comparison():

    path = (
        "model_comparison.json"
    )

    if not os.path.exists(
        path
    ):

        return None


    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return None


# ============================================================
# 20. DOWNLOAD REPORT
# ============================================================

def create_report():

    predicted_class = (
        st.session_state.prediction
    )

    confidence = (
        st.session_state.confidence
    )

    filename = (
        st.session_state.filename
        or "Uploaded image"
    )

    info = get_info(
        predicted_class
    )

    level, icon = (
        confidence_level(
            confidence
        )
    )


    report = []

    report.append(
        "POTATO PLANT DISEASE AI ANALYSIS REPORT"
    )

    report.append(
        "=" * 50
    )

    report.append(
        f"Image: {filename}"
    )

    report.append(
        f"Prediction: {info['name']}"
    )

    report.append(
        f"Confidence: {confidence * 100:.2f}%"
    )

    report.append(
        f"Reliability: {icon} {level}"
    )


    report.append(
        "\nDESCRIPTION"
    )

    report.append(
        info["description"]
    )


    report.append(
        "\nSYMPTOMS"
    )

    for item in info["symptoms"]:

        report.append(
            "- " + item
        )


    report.append(
        "\nPOSSIBLE CAUSES"
    )

    for item in info["causes"]:

        report.append(
            "- " + item
        )


    report.append(
        "\nRECOMMENDED ACTIONS"
    )

    for item in info["actions"]:

        report.append(
            "- " + item
        )


    report.append(
        "\nPREVENTION"
    )

    for item in info["prevention"]:

        report.append(
            "- " + item
        )


    report.append(
        "\nNOTE"
    )

    report.append(
        "This system provides AI-based preliminary "
        "classification and decision-support information."
    )


    return "\n".join(
        report
    )


# ============================================================
# 21. MODEL ERROR CHECK
# ============================================================

if model is None:

    st.error(
        "❌ Model could not be loaded."
    )

    st.code(
        model_error
        or
        "Unknown model error"
    )

    st.info(
        """
        Make sure `potato_gradcam_model.keras`
        is in the same GitHub folder as app.py.
        """
    )

    st.stop()


# ============================================================
# 22. SIDEBAR
# ============================================================

st.sidebar.title(
    "🥔 Navigation"
)

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Home",
        "🔍 Disease Detection",
        "🧠 Explainable AI",
        "📚 Disease Information",
        "🩺 Causes & Solutions",
        "🩺 Severity / Attention",
        "📄 Analysis Report",
        "📦 Batch Analysis",
        "📊 Performance",
        "⚖️ Model Comparison",
        "⚡ Model Efficiency"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Custom CNN • Explainable AI • Grad-CAM"
)


# ============================================================
# 23. HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "🥔 Potato Plant Disease Detection System"
    )

    st.markdown(
        """
        ### Advanced AI-powered potato leaf analysis

        Upload **one image only**. The same image will be
        automatically used by the other analysis sections.

        **Detection → Grad-CAM → Disease Information →
        Causes & Solutions → Attention Analysis → Report**
        """
    )


    uploaded_file = st.file_uploader(
        "📷 Upload Potato Leaf",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="main_uploader"
    )


    if uploaded_file:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")


            with st.spinner(
                "Analyzing leaf..."
            ):

                (
                    predicted_class,
                    confidence,
                    probabilities
                ) = predict_image(
                    image
                )


            st.session_state.image = (
                image
            )

            st.session_state.prediction = (
                predicted_class
            )

            st.session_state.confidence = (
                confidence
            )

            st.session_state.probabilities = (
                probabilities
            )

            st.session_state.filename = (
                uploaded_file.name
            )

            st.session_state.gradcam = None

            st.session_state.gradcam_layer = None


            level, icon = (
                confidence_level(
                    confidence
                )
            )


            c1, c2, c3 = st.columns(
                3
            )


            with c1:

                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )


            with c2:

                st.subheader(
                    "🎯 Prediction"
                )

                st.success(
                    DISPLAY_NAMES[
                        predicted_class
                    ]
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )


            with c3:

                st.subheader(
                    "Reliability"
                )

                st.metric(
                    "Level",
                    f"{icon} {level}"
                )


                if (
                    predicted_class
                    ==
                    "Potato___healthy"
                ):

                    st.success(
                        "✅ Healthy pattern detected."
                    )

                else:

                    st.warning(
                        "⚠️ Disease pattern detected."
                    )


            st.divider()


            st.subheader(
                "🔝 Top-3 Predictions"
            )


            for rank, (
                class_name,
                value
            ) in enumerate(
                get_top_predictions(
                    probabilities,
                    3
                ),
                start=1
            ):

                st.write(
                    f"**{rank}. "
                    f"{DISPLAY_NAMES[class_name]}** "
                    f"— {value * 100:.2f}%"
                )

                st.progress(
                    min(
                        max(
                            value,
                            0
                        ),
                        1
                    )
                )


            st.success(
                "✓ Analysis completed."
            )


        except Exception as e:

            st.error(
                "❌ Image analysis failed."
            )

            st.code(
                str(e)
            )


    elif st.session_state.image:

        st.info(
            "An image is already loaded."
        )

        st.image(
            st.session_state.image,
            caption=(
                st.session_state.filename
                or "Loaded image"
            )
        )


    else:

        st.info(
            "👆 Upload a potato leaf image."
        )


# ============================================================
# 24. CHECK SINGLE IMAGE
# ============================================================

elif page in [
    "🔍 Disease Detection",
    "🧠 Explainable AI",
    "📚 Disease Information",
    "🩺 Causes & Solutions",
    "🩺 Severity / Attention",
    "📄 Analysis Report"
]:

    if st.session_state.image is None:

        st.warning(
            """
            ⚠️ No image loaded.

            Go to Home and upload one potato leaf image first.
            """
        )

        st.stop()


# ============================================================
# 25. DISEASE DETECTION
# ============================================================

if page == "🔍 Disease Detection":

    st.title(
        "🔍 Disease Detection"
    )

    image = (
        st.session_state.image
    )

    predicted_class = (
        st.session_state.prediction
    )

    confidence = (
        st.session_state.confidence
    )

    probabilities = (
        st.session_state.probabilities
    )


    level, icon = (
        confidence_level(
            confidence
        )
    )


    c1, c2 = st.columns(
        2
    )


    with c1:

        st.image(
            image,
            caption="Analyzed Leaf",
            use_container_width=True
        )


    with c2:

        st.subheader(
            "Prediction"
        )

        st.success(
            DISPLAY_NAMES[
                predicted_class
            ]
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.metric(
            "Reliability",
            f"{icon} {level}"
        )


        if confidence < 0.60:

            st.warning(
                "⚠️ Low confidence. "
                "Try a clearer image."
            )


    st.divider()


    st.subheader(
        "🔝 Top-3 Predictions"
    )


    for rank, (
        class_name,
        value
    ) in enumerate(
        get_top_predictions(
            probabilities,
            3
        ),
        start=1
    ):

        st.write(
            f"**{rank}. "
            f"{DISPLAY_NAMES[class_name]}** "
            f"— {value * 100:.2f}%"
        )

        st.progress(
            min(
                max(value, 0),
                1
            )
        )


# ============================================================
# 26. EXPLAINABLE AI
# ============================================================

elif page == "🧠 Explainable AI":

    st.title(
        "🧠 Explainable AI — Grad-CAM"
    )

    image = (
        st.session_state.image
    )

    probabilities = (
        st.session_state.probabilities
    )

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    st.write(
        """
        Grad-CAM shows which image regions contributed
        most strongly to the CNN prediction.
        """
    )


    try:

        with st.spinner(
            "Generating Grad-CAM..."
        ):

            heatmap, layer_name = (
                make_gradcam(
                    image,
                    predicted_index
                )
            )


        st.session_state.gradcam = (
            heatmap
        )

        st.session_state.gradcam_layer = (
            layer_name
        )


        heatmap_image = (
            create_heatmap_image(
                heatmap
            )
        )

        overlay = (
            create_overlay(
                image,
                heatmap
            )
        )


        st.success(
            "✓ Grad-CAM generated successfully."
        )

        st.info(
            f"Convolutional Layer: `{layer_name}`"
        )


        c1, c2, c3 = st.columns(
            3
        )


        with c1:

            st.subheader(
                "Original"
            )

            st.image(
                image,
                use_container_width=True
            )


        with c2:

            st.subheader(
                "Heatmap"
            )

            st.image(
                heatmap_image,
                use_container_width=True
            )


        with c3:

            st.subheader(
                "Overlay"
            )

            st.image(
                overlay,
                use_container_width=True
            )


        st.divider()


        st.subheader(
            "🧠 Interpretation"
        )

        st.write(
            """
            Brighter regions indicate areas that contributed
            more strongly to the model's prediction. This can
            help determine whether the CNN is focusing on
            relevant leaf regions.
            """
        )


    except Exception as e:

        st.error(
            "❌ Grad-CAM could not be generated."
        )

        st.code(
            str(e)
        )

        st.warning(
            """
            Make sure the GitHub model is the Functional API
            model saved from your Kaggle Grad-CAM notebook:
            
            potato_gradcam_model.keras
            """
        )


# ============================================================
# 27. DISEASE INFORMATION
# ============================================================

elif page == "📚 Disease Information":

    st.title(
        "📚 Disease Information"
    )

    predicted_class = (
        st.session_state.prediction
    )

    confidence = (
        st.session_state.confidence
    )

    info = get_info(
        predicted_class
    )


    st.subheader(
        f"🌿 {info['name']}"
    )

    st.write(
        info["description"]
    )

    st.metric(
        "Model Confidence",
        f"{confidence * 100:.2f}%"
    )


    st.divider()


    st.subheader(
        "🔎 Symptoms"
    )

    for item in info[
        "symptoms"
    ]:

        st.write(
            "• " + item
        )


    st.divider()


    st.subheader(
        "🦠 Possible Causes"
    )

    for item in info[
        "causes"
    ]:

        st.write(
            "• " + item
        )


# ============================================================
# 28. CAUSES & SOLUTIONS
# ============================================================

elif page == "🩺 Causes & Solutions":

    st.title(
        "🩺 Causes & Solutions"
    )

    predicted_class = (
        st.session_state.prediction
    )

    info = get_info(
        predicted_class
    )


    st.subheader(
        f"Detected Condition: {info['name']}"
    )


    st.divider()


    st.subheader(
        "❓ Why can this problem happen?"
    )

    for item in info[
        "causes"
    ]:

        st.write(
            "• " + item
        )


    st.divider()


    st.subheader(
        "⚡ What should be done first?"
    )

    for item in info[
        "actions"
    ]:

        st.write(
            "✅ " + item
        )


    st.divider()


    st.subheader(
        "🛡️ Prevention"
    )

    for item in info[
        "prevention"
    ]:

        st.write(
            "• " + item
        )


    st.warning(
        """
        ⚠️ This application provides AI-based preliminary
        classification and general management information.
        Follow local agricultural recommendations for treatment.
        """
    )


# ============================================================
# 29. VISUAL ATTENTION / EXPERIMENTAL SEVERITY
# ============================================================

elif page == "🩺 Severity / Attention":

    st.title(
        "🩺 Visual Attention Analysis"
    )

    st.info(
        """
        This is an experimental Grad-CAM-based visual attention
        indicator. It is NOT a clinically/agronomically validated
        disease-severity percentage because the current CNN was
        trained for disease classification, not severity classes.
        """
    )


    image = (
        st.session_state.image
    )

    predicted_index = int(
        np.argmax(
            st.session_state.probabilities
        )
    )


    try:

        heatmap, layer_name = (
            make_gradcam(
                image,
                predicted_index
            )
        )


        area, level = (
            estimate_attention(
                heatmap
            )
        )


        c1, c2 = st.columns(
            2
        )


        with c1:

            st.metric(
                "High-activation area",
                f"{area:.2f}%"
            )


        with c2:

            st.metric(
                "Attention Level",
                level
            )


        st.image(
            create_overlay(
                image,
                heatmap
            ),
            caption="Grad-CAM Visual Attention",
            use_container_width=True
        )


        st.write(
            """
            A larger highlighted region means the model's
            prediction is distributed across a larger image area.
            It should not be interpreted as the exact percentage
            of diseased tissue.
            """
        )


    except Exception as e:

        st.error(
            "Visual attention analysis failed."
        )

        st.code(
            str(e)
        )


# ============================================================
# 30. ANALYSIS REPORT
# ============================================================

elif page == "📄 Analysis Report":

    st.title(
        "📄 Analysis Report"
    )

    report = create_report()


    st.text_area(
        "Report Preview",
        report,
        height=500
    )


    st.download_button(
        "⬇️ Download Analysis Report",
        data=report.encode(
            "utf-8"
        ),
        file_name=(
            "potato_disease_report.txt"
        ),
        mime="text/plain"
    )


# ============================================================
# 31. BATCH ANALYSIS
# ============================================================

elif page == "📦 Batch Analysis":

    st.title(
        "📦 Batch Analysis"
    )

    st.write(
        """
        Upload multiple potato leaf images and analyze
        them together.
        """
    )


    files = st.file_uploader(
        "Upload multiple images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True,
        key="batch_uploader"
    )


    if files:

        rows = []

        progress = st.progress(
            0
        )

        start_time = time.time()


        for index, file in enumerate(
            files
        ):

            try:

                image = Image.open(
                    file
                ).convert("RGB")


                (
                    predicted_class,
                    confidence,
                    probabilities
                ) = predict_image(
                    image
                )


                level, _ = (
                    confidence_level(
                        confidence
                    )
                )


                rows.append(
                    {
                        "Image": file.name,
                        "Prediction":
                            DISPLAY_NAMES[
                                predicted_class
                            ],
                        "Confidence":
                            f"{confidence * 100:.2f}%",
                        "Reliability":
                            level
                    }
                )


            except Exception as e:

                rows.append(
                    {
                        "Image": file.name,
                        "Prediction": "Error",
                        "Confidence": "N/A",
                        "Reliability": str(e)
                    }
                )


            progress.progress(
                int(
                    (
                        (index + 1)
                        /
                        len(files)
                    )
                    * 100
                )
            )


        elapsed = (
            time.time()
            - start_time
        )


        st.success(
            f"✓ {len(files)} images analyzed "
            f"in {elapsed:.2f} seconds."
        )


        st.dataframe(
            rows,
            use_container_width=True
        )


        healthy = sum(
            1
            for row in rows
            if row["Prediction"]
            == "Healthy"
        )


        early = sum(
            1
            for row in rows
            if row["Prediction"]
            == "Early Blight"
        )


        late = sum(
            1
            for row in rows
            if row["Prediction"]
            == "Late Blight"
        )


        c1, c2, c3, c4 = (
            st.columns(4)
        )


        with c1:

            st.metric(
                "Total",
                len(files)
            )


        with c2:

            st.metric(
                "Healthy",
                healthy
            )


        with c3:

            st.metric(
                "Early Blight",
                early
            )


        with c4:

            st.metric(
                "Late Blight",
                late
            )


        st.subheader(
            "📊 Disease Distribution"
        )


        chart_data = {
            "Class": [
                "Healthy",
                "Early Blight",
                "Late Blight"
            ],
            "Images": [
                healthy,
                early,
                late
            ]
        }


        st.bar_chart(
            chart_data,
            x="Class",
            y="Images"
        )


        csv_output = io.StringIO()


        writer = csv.DictWriter(
            csv_output,
            fieldnames=[
                "Image",
                "Prediction",
                "Confidence",
                "Reliability"
            ]
        )


        writer.writeheader()

        writer.writerows(
            rows
        )


        st.download_button(
            "⬇️ Download Batch CSV",
            data=csv_output.getvalue().encode(
                "utf-8-sig"
            ),
            file_name=(
                "batch_analysis.csv"
            ),
            mime="text/csv"
        )


    else:

        st.info(
            "Upload multiple images to start."
        )


# ============================================================
# 32. PERFORMANCE
# ============================================================

elif page == "📊 Performance":

    st.title(
        "📊 Model Performance"
    )

    st.write(
        """
        Real evaluation metrics are displayed only when
        a compatible test dataset or exported training history
        is available.
        """
    )


    test_candidates = [
        "test",
        "Test",
        "dataset/test",
        "data/test"
    ]


    test_dir = None


    for candidate in test_candidates:

        if os.path.isdir(
            candidate
        ):

            test_dir = candidate

            break


    if test_dir is None:

        st.warning(
            """
            No test dataset folder was found.

            A `.keras` model alone cannot reconstruct the
            original test Accuracy, Precision, Recall,
            F1 Score and Confusion Matrix.
            """
        )


    else:

        try:

            test_ds = (
                tf.keras.utils
                .image_dataset_from_directory(
                    test_dir,
                    image_size=IMG_SIZE,
                    batch_size=32,
                    shuffle=False,
                    class_names=CLASS_NAMES
                )
            )


            y_true = []

            y_pred = []


            start = time.time()


            for images, labels in test_ds:

                images = (
                    tf.cast(
                        images,
                        tf.float32
                    )
                    / 255.0
                )


                outputs = (
                    model.predict(
                        images,
                        verbose=0
                    )
                )


                y_true.extend(
                    labels.numpy().tolist()
                )


                y_pred.extend(
                    np.argmax(
                        outputs,
                        axis=1
                    ).tolist()
                )


            elapsed = (
                time.time()
                - start
            )


            (
                accuracy,
                precision,
                recall,
                f1,
                matrix
            ) = calculate_metrics(
                y_true,
                y_pred,
                len(CLASS_NAMES)
            )


            st.subheader(
                "🎯 Evaluation Components"
            )


            c1, c2, c3, c4 = (
                st.columns(4)
            )


            with c1:

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )


            with c2:

                st.metric(
                    "Precision",
                    f"{precision * 100:.2f}%"
                )


            with c3:

                st.metric(
                    "Recall",
                    f"{recall * 100:.2f}%"
                )


            with c4:

                st.metric(
                    "F1 Score",
                    f"{f1 * 100:.2f}%"
                )


            st.divider()


            st.subheader(
                "🔲 Confusion Matrix"
            )


            display_names = [
                DISPLAY_NAMES[
                    x
                ]
                for x in CLASS_NAMES
            ]


            matrix_rows = []


            for i, actual in enumerate(
                display_names
            ):

                row = {
                    "Actual": actual
                }


                for j, predicted in enumerate(
                    display_names
                ):

                    row[predicted] = int(
                        matrix[
                            i,
                            j
                        ]
                    )


                matrix_rows.append(
                    row
                )


            st.dataframe(
                matrix_rows,
                use_container_width=True
            )


            st.divider()


            correct = int(
                np.sum(
                    np.asarray(
                        y_true
                    )
                    ==
                    np.asarray(
                        y_pred
                    )
                )
            )


            incorrect = (
                len(y_true)
                - correct
            )


            c1, c2, c3 = (
                st.columns(3)
            )


            with c1:

                st.metric(
                    "Test Images",
                    len(y_true)
                )


            with c2:

                st.metric(
                    "Correct",
                    correct
                )


            with c3:

                st.metric(
                    "Incorrect",
                    incorrect
                )


            st.caption(
                f"Evaluation time: "
                f"{elapsed:.2f} seconds"
            )


        except Exception as e:

            st.error(
                "❌ Test evaluation failed."
            )

            st.code(
                str(e)
            )


    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📈 Training / Validation Performance"
    )


    history = read_history()


    if history is None:

        st.info(
            """
            `training_history.json` was not found.

            Export your Kaggle `history.history` dictionary
            as JSON and put it beside app.py.
            """
        )


    else:

        try:

            acc_key = (
                "accuracy"
                if "accuracy" in history
                else
                "acc"
                if "acc" in history
                else None
            )


            val_acc_key = (
                "val_accuracy"
                if "val_accuracy" in history
                else
                "val_acc"
                if "val_acc" in history
                else None
            )


            loss_key = (
                "loss"
                if "loss" in history
                else None
            )


            val_loss_key = (
                "val_loss"
                if "val_loss" in history
                else None
            )


            if (
                acc_key
                and
                val_acc_key
            ):

                chart = []


                length = min(
                    len(
                        history[
                            acc_key
                        ]
                    ),
                    len(
                        history[
                            val_acc_key
                        ]
                    )
                )


                for i in range(
                    length
                ):

                    chart.append(
                        {
                            "Epoch":
                                i + 1,

                            "Training Accuracy":
                                history[
                                    acc_key
                                ][i],

                            "Validation Accuracy":
                                history[
                                    val_acc_key
                                ][i]
                        }
                    )


                st.write(
                    "Training vs Validation Accuracy"
                )


                st.line_chart(
                    chart,
                    x="Epoch",
                    y=[
                        "Training Accuracy",
                        "Validation Accuracy"
                    ]
                )


            if (
                loss_key
                and
                val_loss_key
            ):

                chart = []


                length = min(
                    len(
                        history[
                            loss_key
                        ]
                    ),
                    len(
                        history[
                            val_loss_key
                        ]
                    )
                )


                for i in range(
                    length
                ):

                    chart.append(
                        {
                            "Epoch":
                                i + 1,

                            "Training Loss":
                                history[
                                    loss_key
                                ][i],

                            "Validation Loss":
                                history[
                                    val_loss_key
                                ][i]
                        }
                    )


                st.write(
                    "Training vs Validation Loss"
                )


                st.line_chart(
                    chart,
                    x="Epoch",
                    y=[
                        "Training Loss",
                        "Validation Loss"
                    ]
                )


        except Exception as e:

            st.error(
                "Training history could not be displayed."
            )

            st.code(
                str(e)
            )


# ============================================================
# 33. MODEL COMPARISON
# ============================================================

elif page == "⚖️ Model Comparison":

    st.title(
        "⚖️ Model Comparison"
    )


    comparison = (
        read_model_comparison()
    )


    if comparison is None:

        st.warning(
            """
            `model_comparison.json` was not found.

            I am not putting fake Accuracy, Loss or Parameter
            values into your thesis application.
            """
        )


    else:

        st.subheader(
            "📊 Experimental Results"
        )


        if isinstance(
            comparison,
            list
        ):

            st.dataframe(
                comparison,
                use_container_width=True
            )


        elif isinstance(
            comparison,
            dict
        ):

            if isinstance(
                comparison.get(
                    "rows"
                ),
                list
            ):

                st.dataframe(
                    comparison[
                        "rows"
                    ],
                    use_container_width=True
                )

            else:

                st.json(
                    comparison
                )


    st.divider()


    st.subheader(
        "🏗️ Architecture Comparison"
    )


    architecture = [
        {
            "Model":
                "Custom CNN",

            "Type":
                "Proposed Lightweight CNN",

            "Input":
                "224 × 224 × 3",

            "Classes":
                "3",

            "Purpose":
                "Lightweight disease classification"
        },

        {
            "Model":
                "ResNet50",

            "Type":
                "Transfer Learning",

            "Input":
                "224 × 224 × 3",

            "Classes":
                "3",

            "Purpose":
                "Baseline comparison"
        },

        {
            "Model":
                "MobileNetV2",

            "Type":
                "Transfer Learning",

            "Input":
                "224 × 224 × 3",

            "Classes":
                "3",

            "Purpose":
                "Lightweight transfer-learning comparison"
        }
    ]


    st.dataframe(
        architecture,
        use_container_width=True
    )


# ============================================================
# 34. MODEL EFFICIENCY
# ============================================================

elif page == "⚡ Model Efficiency":

    st.title(
        "⚡ Model Efficiency"
    )


    st.write(
        """
        This section measures the deployed Custom CNN itself.
        """
    )


    try:

        parameters = (
            model.count_params()
        )


        model_size = (
            os.path.getsize(
                MODEL_PATH
            )
            /
            (1024 * 1024)
        )


        dummy = np.zeros(
            (
                1,
                224,
                224,
                3
            ),
            dtype=np.float32
        )


        # Warm-up
        model.predict(
            dummy,
            verbose=0
        )


        runs = 5


        start = (
            time.perf_counter()
        )


        for _ in range(
            runs
        ):

            model.predict(
                dummy,
                verbose=0
            )


        elapsed = (
            time.perf_counter()
            - start
        )


        inference_ms = (
            elapsed
            /
            runs
        ) * 1000


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            st.metric(
                "Parameters",
                f"{parameters:,}"
            )


        with c2:

            st.metric(
                "Model Size",
                f"{model_size:.2f} MB"
            )


        with c3:

            st.metric(
                "Inference Time",
                f"{inference_ms:.2f} ms"
            )


        st.success(
            "✓ Values measured from the deployed model."
        )


    except Exception as e:

        st.error(
            "Model efficiency calculation failed."
        )

        st.code(
            str(e)
        )


    st.divider()


    st.subheader(
        "🤖 Other Model Comparison"
    )


    comparison = (
        read_model_comparison()
    )


    if comparison is not None:

        if isinstance(
            comparison,
            list
        ):

            st.dataframe(
                comparison,
                use_container_width=True
            )

        else:

            st.json(
                comparison
            )

    else:

        st.info(
            "Add model_comparison.json to show "
            "Custom CNN vs ResNet50 vs MobileNetV2."
        )


# ============================================================
# 35. FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "🥔 Potato Plant Disease Detection"
)

st.sidebar.caption(
    "Custom CNN • Grad-CAM • Explainable AI"
)
