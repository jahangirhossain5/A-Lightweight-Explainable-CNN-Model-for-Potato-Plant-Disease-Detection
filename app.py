import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import time
import csv
from io import BytesIO
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as ReportLabImage, PageBreak
)
from reportlab.lib.utils import ImageReader

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Potato Plant Disease Detection",
    page_icon="🥔",
    layout="wide"
)

MODEL_PATH = "explainable_cnn_model.keras"
IMG_SIZE = (224, 224)

HISTORY_FILE = "prediction_history.csv"
LOW_CONFIDENCE_THRESHOLD = 0.60
SEVERITY_MILD_THRESHOLD = 0.15
SEVERITY_MODERATE_THRESHOLD = 0.35

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
# DISEASE KNOWLEDGE
# ============================================================

DISEASE_INFO = {
    "Potato___Early_blight": {
        "name": "Early Blight",
        "description": "A common potato leaf disease that can reduce plant growth and yield.",
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
        "description": "A potentially fast-spreading potato disease, especially under cool and humid conditions.",
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
        "description": "The model classified the uploaded potato leaf as healthy.",
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
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        loaded_model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        return loaded_model, None
    except Exception as exc:
        return None, str(exc)


model, model_error = load_model()

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "image": None,
    "prediction": None,
    "probabilities": None,
    "confidence": None,
    "filename": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# HELPERS
# ============================================================

def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(image):
    processed = preprocess_image(image)
    output = model.predict(processed, verbose=0)
    output = np.asarray(output)

    if output.ndim == 2:
        probabilities = output[0]
    else:
        probabilities = output.reshape(-1)

    # If the model returns logits rather than probabilities,
    # convert them to probabilities safely.
    if (
        np.min(probabilities) < 0
        or np.max(probabilities) > 1.0
        or not np.isclose(np.sum(probabilities), 1.0, atol=1e-3)
    ):
        exp_values = np.exp(
            probabilities - np.max(probabilities)
        )
        probabilities = exp_values / np.sum(exp_values)

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probabilities)} outputs, "
            f"but this app expects {len(CLASS_NAMES)} classes."
        )

    index = int(np.argmax(probabilities))
    class_name = CLASS_NAMES[index]
    confidence = float(probabilities[index])

    return class_name, confidence, probabilities


def find_conv_layer(layer):
    # Direct Conv2D layer
    if isinstance(layer, tf.keras.layers.Conv2D):
        return layer

    # Search inside nested Functional/Sequential models
    if hasattr(layer, "layers"):
        for child in reversed(layer.layers):
            found = find_conv_layer(child)
            if found is not None:
                return found

    return None


def get_gradcam_model():
    conv_layer = find_conv_layer(model)

    if conv_layer is None:
        raise ValueError(
            "No Conv2D layer was found in the loaded model. "
            "Grad-CAM requires a convolutional layer."
        )

    # Standard case: convolutional layer belongs to the loaded model.
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )
        return grad_model, conv_layer.name
    except Exception:
        raise ValueError(
            "The model's convolutional layer could not be connected "
            "to the model output for Grad-CAM."
        )


def make_gradcam(image, class_index):
    grad_model, layer_name = get_gradcam_model()
    input_tensor = preprocess_image(image)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_tensor)
        score = predictions[:, class_index]

    gradients = tape.gradient(score, conv_output)

    if gradients is None:
        raise ValueError("Gradients were not produced for Grad-CAM.")

    # Works for standard 4-D convolutional feature maps.
    if len(conv_output.shape) != 4:
        raise ValueError(
            f"Grad-CAM feature map has unsupported shape: {conv_output.shape}"
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

    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)

    if float(maximum) > 0:
        heatmap = heatmap / maximum

    return heatmap.numpy(), layer_name


def make_overlay(image, heatmap):
    original = image.convert("RGB").resize(IMG_SIZE)

    heat = Image.fromarray(
        np.uint8(np.clip(heatmap * 255, 0, 255))
    ).resize(IMG_SIZE)

    heat_array = np.asarray(heat, dtype=np.float32) / 255.0

    # Simple dependency-free RGB heatmap.
    red = heat_array
    green = np.sqrt(heat_array)
    blue = 1.0 - heat_array

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.uint8(np.clip(rgb * 255, 0, 255))

    heat_rgb = Image.fromarray(rgb)
    return Image.blend(original, heat_rgb, 0.45)


def calculate_metrics(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    for actual, predicted in zip(y_true, y_pred):
        if 0 <= actual < n_classes and 0 <= predicted < n_classes:
            matrix[actual, predicted] += 1

    accuracy = float(np.mean(y_true == y_pred))

    precisions = []
    recalls = []
    f1s = []

    for i in range(n_classes):
        tp = matrix[i, i]
        fp = int(np.sum(matrix[:, i]) - tp)
        fn = int(np.sum(matrix[i, :]) - tp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return (
        accuracy,
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
        matrix
    )


def dataframe_rows_from_matrix(matrix, names):
    rows = []
    for i, actual_name in enumerate(names):
        row = {"Actual": actual_name}
        for j, predicted_name in enumerate(names):
            row[predicted_name] = int(matrix[i, j])
        rows.append(row)
    return rows


def read_history():
    path = "training_history.json"

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None



# ============================================================
# ADDITIONAL FEATURES
# ============================================================

def get_top_predictions(probabilities, top_n=3):
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    order = np.argsort(probabilities)[::-1][:min(top_n, len(probabilities))]
    return [
        (CLASS_NAMES[int(i)], float(probabilities[int(i)]))
        for i in order
    ]


def estimate_attention_severity(heatmap):
    """
    This is an approximate Grad-CAM attention estimate, NOT a
    medically/agronomically validated percentage of infected tissue.
    It measures the fraction of highly activated Grad-CAM pixels.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.size == 0:
        return 0.0, "Unknown"

    normalized = np.clip(heatmap, 0.0, 1.0)
    active_ratio = float(np.mean(normalized >= 0.55))

    if active_ratio < SEVERITY_MILD_THRESHOLD:
        level = "Mild"
    elif active_ratio < SEVERITY_MODERATE_THRESHOLD:
        level = "Moderate"
    else:
        level = "Severe"

    return active_ratio, level


def append_prediction_history(filename, prediction, confidence, probabilities):
    file_exists = os.path.exists(HISTORY_FILE)

    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image": filename or "Unknown",
        "Prediction": DISPLAY_NAMES.get(prediction, prediction),
        "Confidence": round(float(confidence) * 100, 2),
        "Early Blight": round(float(probabilities[0]) * 100, 2),
        "Late Blight": round(float(probabilities[1]) * 100, 2),
        "Healthy": round(float(probabilities[2]) * 100, 2),
    }

    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_prediction_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(HISTORY_FILE)
        return df
    except Exception:
        return pd.DataFrame()


def make_result_dataframe():
    probabilities = st.session_state.probabilities
    prediction = st.session_state.prediction
    confidence = st.session_state.confidence
    filename = st.session_state.filename

    if probabilities is None or prediction is None:
        return pd.DataFrame()

    return pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image": filename or "Unknown",
        "Prediction": DISPLAY_NAMES.get(prediction, prediction),
        "Confidence (%)": round(float(confidence) * 100, 2),
        "Early Blight (%)": round(float(probabilities[0]) * 100, 2),
        "Late Blight (%)": round(float(probabilities[1]) * 100, 2),
        "Healthy (%)": round(float(probabilities[2]) * 100, 2),
    }])


def create_pdf_report():
    if st.session_state.image is None:
        raise ValueError("No image is currently loaded.")

    image = st.session_state.image
    prediction = st.session_state.prediction
    confidence = float(st.session_state.confidence)
    probabilities = np.asarray(st.session_state.probabilities)
    info = DISEASE_INFO[prediction]

    heatmap, layer_name = make_gradcam(
        image,
        int(np.argmax(probabilities))
    )
    overlay = make_overlay(image, heatmap)

    severity_ratio, severity_level = estimate_attention_severity(heatmap)

    pdf_buffer = BytesIO()
    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    story = []

    story.append(Paragraph(
        "Potato Plant Disease Detection Report",
        title_style
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        normal_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        f"<b>Prediction:</b> {DISPLAY_NAMES[prediction]}",
        heading_style
    ))
    story.append(Paragraph(
        f"<b>Confidence:</b> {confidence * 100:.2f}%",
        normal_style
    ))
    story.append(Paragraph(
        f"<b>Grad-CAM layer:</b> {layer_name}",
        normal_style
    ))
    story.append(Paragraph(
        f"<b>Attention severity estimate:</b> {severity_level} "
        f"({severity_ratio * 100:.2f}% highly activated area)",
        normal_style
    ))
    story.append(Spacer(1, 12))

    original_buffer = BytesIO()
    image.save(original_buffer, format="PNG")
    original_buffer.seek(0)

    overlay_buffer = BytesIO()
    overlay.save(overlay_buffer, format="PNG")
    overlay_buffer.seek(0)

    image_table = Table([
        [
            ReportLabImage(original_buffer, width=3.1*inch, height=3.1*inch),
            ReportLabImage(overlay_buffer, width=3.1*inch, height=3.1*inch)
        ]
    ], colWidths=[3.2*inch, 3.2*inch])

    image_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER")
    ]))

    story.append(image_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Class Probability Distribution",
        heading_style
    ))

    probability_rows = [["Class", "Probability"]]
    for class_name, value in get_top_predictions(probabilities, 3):
        probability_rows.append([
            DISPLAY_NAMES[class_name],
            f"{value * 100:.2f}%"
        ])

    probability_table = Table(
        probability_rows,
        colWidths=[4.0*inch, 2.0*inch]
    )
    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (1, -1), "RIGHT")
    ]))
    story.append(probability_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Description", heading_style))
    story.append(Paragraph(info["description"], normal_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Symptoms", heading_style))
    for item in info["symptoms"]:
        story.append(Paragraph(f"• {item}", normal_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Possible Causes", heading_style))
    for item in info["causes"]:
        story.append(Paragraph(f"• {item}", normal_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Recommended Actions", heading_style))
    for item in info["actions"]:
        story.append(Paragraph(f"• {item}", normal_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Prevention", heading_style))
    for item in info["prevention"]:
        story.append(Paragraph(f"• {item}", normal_style))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Disclaimer: This report provides AI-based preliminary "
        "classification and general information. The Grad-CAM "
        "attention severity is an experimental visualization metric, "
        "not a scientifically validated disease-severity percentage. "
        "For serious or rapidly spreading crop disease, consult a "
        "qualified agricultural professional.",
        normal_style
    ))

    document.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def get_test_directory():
    test_candidates = [
        "test",
        "Test",
        "dataset/test",
        "data/test"
    ]

    for candidate in test_candidates:
        if os.path.isdir(candidate):
            return candidate

    return None


def load_test_predictions():
    test_dir = get_test_directory()

    if test_dir is None:
        raise FileNotFoundError(
            "Test dataset folder not found. Expected one of: "
            "test, Test, dataset/test, data/test."
        )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=False,
        class_names=CLASS_NAMES
    )

    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in test_ds:
        images = tf.cast(images, tf.float32) / 255.0
        outputs = model.predict(images, verbose=0)
        outputs = np.asarray(outputs)

        if outputs.ndim != 2 or outputs.shape[1] != len(CLASS_NAMES):
            raise ValueError(
                f"Unexpected model output shape: {outputs.shape}"
            )

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(outputs, axis=1).tolist())
        y_prob.extend(outputs.tolist())

    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(y_prob, dtype=np.float32)
    )


# ============================================================
# MODEL ERROR CHECK
# ============================================================

if model is None:
    st.error("❌ The trained model could not be loaded.")
    st.code(model_error or "Unknown model loading error.")
    st.info(
        "Put explainable_cnn_model.keras in the same GitHub "
        "folder as app.py and make sure TensorFlow is in requirements.txt."
    )
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🥔 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Home",
        "🔍 Disease Detection",
        "🔥 Explainable AI",
        "📚 Disease Information",
        "🩺 Causes & Solutions",
        "📦 Batch Analysis",
        "📊 Performance",
        "📈 ROC-AUC",
        "📜 Prediction History",
        "📄 PDF Report",
        "⚖️ Model Comparison"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Custom CNN • TensorFlow/Keras • Grad-CAM")
st.sidebar.caption("PDF • ROC-AUC • History • Batch Analysis")

# ============================================================
# HOME - ONLY MAIN IMAGE UPLOAD
# ============================================================

if page == "🏠 Home":

    st.title("🥔 Potato Plant Disease Detection System")

    st.markdown(
        """
        ### AI-powered potato leaf analysis

        Upload **one image here only**. After analysis, the same
        image and prediction are automatically available in:

        **Disease Detection → Explainable AI → Disease Information
        → Causes & Solutions**
        """
    )

    selected_file = st.file_uploader(
        "📷 Upload a potato leaf image",
        type=["jpg", "jpeg", "png"],
        key="main_uploader"
    )

    camera_file = st.camera_input(
        "📷 Or take a potato leaf photo with your camera",
        key="main_camera"
    )

    selected_file = selected_file if selected_file is not None else camera_file

    if selected_file is not None:
        try:
            image = Image.open(selected_file).convert("RGB")

            with st.spinner("Analyzing the potato leaf..."):
                predicted_class, confidence, probabilities = predict_image(image)

            st.session_state.image = image
            st.session_state.prediction = predicted_class
            st.session_state.confidence = confidence
            st.session_state.probabilities = probabilities
            st.session_state.filename = selected_file.name

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image,
                    caption=selected_file.name,
                    use_container_width=True
                )

            with col2:
                st.subheader("🎯 Detection Result")
                st.success(DISPLAY_NAMES[predicted_class])
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                if predicted_class == "Potato___healthy":
                    st.success("✅ No strong disease pattern was detected.")
                else:
                    st.warning("⚠️ A disease pattern was detected.")

            st.divider()

            st.subheader("📊 All Class Probabilities")

            for i, class_name in enumerate(CLASS_NAMES):
                value = float(probabilities[i])
                st.write(DISPLAY_NAMES[class_name])
                st.progress(min(max(value, 0.0), 1.0))
                st.caption(f"{value * 100:.2f}%")

            st.success(
                "✓ Done. You do not need to upload this image again "
                "for the other single-image sections."
            )

        except Exception as exc:
            st.error("❌ Image analysis failed.")
            st.code(str(exc))

    elif st.session_state.image is not None:
        st.info(
            "An image is already loaded. Use the sidebar to view "
            "the analysis, or upload another image above."
        )

        st.image(
            st.session_state.image,
            caption=st.session_state.filename or "Loaded image",
            width=500
        )

    else:
        st.info("👆 Upload a potato leaf image to begin.")


# ============================================================
# SINGLE IMAGE PAGE CHECK
# ============================================================

elif page in [
    "🔍 Disease Detection",
    "🔥 Explainable AI",
    "📚 Disease Information",
    "🩺 Causes & Solutions"
]:

    if st.session_state.image is None:
        st.warning(
            "⚠️ No image is loaded. Go to **Home** and upload "
            "one potato leaf image first."
        )
        st.stop()


# ============================================================
# DISEASE DETECTION
# ============================================================

if page == "🔍 Disease Detection":

    st.title("🔍 Disease Detection")

    image = st.session_state.image
    predicted_class = st.session_state.prediction
    confidence = st.session_state.confidence
    probabilities = st.session_state.probabilities

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Analyzed Potato Leaf",
            use_container_width=True
        )

    with col2:
        st.subheader("Prediction")
        st.success(DISPLAY_NAMES[predicted_class])
        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                "⚠️ Confidence is relatively low. Consider "
                "checking the leaf under good lighting or "
                "consulting an agricultural expert."
            )

    st.divider()

    st.subheader("📊 Class Probability Distribution")

    for i, class_name in enumerate(CLASS_NAMES):
        value = float(probabilities[i])

        col_a, col_b = st.columns([4, 1])

        with col_a:
            st.write(DISPLAY_NAMES[class_name])
            st.progress(min(max(value, 0.0), 1.0))

        with col_b:
            st.write(f"{value * 100:.2f}%")


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🔥 Explainable AI":

    st.title("🔥 Explainable AI - Grad-CAM")

    image = st.session_state.image
    predicted_class = st.session_state.prediction
    probabilities = st.session_state.probabilities
    predicted_index = int(np.argmax(probabilities))

    st.write(
        "Grad-CAM highlights image regions that contributed "
        "to the selected CNN prediction."
    )

    try:
        with st.spinner("Generating Grad-CAM..."):
            heatmap, layer_name = make_gradcam(
                image,
                predicted_index
            )
            overlay = make_overlay(
                image,
                heatmap
            )

        st.success("✓ Grad-CAM generated successfully.")

        severity_ratio, severity_level = estimate_attention_severity(heatmap)

        st.subheader("🗺️ Experimental Attention Severity")
        st.metric("Attention Level", severity_level)
        st.write(
            f"Highly activated Grad-CAM area: "
            f"**{severity_ratio * 100:.2f}%**"
        )
        st.caption(
            "This is an experimental Grad-CAM attention estimate, "
            "not a validated percentage of infected leaf tissue."
        )

        st.write(
            f"**Prediction:** {DISPLAY_NAMES[predicted_class]}"
        )
        st.write(f"**Convolutional layer:** `{layer_name}`")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                image,
                use_container_width=True
            )

        with col2:
            st.subheader("Grad-CAM")
            st.image(
                overlay,
                use_container_width=True
            )

        st.divider()

        st.subheader("🧠 What the visualization means")

        st.write(
            """
            The brighter highlighted regions indicate areas that
            contributed more strongly to the model's prediction.
            This helps you inspect whether the CNN is focusing on
            relevant leaf regions rather than unrelated background.
            """
        )

    except Exception as exc:
        st.error("❌ Grad-CAM could not be generated.")
        st.code(str(exc))

        st.info(
            "The prediction model can still work even when Grad-CAM "
            "is unavailable. This usually means the saved model's "
            "internal convolutional graph is not compatible with "
            "the standard Grad-CAM connection."
        )


# ============================================================
# DISEASE INFORMATION
# ============================================================

elif page == "📚 Disease Information":

    st.title("📚 Disease Information")

    predicted_class = st.session_state.prediction
    confidence = st.session_state.confidence
    info = DISEASE_INFO[predicted_class]

    st.subheader(f"🌿 {info['name']}")

    st.write(info["description"])

    st.metric(
        "Model Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.divider()

    st.subheader("🔎 Symptoms")

    for item in info["symptoms"]:
        st.write(f"• {item}")

    st.divider()

    st.subheader("🦠 Possible Causes")

    for item in info["causes"]:
        st.write(f"• {item}")


# ============================================================
# CAUSES & SOLUTIONS
# ============================================================

elif page == "🩺 Causes & Solutions":

    st.title("🩺 Why Did This Problem Occur?")

    predicted_class = st.session_state.prediction
    info = DISEASE_INFO[predicted_class]

    st.subheader(
        f"Detected condition: {info['name']}"
    )

    st.divider()

    st.subheader("❓ Possible Reasons")

    for item in info["causes"]:
        st.write(f"• {item}")

    st.divider()

    st.subheader("⚡ What Can Be Done First?")

    for item in info["actions"]:
        st.write(f"✅ {item}")

    st.divider()

    st.subheader("🛡️ Prevention")

    for item in info["prevention"]:
        st.write(f"• {item}")

    st.warning(
        """
        ⚠️ The application provides AI-based preliminary
        classification and general management information.
        For serious or rapidly spreading crop disease,
        consult a qualified agricultural professional.
        """
    )


# ============================================================
# BATCH ANALYSIS
# ============================================================

elif page == "📦 Batch Analysis":

    st.title("📦 Batch Analysis")

    st.write(
        "This section is intentionally separate because batch "
        "analysis accepts multiple images at once."
    )

    files = st.file_uploader(
        "Upload multiple potato leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if files:
        rows = []
        progress = st.progress(0)

        for index, file in enumerate(files):
            try:
                image = Image.open(file).convert("RGB")
                predicted_class, confidence, _ = predict_image(image)

                rows.append({
                    "Image": file.name,
                    "Prediction": DISPLAY_NAMES[predicted_class],
                    "Confidence": f"{confidence * 100:.2f}%"
                })

            except Exception as exc:
                rows.append({
                    "Image": file.name,
                    "Prediction": "Error",
                    "Confidence": str(exc)
                })

            progress.progress(
                int(((index + 1) / len(files)) * 100)
            )

        st.success(f"✓ {len(files)} images analyzed.")
        st.dataframe(rows, use_container_width=True)

        batch_df = pd.DataFrame(rows)
        st.download_button(
            "⬇️ Download Batch Analysis CSV",
            data=batch_df.to_csv(index=False).encode("utf-8"),
            file_name="potato_batch_analysis.csv",
            mime="text/csv"
        )

        healthy = sum(
            1 for row in rows
            if row["Prediction"] == "Healthy"
        )

        disease = sum(
            1 for row in rows
            if row["Prediction"] in ["Early Blight", "Late Blight"]
        )

        c1, c2 = st.columns(2)

        with c1:
            st.success(f"Healthy: {healthy}")

        with c2:
            st.warning(f"Disease detected: {disease}")

    else:
        st.info("Upload multiple images to start batch analysis.")


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "📊 Performance":

    st.title("📊 Model Performance")

    st.write(
        """
        This section shows real evaluation values only when the
        test dataset and/or training history are available in
        the GitHub repository. The app does not invent metrics.
        """
    )

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    test_dir = get_test_directory()

    if test_dir is None:

        st.warning(
            """
            ⚠️ A test dataset folder was not found.

            Therefore Accuracy, Precision, Recall, F1 Score,
            Confusion Matrix and ROC-AUC cannot be calculated
            automatically.

            Add a folder such as:

            test/
            ├── Potato___Early_blight/
            ├── Potato___Late_blight/
            └── Potato___healthy/
            """
        )

    else:

        try:
            start = time.time()

            y_true, y_pred, y_prob = load_test_predictions()

            elapsed = time.time() - start

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

            st.subheader("🎯 Evaluation Components")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Accuracy", f"{accuracy * 100:.2f}%")

            with c2:
                st.metric("Precision", f"{precision * 100:.2f}%")

            with c3:
                st.metric("Recall", f"{recall * 100:.2f}%")

            with c4:
                st.metric("F1 Score", f"{f1 * 100:.2f}%")

            st.divider()

            st.subheader("🔲 Confusion Matrix")

            matrix_df = pd.DataFrame(
                matrix,
                index=[DISPLAY_NAMES[x] for x in CLASS_NAMES],
                columns=[DISPLAY_NAMES[x] for x in CLASS_NAMES]
            )

            st.dataframe(matrix_df, use_container_width=True)

            st.subheader("📊 Confusion Matrix Heatmap")

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.imshow(matrix, interpolation="nearest")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels([DISPLAY_NAMES[x] for x in CLASS_NAMES],
                               rotation=25, ha="right")
            ax.set_yticklabels([DISPLAY_NAMES[x] for x in CLASS_NAMES])

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, int(matrix[i, j]),
                            ha="center", va="center")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.divider()

            correct = int(np.sum(y_true == y_pred))
            incorrect = int(len(y_true) - correct)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Test Images", len(y_true))

            with c2:
                st.metric("Correct", correct)

            with c3:
                st.metric("Incorrect", incorrect)

            st.caption(f"Evaluation time: {elapsed:.2f} seconds")

        except Exception as exc:
            st.error("❌ Test-set evaluation failed.")
            st.code(str(exc))

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    st.divider()
    st.subheader("📈 Training / Validation Performance")

    history = read_history()

    if history is None:

        st.info(
            """
            `training_history.json` was not found.

            The `.keras` model does not reliably contain the
            complete epoch-by-epoch training history.

            If you want Training Accuracy, Validation Accuracy,
            Training Loss and Validation Loss here, export the
            Kaggle `history.history` dictionary as
            `training_history.json` and put it beside app.py.
            """
        )

    else:

        try:
            acc_key = (
                "accuracy"
                if "accuracy" in history
                else "acc"
                if "acc" in history
                else None
            )

            val_acc_key = (
                "val_accuracy"
                if "val_accuracy" in history
                else "val_acc"
                if "val_acc" in history
                else None
            )

            loss_key = "loss" if "loss" in history else None
            val_loss_key = (
                "val_loss"
                if "val_loss" in history
                else None
            )

            if acc_key and val_acc_key:
                chart_rows = []

                for i in range(len(history[acc_key])):
                    chart_rows.append({
                        "Epoch": i + 1,
                        "Training Accuracy": history[acc_key][i],
                        "Validation Accuracy": history[val_acc_key][i]
                    })

                st.write("Training vs Validation Accuracy")
                st.line_chart(
                    chart_rows,
                    x="Epoch",
                    y=[
                        "Training Accuracy",
                        "Validation Accuracy"
                    ]
                )

            if loss_key and val_loss_key:
                loss_rows = []

                for i in range(len(history[loss_key])):
                    loss_rows.append({
                        "Epoch": i + 1,
                        "Training Loss": history[loss_key][i],
                        "Validation Loss": history[val_loss_key][i]
                    })

                st.write("Training vs Validation Loss")
                st.line_chart(
                    loss_rows,
                    x="Epoch",
                    y=[
                        "Training Loss",
                        "Validation Loss"
                    ]
                )

        except Exception as exc:
            st.error("❌ Training history could not be displayed.")
            st.code(str(exc))



# ============================================================
# ROC-AUC
# ============================================================

elif page == "📈 ROC-AUC":

    st.title("📈 ROC Curve & AUC")
    st.write(
        "One-vs-rest ROC curves are calculated from the actual "
        "test-set predictions when the test dataset is available."
    )

    try:
        y_true, y_pred, y_prob = load_test_predictions()

        fig, ax = plt.subplots(figsize=(8, 6))

        auc_rows = []

        for class_index, class_name in enumerate(CLASS_NAMES):
            binary_true = (y_true == class_index).astype(int)

            if len(np.unique(binary_true)) < 2:
                st.warning(
                    f"ROC-AUC for {DISPLAY_NAMES[class_name]} "
                    "cannot be calculated because the test set "
                    "does not contain both positive and negative samples."
                )
                continue

            fpr, tpr, _ = roc_curve(
                binary_true,
                y_prob[:, class_index]
            )
            class_auc = auc(fpr, tpr)

            ax.plot(
                fpr,
                tpr,
                label=f"{DISPLAY_NAMES[class_name]} "
                      f"(AUC = {class_auc:.4f})"
            )

            auc_rows.append({
                "Class": DISPLAY_NAMES[class_name],
                "AUC": round(float(class_auc), 4)
            })

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("One-vs-Rest ROC Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        if auc_rows:
            st.subheader("AUC Scores")
            st.dataframe(
                pd.DataFrame(auc_rows),
                use_container_width=True
            )

    except Exception as exc:
        st.error("❌ ROC-AUC calculation failed.")
        st.code(str(exc))
        st.info(
            "Make sure the test folder exists and contains all "
            "three class folders."
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    history_df = read_prediction_history()

    if history_df.empty:
        st.info(
            "No saved predictions yet. Go to Disease Detection "
            "and click 'Save This Prediction to History'."
        )
    else:
        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear Prediction History"):
            try:
                os.remove(HISTORY_FILE)
                st.success("✓ Prediction history cleared.")
                st.rerun()
            except Exception as exc:
                st.error("❌ Could not clear history.")
                st.code(str(exc))


# ============================================================
# PDF REPORT
# ============================================================

elif page == "📄 PDF Report":

    st.title("📄 Generate Professional PDF Report")

    if st.session_state.image is None:
        st.warning(
            "⚠️ First upload and analyze a potato leaf image "
            "from the Home page."
        )
    else:
        image = st.session_state.image
        prediction = st.session_state.prediction
        confidence = st.session_state.confidence

        st.image(
            image,
            caption=st.session_state.filename or "Analyzed image",
            width=450
        )

        st.subheader("Report Summary")
        st.write(
            f"**Prediction:** {DISPLAY_NAMES[prediction]}"
        )
        st.write(
            f"**Confidence:** {confidence * 100:.2f}%"
        )

        if st.button("📄 Generate PDF Report"):
            try:
                with st.spinner("Creating PDF report..."):
                    pdf_bytes = create_pdf_report()

                st.success("✓ PDF report generated successfully.")

                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name="potato_disease_report.pdf",
                    mime="application/pdf"
                )

            except Exception as exc:
                st.error("❌ PDF report generation failed.")
                st.code(str(exc))
                st.info(
                    "Make sure reportlab is installed from "
                    "requirements.txt."
                )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "⚖️ Model Comparison":

    st.title("⚖️ Model Comparison")

    st.write(
        """
        Comparison of the models used in the project.
        Only verified values should be entered for Accuracy,
        Loss and Parameters.
        """
    )

    st.subheader("Architecture Comparison")

    comparison_rows = [
        {
            "Model": "Custom CNN",
            "Type": "Proposed Lightweight CNN",
            "Input": "224 × 224 × 3",
            "Classes": "3",
            "Main Goal": "Lightweight disease classification"
        },
        {
            "Model": "ResNet50",
            "Type": "Transfer Learning",
            "Input": "224 × 224 × 3",
            "Classes": "3",
            "Main Goal": "Baseline comparison"
        },
        {
            "Model": "MobileNetV2",
            "Type": "Transfer Learning",
            "Input": "224 × 224 × 3",
            "Classes": "3",
            "Main Goal": "Lightweight transfer-learning comparison"
        }
    ]

    st.dataframe(
        comparison_rows,
        use_container_width=True
    )

    st.divider()

    st.subheader("📌 Experimental Results")

    result_file = "model_comparison.json"

    if os.path.exists(result_file):

        try:
            with open(result_file, "r", encoding="utf-8") as file:
                comparison_data = json.load(file)

            st.dataframe(
                comparison_data,
                use_container_width=True
            )

        except Exception as exc:
            st.error("Could not read model_comparison.json.")
            st.code(str(exc))

    else:

        st.info(
            """
            `model_comparison.json` was not found.

            I am intentionally not putting guessed Accuracy,
            Test Loss or Parameter values here. Add the actual
            values from your Kaggle experiments to display them.
            """
        )

# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption("Potato Plant Disease Detection")
st.sidebar.caption("Custom CNN • Explainable AI • Grad-CAM")
