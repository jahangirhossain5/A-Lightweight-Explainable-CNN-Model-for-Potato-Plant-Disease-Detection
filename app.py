import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CONSTANTS
# ============================================================

MODEL_PATH = "explainable_cnn_model.keras"

IMG_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.70

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
        "description": (
            "Early blight is a common potato leaf disease "
            "that can cause dark lesions on leaves."
        ),
        "symptoms": [
            "Dark or brown lesions on leaves",
            "Lesions may develop concentric patterns",
            "Older leaves can be affected first",
            "Severe infection may cause leaf yellowing and drying"
        ],
        "recommendations": [
            "Remove severely affected plant material where appropriate",
            "Maintain good field sanitation",
            "Avoid prolonged leaf wetness",
            "Use locally recommended disease-management practices"
        ]
    },

    "Potato___Late_blight": {
        "name": "Late Blight",
        "description": (
            "Late blight is a serious potato disease that can "
            "spread rapidly under favorable environmental conditions."
        ),
        "symptoms": [
            "Dark irregular lesions",
            "Brown or black affected areas",
            "Rapid progression may occur under favorable conditions",
            "Leaves may become damaged and die"
        ],
        "recommendations": [
            "Remove severely affected plant material where appropriate",
            "Improve field ventilation and reduce prolonged moisture",
            "Monitor nearby plants carefully",
            "Follow locally recommended disease-management practices"
        ]
    },

    "Potato___healthy": {
        "name": "Healthy Potato Leaf",
        "description": (
            "The model classified the uploaded image as a healthy "
            "potato leaf."
        ),
        "symptoms": [
            "No major disease pattern detected by the model",
            "Leaf appears comparatively healthy"
        ],
        "recommendations": [
            "Continue regular crop monitoring",
            "Maintain appropriate irrigation and nutrition",
            "Monitor plants regularly for new symptoms"
        ]
    }
}


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# 5. FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(model):

    # Search from the end of the model
    for layer in reversed(model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):
            return layer

    return None


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image_resized = image.resize(
        IMG_SIZE
    )

    img_array = np.array(
        image_resized
    ).astype("float32")

    # Same normalization used during training
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def predict_image(model, image):

    processed_image = preprocess_image(
        image
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        probabilities,
        processed_image
    )


# ============================================================
# 8. IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image):

    image_rgb = image.convert("RGB")

    img_array = np.array(
        image_rgb
    ).astype("float32")

    brightness = float(
        np.mean(img_array)
    )

    contrast = float(
        np.std(img_array)
    )

    issues = []

    # Very dark
    if brightness < 25:
        issues.append(
            "Image is very dark."
        )

    # Very bright
    if brightness > 240:
        issues.append(
            "Image is very bright."
        )

    # Very low contrast
    if contrast < 15:
        issues.append(
            "Image has very low contrast."
        )

    if len(issues) == 0:

        return True, (
            "Image quality looks acceptable "
            "for analysis."
        )

    return False, " ".join(issues)


# ============================================================
# 9. GRAD-CAM
# ============================================================

def make_gradcam(
    model,
    image_array,
    predicted_index
):

    last_conv_layer = find_last_conv_layer(
        model
    )

    if last_conv_layer is None:

        return None, None

    # Create model that returns:
    # convolutional feature maps + predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(image_array)
        )

        class_channel = predictions[
            :, predicted_index
        ]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    if grads is None:

        return None, None

    # Average gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_grads = pooled_grads[0]

    # Weight feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    # Resize heatmap to original image size
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        (224, 224)
    )

    heatmap_array = np.array(
        heatmap_image
    ) / 255.0

    return heatmap_array, last_conv_layer.name


# ============================================================
# 10. CREATE GRAD-CAM FIGURE
# ============================================================

def create_gradcam_figure(
    original_image,
    heatmap
):

    original = original_image.convert(
        "RGB"
    ).resize(
        IMG_SIZE
    )

    original_array = np.array(
        original
    )

    fig1, ax1 = plt.subplots(
        figsize=(6, 5)
    )

    ax1.imshow(
        heatmap,
        cmap="jet"
    )

    ax1.axis("off")

    plt.tight_layout()

    # Convert heatmap to RGB
    heatmap_uint8 = np.uint8(
        heatmap * 255
    )

    heatmap_img = Image.fromarray(
        heatmap_uint8
    ).resize(
        IMG_SIZE
    )

    heatmap_array = np.array(
        heatmap_img
    )

    # Create RGB heatmap using matplotlib
    cmap = plt.get_cmap("jet")

    colored_heatmap = cmap(
        heatmap_array / 255.0
    )[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    # Blend original + heatmap
    overlay = (
        0.55 * original_array
        + 0.45 * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype("uint8")

    overlay_image = Image.fromarray(
        overlay
    )

    return overlay_image


# ============================================================
# 11. REPORT GENERATOR
# ============================================================

def generate_report(
    filename,
    predicted_class,
    confidence,
    probabilities
):

    disease = DISEASE_INFO[
        predicted_class
    ]

    date_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    probability_html = ""

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            float(probabilities[i])
            * 100
        )

        probability_html += f"""
        <tr>
            <td>{DISPLAY_NAMES[class_name]}</td>
            <td>{probability:.2f}%</td>
        </tr>
        """

    symptoms_html = ""

    for item in disease["symptoms"]:

        symptoms_html += (
            f"<li>{item}</li>"
        )

    recommendations_html = ""

    for item in disease["recommendations"]:

        recommendations_html += (
            f"<li>{item}</li>"
        )

    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

    <meta charset="UTF-8">

    <title>Potato Disease Report</title>

    <style>

    body {{
        font-family: Arial, sans-serif;
        margin: 40px;
        line-height: 1.6;
    }}

    h1 {{
        text-align: center;
    }}

    .result {{
        padding: 20px;
        border: 1px solid #ccc;
        margin-top: 20px;
    }}

    table {{
        border-collapse: collapse;
        width: 100%;
    }}

    th, td {{
        border: 1px solid #ccc;
        padding: 10px;
        text-align: left;
    }}

    </style>

    </head>

    <body>

    <h1>Potato Plant Disease Detection Report</h1>

    <p>
    <strong>Date:</strong> {date_time}
    </p>

    <p>
    <strong>Image:</strong> {filename}
    </p>

    <div class="result">

    <h2>
    Prediction:
    {disease["name"]}
    </h2>

    <h3>
    Confidence:
    {confidence * 100:.2f}%
    </h3>

    </div>

    <h2>Disease Information</h2>

    <p>
    {disease["description"]}
    </p>

    <h2>Symptoms</h2>

    <ul>
    {symptoms_html}
    </ul>

    <h2>Recommended Actions</h2>

    <ul>
    {recommendations_html}
    </ul>

    <h2>Class Probabilities</h2>

    <table>

    <tr>
        <th>Class</th>
        <th>Probability</th>
    </tr>

    {probability_html}

    </table>

    <br>

    <p>
    <strong>Note:</strong>
    This result is generated by a machine-learning
    model and should be treated as decision-support
    information rather than a definitive agricultural diagnosis.
    </p>

    </body>

    </html>
    """

    return html


# ============================================================
# 12. SIDEBAR
# ============================================================

st.sidebar.title(
    "Potato Disease AI"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Disease Detection",
        "Batch Analysis",
        "Explainable AI",
        "Disease Information",
        "Model Comparison",
        "Performance Dashboard"
    ]
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "Model could not be loaded."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.title(
        "Potato Plant Disease Detection System"
    )

    st.write(
        """
        An AI-based potato leaf disease detection
        and explainability system using a lightweight
        Custom CNN model.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Input Size",
            "224 × 224"
        )

    with col2:

        st.metric(
            "Disease Classes",
            "3"
        )

    with col3:

        st.metric(
            "Model",
            "Custom CNN"
        )

    st.divider()

    st.subheader(
        "System Features"
    )

    features = [
        "Single image disease detection",
        "Multiple image batch analysis",
        "Confidence score",
        "Class probability analysis",
        "Image quality checking",
        "Grad-CAM Explainable AI",
        "Disease information",
        "Model comparison",
        "Performance dashboard",
        "Downloadable detection report"
    ]

    for feature in features:

        st.write(
            f"✓ {feature}"
        )


# ============================================================
# DISEASE DETECTION PAGE
# ============================================================

elif page == "Disease Detection":

    st.title(
        "Disease Detection"
    )

    uploaded_file = st.file_uploader(
        "Upload a potato leaf image",
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

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Uploaded Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader(
                "Image Quality"
            )

            quality_ok, quality_message = (
                check_image_quality(image)
            )

            if quality_ok:

                st.success(
                    quality_message
                )

            else:

                st.warning(
                    quality_message
                )

        st.divider()

        if st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing..."
            ):

                (
                    predicted_class,
                    confidence,
                    probabilities,
                    processed_image
                ) = predict_image(
                    model,
                    image
                )

            confidence_percent = (
                confidence * 100
            )

            st.subheader(
                "Prediction Result"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.success(
                    f"Prediction: "
                    f"{DISPLAY_NAMES[predicted_class]}"
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence_percent:.2f}%"
                )

            # Confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:

                st.warning(
                    "Low-confidence prediction. "
                    "Please upload a clear potato leaf "
                    "image for a more reliable result."
                )

            else:

                st.info(
                    "The model produced a prediction "
                    "above the confidence threshold."
                )

            # Health status
            if predicted_class == (
                "Potato___healthy"
            ):

                st.success(
                    "The potato leaf appears healthy."
                )

            else:

                st.warning(
                    f"Disease detected: "
                    f"{DISPLAY_NAMES[predicted_class]}"
                )

            # Probability chart
            st.subheader(
                "Class Probability"
            )

            chart_data = {}

            for i, class_name in enumerate(
                CLASS_NAMES
            ):

                chart_data[
                    DISPLAY_NAMES[class_name]
                ] = float(
                    probabilities[i]
                ) * 100

            st.bar_chart(
                chart_data
            )

            # Disease information
            disease = DISEASE_INFO[
                predicted_class
            ]

            st.subheader(
                "Disease Information"
            )

            st.write(
                disease["description"]
            )

            st.write(
                "**Symptoms**"
            )

            for symptom in disease["symptoms"]:

                st.write(
                    f"• {symptom}"
                )

            st.write(
                "**Recommended Actions**"
            )

            for recommendation in (
                disease["recommendations"]
            ):

                st.write(
                    f"• {recommendation}"
                )

            # Download report
            report = generate_report(
                uploaded_file.name,
                predicted_class,
                confidence,
                probabilities
            )

            st.download_button(
                "Download Detection Report",
                data=report,
                file_name=(
                    "potato_disease_report.html"
                ),
                mime="text/html"
            )


# ============================================================
# BATCH ANALYSIS
# ============================================================

elif page == "Batch Analysis":

    st.title(
        "Batch Image Analysis"
    )

    st.write(
        "Upload multiple potato leaf images "
        "and analyze them together."
    )

    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        results = []

        progress = st.progress(0)

        total = len(
            uploaded_files
        )

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            (
                predicted_class,
                confidence,
                probabilities,
                _
            ) = predict_image(
                model,
                image
            )

            results.append({
                "Image": uploaded_file.name,
                "Prediction": DISPLAY_NAMES[
                    predicted_class
                ],
                "Confidence": (
                    f"{confidence * 100:.2f}%"
                )
            })

            progress.progress(
                (index + 1) / total
            )

        st.success(
            f"Analyzed {total} images."
        )

        st.dataframe(
            results,
            use_container_width=True
        )


# ============================================================
# EXPLAINABLE AI PAGE
# ============================================================

elif page == "Explainable AI":

    st.title(
        "Explainable AI - Grad-CAM"
    )

    st.write(
        """
        Grad-CAM helps visualize the regions of the
        image that contributed to the CNN prediction.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a potato leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="gradcam_upload"
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        (
            predicted_class,
            confidence,
            probabilities,
            processed_image
        ) = predict_image(
            model,
            image
        )

        predicted_index = int(
            np.argmax(probabilities)
        )

        heatmap, layer_name = (
            make_gradcam(
                model,
                processed_image,
                predicted_index
            )
        )

        st.subheader(
            "Model Prediction"
        )

        st.success(
            f"{DISPLAY_NAMES[predicted_class]} "
            f"({confidence * 100:.2f}%)"
        )

        if heatmap is not None:

            overlay_image = (
                create_gradcam_figure(
                    image,
                    heatmap
                )
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Original Image"
                )

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader(
                    "Grad-CAM Overlay"
                )

                st.image(
                    overlay_image,
                    use_container_width=True
                )

            st.info(
                f"Grad-CAM convolutional layer: "
                f"{layer_name}"
            )

            st.caption(
                "Highlighted regions indicate image "
                "areas that contributed more strongly "
                "to the model's prediction."
            )

        else:

            st.warning(
                "Could not generate Grad-CAM for this model."
            )


# ============================================================
# DISEASE INFORMATION PAGE
# ============================================================

elif page == "Disease Information":

    st.title(
        "Potato Disease Information"
    )

    selected_disease = st.selectbox(
        "Select disease",
        CLASS_NAMES,
        format_func=lambda x: DISPLAY_NAMES[x]
    )

    disease = DISEASE_INFO[
        selected_disease
    ]

    st.header(
        disease["name"]
    )

    st.write(
        disease["description"]
    )

    st.subheader(
        "Symptoms"
    )

    for symptom in disease["symptoms"]:

        st.write(
            f"• {symptom}"
        )

    st.subheader(
        "Recommended Actions"
    )

    for recommendation in (
        disease["recommendations"]
    ):

        st.write(
            f"• {recommendation}"
        )


# ============================================================
# MODEL COMPARISON PAGE
# ============================================================

elif page == "Model Comparison":

    st.title(
        "Model Comparison"
    )

    st.write(
        """
        The project evaluates a lightweight Custom CNN
        against pretrained architectures such as
        ResNet50 and MobileNetV2.
        """
    )

    st.warning(
        "Enter the exact test results from your Kaggle "
        "notebook before using these values in your thesis."
    )

    st.subheader(
        "Architecture Comparison"
    )

    comparison_data = [
        {
            "Model": "Custom CNN",
            "Test Accuracy": "From Kaggle",
            "Test Loss": "From Kaggle",
            "Parameters": "From Kaggle",
            "Remarks": "Lightweight custom architecture"
        },
        {
            "Model": "ResNet50",
            "Test Accuracy": "From Kaggle",
            "Test Loss": "From Kaggle",
            "Parameters": "From Kaggle",
            "Remarks": "Deep pretrained architecture"
        },
        {
            "Model": "MobileNetV2",
            "Test Accuracy": "From Kaggle",
            "Test Loss": "From Kaggle",
            "Parameters": "From Kaggle",
            "Remarks": "Efficient pretrained architecture"
        }
    ]

    st.dataframe(
        comparison_data,
        use_container_width=True
    )

    st.info(
        "Do not put estimated accuracy, loss, or parameter "
        "values in your thesis. Use the exact values obtained "
        "from your notebook."
    )


# ============================================================
# PERFORMANCE DASHBOARD
# ============================================================

elif page == "Performance Dashboard":

    st.title(
        "Model Performance Dashboard"
    )

    st.write(
        """
        This page can present the evaluation results
        obtained from the Kaggle notebook.
        """
    )

    st.warning(
        "The numerical performance values below should be "
        "filled with your actual Kaggle evaluation results."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            "From Kaggle"
        )

    with col2:

        st.metric(
            "Precision",
            "From Kaggle"
        )

    with col3:

        st.metric(
            "Recall",
            "From Kaggle"
        )

    with col4:

        st.metric(
            "F1 Score",
            "From Kaggle"
        )

    st.divider()

    st.subheader(
        "Recommended Evaluation Visualizations"
    )

    st.write(
        "• Confusion Matrix"
    )

    st.write(
        "• Training vs Validation Accuracy"
    )

    st.write(
        "• Training vs Validation Loss"
    )

    st.write(
        "• Precision / Recall / F1-score"
    )

    st.write(
        "• ROC-AUC curve where applicable"
    )

    st.info(
        "These charts should use the actual evaluation "
        "results generated in your Kaggle notebook."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Potato Plant Disease Detection"
)

st.sidebar.caption(
    "Custom Lightweight CNN + Explainable AI"
)
