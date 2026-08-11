import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Potato Plant Disease Detection",
    page_icon="🥔",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "explainable_cnn_model.keras"

IMG_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# CLASS NAMES
# ============================================================

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
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Potato___Early_blight": {

        "name": "Early Blight",

        "description":
            "Early blight is a fungal disease that can affect "
            "potato foliage and reduce plant health.",

        "symptoms": [
            "Dark or brown spots on leaves",
            "Concentric ring-like lesions",
            "Older leaves may show symptoms first",
            "Severe infection may cause leaf drying"
        ],

        "causes": [
            "Favorable environmental conditions",
            "Extended leaf wetness",
            "Infected plant material",
            "Poor field sanitation"
        ],

        "prevention": [
            "Maintain good field sanitation",
            "Monitor plants regularly",
            "Avoid unnecessary prolonged leaf wetness",
            "Use healthy planting material",
            "Follow appropriate crop management practices"
        ],

        "management": [
            "Identify affected plants early",
            "Remove or manage severely affected plant material "
            "where appropriate",
            "Improve field ventilation",
            "Reduce prolonged moisture on foliage",
            "Follow locally approved disease-management practices"
        ]
    },


    "Potato___Late_blight": {

        "name": "Late Blight",

        "description":
            "Late blight is a serious potato disease that can "
            "develop rapidly under favorable environmental conditions.",

        "symptoms": [
            "Dark irregular lesions",
            "Brown or black affected areas",
            "Rapid development of symptoms",
            "Leaves may become damaged and die"
        ],

        "causes": [
            "Favorable cool and humid conditions",
            "Extended leaf wetness",
            "Infected plant material",
            "Disease spread between plants"
        ],

        "prevention": [
            "Monitor the crop frequently",
            "Avoid prolonged leaf wetness",
            "Maintain appropriate plant spacing",
            "Remove affected plant material where appropriate",
            "Follow locally recommended disease-management practices"
        ],

        "management": [
            "Act quickly after symptoms are detected",
            "Remove or manage severely affected material where appropriate",
            "Improve field ventilation",
            "Reduce prolonged moisture on leaves",
            "Follow locally approved management recommendations"
        ]
    },


    "Potato___healthy": {

        "name": "Healthy Potato Leaf",

        "description":
            "The model classified this image as a healthy potato leaf.",

        "symptoms": [
            "No major disease pattern detected by the model",
            "Leaf appears comparatively healthy"
        ],

        "causes": [],

        "prevention": [
            "Continue regular crop monitoring",
            "Maintain appropriate irrigation and nutrition",
            "Maintain field hygiene",
            "Monitor regularly for new symptoms"
        ],

        "management": [
            "No disease management is indicated by this prediction",
            "Continue regular monitoring",
            "Maintain good crop management practices"
        ]
    }
}


# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


try:

    model = load_model()

except Exception:

    st.error(
        "❌ Model could not be loaded."
    )

    st.info(
        "Make sure 'explainable_cnn_model.keras' "
        "is available in your GitHub repository."
    )

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    img_array = np.array(
        image
    ).astype("float32")

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

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
        probabilities
    )


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image):

    img_array = np.array(
        image.convert("RGB")
    ).astype("float32")

    brightness = float(
        np.mean(img_array)
    )

    contrast = float(
        np.std(img_array)
    )

    problems = []

    if brightness < 25:

        problems.append(
            "Image is too dark."
        )

    if brightness > 240:

        problems.append(
            "Image is too bright."
        )

    if contrast < 15:

        problems.append(
            "Image has very low contrast."
        )

    if len(problems) == 0:

        return True, "Image quality looks acceptable."

    return False, " ".join(problems)


# ============================================================
# REPORT GENERATOR
# ============================================================

def create_report():

    predicted_class = (
        st.session_state.predicted_class
    )

    confidence = (
        st.session_state.confidence
    )

    probabilities = (
        st.session_state.probabilities
    )

    filename = (
        st.session_state.file_name
    )

    disease = DISEASE_INFO[
        predicted_class
    ]

    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    probability_rows = ""

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            float(probabilities[i]) * 100
        )

        probability_rows += f"""
        <tr>
            <td>{DISPLAY_NAMES[class_name]}</td>
            <td>{probability:.2f}%</td>
        </tr>
        """

    symptoms = ""

    for item in disease["symptoms"]:

        symptoms += f"<li>{item}</li>"

    management = ""

    for item in disease["management"]:

        management += f"<li>{item}</li>"

    html = f"""
    <html>

    <head>

    <meta charset="UTF-8">

    <title>
    Potato Disease Detection Report
    </title>

    <style>

    body {{
        font-family: Arial;
        margin: 40px;
        line-height: 1.6;
    }}

    h1 {{
        text-align: center;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
    }}

    th, td {{
        border: 1px solid #999;
        padding: 10px;
    }}

    </style>

    </head>

    <body>

    <h1>
    Potato Plant Disease Detection Report
    </h1>

    <p>
    <b>Date:</b> {current_time}
    </p>

    <p>
    <b>Image:</b> {filename}
    </p>

    <h2>
    Prediction:
    {disease["name"]}
    </h2>

    <h3>
    Confidence:
    {confidence * 100:.2f}%
    </h3>

    <h2>Disease Information</h2>

    <p>
    {disease["description"]}
    </p>

    <h2>Symptoms</h2>

    <ul>
    {symptoms}
    </ul>

    <h2>Management</h2>

    <ul>
    {management}
    </ul>

    <h2>Class Probabilities</h2>

    <table>

    <tr>
        <th>Class</th>
        <th>Probability</th>
    </tr>

    {probability_rows}

    </table>

    </body>

    </html>
    """

    return html


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title(
    "🥔 Potato Disease AI"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Disease Detection",
        "📁 Batch Analysis",
        "🔥 Explainable AI",
        "🩺 Disease Information",
        "🚑 Disease Solution",
        "📊 Model Comparison",
        "📈 Performance"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "🥔 Potato Plant Disease Detection System"
    )

    st.markdown(
        """
        ### AI-Powered Potato Leaf Analysis

        Upload a potato leaf image once and click
        **Analyze Image**.

        The same image and prediction will then be
        available throughout the application.
        """
    )

    st.divider()

    st.subheader(
        "📷 Upload Potato Leaf Image"
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="home_uploader"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.session_state.uploaded_image = image

        st.session_state.file_name = (
            uploaded_file.name
        )

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            width=500
        )

        quality_ok, quality_message = (
            check_image_quality(image)
        )

        if quality_ok:

            st.success(
                "✓ " + quality_message
            )

        else:

            st.warning(
                "⚠️ " + quality_message
            )

        if st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "AI is analyzing the image..."
            ):

                (
                    predicted_class,
                    confidence,
                    probabilities
                ) = predict_image(
                    image
                )

            st.session_state.predicted_class = (
                predicted_class
            )

            st.session_state.confidence = (
                confidence
            )

            st.session_state.probabilities = (
                probabilities
            )

            st.session_state.prediction_done = True

            st.success(
                "✓ Analysis completed successfully."
            )

    if st.session_state.prediction_done:

        st.divider()

        predicted_class = (
            st.session_state.predicted_class
        )

        confidence = (
            st.session_state.confidence
        )

        st.subheader(
            "🎯 AI Result"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                "Prediction: "
                + DISPLAY_NAMES[
                    predicted_class
                ]
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        st.info(
            "Your image has been analyzed. "
            "You can now explore all tabs without "
            "uploading the image again."
        )

    else:

        st.info(
            "👆 Upload an image and click "
            "'Analyze Image' to begin."
        )


# ============================================================
# ALL OTHER TABS REQUIRE ANALYZED IMAGE
# ============================================================

elif page != "📁 Batch Analysis" and page != "📊 Model Comparison" and page != "📈 Performance":

    if st.session_state.uploaded_image is None:

        st.warning(
            "⚠️ Please upload and analyze an image "
            "from the Home page first."
        )

        st.stop()

    if not st.session_state.prediction_done:

        st.warning(
            "⚠️ Please click 'Analyze Image' "
            "on the Home page first."
        )

        st.stop()


# ============================================================
# DISEASE DETECTION
# ============================================================

if page == "🔍 Disease Detection":

    st.title(
        "🔍 Disease Detection"
    )

    image = (
        st.session_state.uploaded_image
    )

    predicted_class = (
        st.session_state.predicted_class
    )

    confidence = (
        st.session_state.confidence
    )

    probabilities = (
        st.session_state.probabilities
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Analyzed Potato Leaf",
            use_container_width=True
        )

    with col2:

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

    st.divider()

    if confidence < CONFIDENCE_THRESHOLD:

        st.warning(
            "⚠️ Low-confidence prediction. "
            "Please consider using a clearer image."
        )

    else:

        st.success(
            "✓ Prediction confidence is above "
            "the selected threshold."
        )

    st.subheader(
        "📊 Class Probabilities"
    )

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            float(probabilities[i]) * 100
        )

        st.write(
            f"{DISPLAY_NAMES[class_name]} — "
            f"{probability:.2f}%"
        )

        st.progress(
            int(
                min(
                    probability,
                    100
                )
            )
        )


# ============================================================
# BATCH ANALYSIS
# ============================================================

elif page == "📁 Batch Analysis":

    st.title(
        "📁 Batch Analysis"
    )

    st.write(
        """
        Analyze multiple potato leaf images at once.
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
        key="batch_upload"
    )

    if files:

        results = []

        progress = st.progress(0)

        total = len(files)

        for index, file in enumerate(files):

            image = Image.open(
                file
            ).convert("RGB")

            (
                predicted_class,
                confidence,
                _
            ) = predict_image(
                image
            )

            results.append({
                "Image": file.name,
                "Prediction":
                    DISPLAY_NAMES[
                        predicted_class
                    ],
                "Confidence":
                    f"{confidence * 100:.2f}%"
            })

            progress.progress(
                int(
                    ((index + 1) / total) * 100
                )
            )

        st.success(
            f"{total} images analyzed successfully."
        )

        st.dataframe(
            results,
            use_container_width=True
        )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🔥 Explainable AI":

    st.title(
        "🔥 Explainable AI"
    )

    image = (
        st.session_state.uploaded_image
    )

    predicted_class = (
        st.session_state.predicted_class
    )

    confidence = (
        st.session_state.confidence
    )

    st.image(
        image,
        caption="Analyzed Image",
        width=500
    )

    st.divider()

    st.subheader(
        "Model Decision"
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

    st.divider()

    st.subheader(
        "What does Explainable AI mean?"
    )

    st.write(
        """
        Explainable AI is used to understand which
        regions or features of an image influenced
        the model's prediction.
        """
    )

    st.info(
        """
        🔥 Grad-CAM can be added here to generate
        a heatmap showing the important regions
        used by the CNN.
        """
    )

    st.warning(
        """
        The Grad-CAM calculation should be connected
        to the exact final convolutional layer of your
        trained Custom CNN.
        """
    )


# ============================================================
# DISEASE INFORMATION
# ============================================================

elif page == "🩺 Disease Information":

    st.title(
        "🩺 Disease Information"
    )

    predicted_class = (
        st.session_state.predicted_class
    )

    image = (
        st.session_state.uploaded_image
    )

    disease = DISEASE_INFO[
        predicted_class
    ]

    st.image(
        image,
        caption="Analyzed Potato Leaf",
        width=400
    )

    st.divider()

    st.header(
        disease["name"]
    )

    st.write(
        disease["description"]
    )

    st.subheader(
        "🔎 Symptoms"
    )

    for symptom in disease["symptoms"]:

        st.write(
            "• " + symptom
        )

    st.subheader(
        "⚠️ Possible Causes / Favorable Conditions"
    )

    for cause in disease["causes"]:

        st.write(
            "• " + cause
        )

    st.subheader(
        "🛡️ Prevention"
    )

    for item in disease["prevention"]:

        st.write(
            "• " + item
        )


# ============================================================
# DISEASE SOLUTION
# ============================================================

elif page == "🚑 Disease Solution":

    st.title(
        "🚑 Disease Solution & Management"
    )

    predicted_class = (
        st.session_state.predicted_class
    )

    confidence = (
        st.session_state.confidence
    )

    disease = DISEASE_INFO[
        predicted_class
    ]

    st.subheader(
        "AI Detected Disease"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            disease["name"]
        )

    with col2:

        st.metric(
            "AI Confidence",
            f"{confidence * 100:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # WHY
    # --------------------------------------------------------

    st.subheader(
        "❓ Why can this problem occur?"
    )

    if disease["causes"]:

        for cause in disease["causes"]:

            st.write(
                "• " + cause
            )

    else:

        st.success(
            "No disease was detected."
        )

    # --------------------------------------------------------
    # SYMPTOMS
    # --------------------------------------------------------

    st.subheader(
        "🔎 What symptoms should you look for?"
    )

    for symptom in disease["symptoms"]:

        st.write(
            "• " + symptom
        )

    # --------------------------------------------------------
    # IMMEDIATE ACTION
    # --------------------------------------------------------

    st.subheader(
        "⚡ What should I do first?"
    )

    if predicted_class == "Potato___healthy":

        st.success(
            """
            The model classified the leaf as healthy.
            Continue regular monitoring and maintain
            good crop management practices.
            """
        )

    else:

        st.warning(
            """
            Early identification and appropriate crop
            management are important when disease symptoms
            are detected.
            """
        )

        for item in disease["management"]:

            st.write(
                "• " + item
            )

    # --------------------------------------------------------
    # PREVENTION
    # --------------------------------------------------------

    st.subheader(
        "🛡️ How can I reduce future risk?"
    )

    for item in disease["prevention"]:

        st.write(
            "• " + item
        )

    # --------------------------------------------------------
    # FASTEST PRACTICAL RESPONSE
    # --------------------------------------------------------

    st.subheader(
        "🚨 Quick Action Guide"
    )

    if predicted_class == "Potato___healthy":

        st.success(
            """
            1. Continue monitoring the crop.

            2. Maintain good field hygiene.

            3. Check new leaves regularly.

            4. Take action if symptoms appear.
            """
        )

    elif predicted_class == "Potato___Early_blight":

        st.warning(
            """
            1. Check surrounding plants for similar symptoms.

            2. Manage affected plant material where appropriate.

            3. Reduce prolonged moisture on foliage.

            4. Maintain field sanitation.

            5. Follow locally approved disease-management
               recommendations.
            """
        )

    elif predicted_class == "Potato___Late_blight":

        st.error(
            """
            1. Inspect nearby plants immediately.

            2. Manage severely affected plant material
               where appropriate.

            3. Reduce prolonged leaf wetness.

            4. Improve field ventilation.

            5. Follow locally approved disease-management
               recommendations promptly.
            """
        )

    # --------------------------------------------------------
    # IMPORTANT NOTICE
    # --------------------------------------------------------

    st.divider()

    st.info(
        """
        ⚠️ Important:

        This AI system provides image-based decision support.
        The prediction should not replace professional
        agricultural diagnosis.

        For chemical control, always follow locally approved
        product labels and recommendations from qualified
        agricultural professionals.
        """
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":

    st.title(
        "📊 Model Comparison"
    )

    st.write(
        """
        Comparison between the models evaluated in
        the research.
        """
    )

    st.warning(
        "Replace these placeholders with the exact values "
        "from your Kaggle notebook."
    )

    data = [

        {
            "Model": "Custom CNN",
            "Test Accuracy": "From Kaggle",
            "Test Loss": "From Kaggle",
            "Parameters": "From Kaggle",
            "Remarks":
                "Lightweight custom model"
        },

        {
            "Model": "ResNet50",
            "Test Accuracy": "From Kaggle",
            "Test Loss": "From Kaggle",
            "Parameters": "From Kaggle",
            "Remarks":
                "Deep pretrained model"
        },

        {
            "Model": "MobileNetV2",
            "Test Accuracy": "From Kaggle",
            "Test Loss": "From Kaggle",
            "Parameters": "From Kaggle",
            "Remarks":
                "Efficient pretrained model"
        }
    ]

    st.dataframe(
        data,
        use_container_width=True
    )


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "📈 Performance":

    st.title(
        "📈 Model Performance"
    )

    st.warning(
        "Insert the exact evaluation results from "
        "your Kaggle notebook."
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
        "Evaluation Components"
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


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "🥔 Potato Plant Disease Detection System"
)

st.sidebar.caption(
    "Custom Lightweight CNN"
)
