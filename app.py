import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Potato Disease Detection",
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
            "Early blight is a common potato leaf disease "
            "that produces dark lesions on leaves.",

        "symptoms": [
            "Dark or brown spots on leaves",
            "Concentric ring-like lesions",
            "Older leaves may be affected first",
            "Severe infection can cause leaf drying"
        ],

        "recommendations": [
            "Remove severely affected plant material where appropriate",
            "Maintain good field sanitation",
            "Avoid prolonged leaf wetness",
            "Follow locally recommended disease-management practices"
        ]
    },


    "Potato___Late_blight": {

        "name": "Late Blight",

        "description":
            "Late blight is a serious potato disease that can "
            "spread rapidly under favorable environmental conditions.",

        "symptoms": [
            "Dark irregular lesions",
            "Brown or black affected areas",
            "Rapid disease development may occur",
            "Leaves may become damaged and die"
        ],

        "recommendations": [
            "Remove severely affected plant material where appropriate",
            "Improve field ventilation",
            "Avoid prolonged moisture on leaves",
            "Follow locally recommended disease-management practices"
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

        "recommendations": [
            "Continue regular crop monitoring",
            "Maintain appropriate irrigation and nutrition",
            "Monitor plants regularly for new symptoms"
        ]
    }
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# SAFE MODEL LOADING
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Model could not be loaded."
    )

    st.info(
        "Make sure explainable_cnn_model.keras "
        "is uploaded to the GitHub repository."
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

    # Same preprocessing used during training
    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# PREDICTION
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

    image_array = np.array(
        image.convert("RGB")
    ).astype("float32")

    brightness = float(
        np.mean(image_array)
    )

    contrast = float(
        np.std(image_array)
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

        return True, "Image quality looks good."

    return False, " ".join(problems)


# ============================================================
# HTML REPORT
# ============================================================

def create_report(
    filename,
    predicted_class,
    confidence,
    probabilities
):

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

    recommendations = ""

    for item in disease["recommendations"]:

        recommendations += (
            f"<li>{item}</li>"
        )

    html = f"""
    <html>

    <head>

    <meta charset="UTF-8">

    <title>Potato Disease Detection Report</title>

    <style>

    body {{
        font-family: Arial;
        margin: 40px;
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

    <h1>Potato Disease Detection Report</h1>

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

    <h2>Recommended Actions</h2>

    <ul>
    {recommendations}
    </ul>

    <h2>Class Probabilities</h2>

    <table>

    <tr>
        <th>Class</th>
        <th>Probability</th>
    </tr>

    {probability_rows}

    </table>

    <br>

    <p>
    This result is generated by a machine-learning model
    and should be treated as decision-support information.
    </p>

    </body>

    </html>
    """

    return html


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🥔 Potato Disease AI"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🔍 Disease Detection",
        "📁 Batch Analysis",
        "🔥 Explainable AI",
        "🩺 Disease Information",
        "📊 Model Comparison",
        "📈 Performance"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "🥔 Potato Plant Disease Detection"
    )

    st.write(
        """
        An AI-based potato leaf disease detection
        and decision-support system using a
        lightweight Custom CNN.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Model",
            "Custom CNN"
        )

    with col2:

        st.metric(
            "Classes",
            "3"
        )

    with col3:

        st.metric(
            "Input",
            "224 × 224"
        )

    with col4:

        st.metric(
            "Preprocessing",
            "1 / 255"
        )

    st.divider()

    st.subheader(
        "System Features"
    )

    features = [
        "Single Image Disease Detection",
        "Confidence Score",
        "Class Probability",
        "Image Quality Check",
        "Batch Image Analysis",
        "Explainable AI",
        "Disease Information",
        "Model Comparison",
        "Performance Dashboard",
        "Downloadable Report"
    ]

    for feature in features:

        st.write(
            "✓ " + feature
        )


# ============================================================
# DISEASE DETECTION
# ============================================================

elif page == "🔍 Disease Detection":

    st.title(
        "🔍 Potato Disease Detection"
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
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                (
                    predicted_class,
                    confidence,
                    probabilities
                ) = predict_image(
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
                    "Prediction: "
                    + DISPLAY_NAMES[
                        predicted_class
                    ]
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence_percent:.2f}%"
                )

            # Confidence threshold

            if confidence < CONFIDENCE_THRESHOLD:

                st.warning(
                    "⚠️ Low confidence prediction. "
                    "Please upload a clearer potato leaf image."
                )

            else:

                st.success(
                    "✓ Prediction confidence is above "
                    "the selected threshold."
                )

            st.divider()

            # Health status

            if predicted_class == (
                "Potato___healthy"
            ):

                st.success(
                    "🌱 The potato leaf appears healthy."
                )

            else:

                st.warning(
                    "⚠️ Disease detected: "
                    + DISPLAY_NAMES[
                        predicted_class
                    ]
                )

            # Probability

            st.subheader(
                "📊 Class Probabilities"
            )

            for i, class_name in enumerate(
                CLASS_NAMES
            ):

                probability = (
                    float(
                        probabilities[i]
                    ) * 100
                )

                st.write(
                    f"{DISPLAY_NAMES[class_name]}: "
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

            st.divider()

            # Disease information

            disease = DISEASE_INFO[
                predicted_class
            ]

            st.subheader(
                "🩺 Disease Information"
            )

            st.write(
                disease["description"]
            )

            st.write(
                "**Symptoms:**"
            )

            for symptom in disease["symptoms"]:

                st.write(
                    "• " + symptom
                )

            st.write(
                "**Recommended Actions:**"
            )

            for recommendation in (
                disease["recommendations"]
            ):

                st.write(
                    "• " + recommendation
                )

            st.divider()

            # Report

            report = create_report(
                uploaded_file.name,
                predicted_class,
                confidence,
                probabilities
            )

            st.download_button(
                "📄 Download Detection Report",
                data=report,
                file_name="potato_disease_report.html",
                mime="text/html",
                use_container_width=True
            )


# ============================================================
# BATCH ANALYSIS
# ============================================================

elif page == "📁 Batch Analysis":

    st.title(
        "📁 Batch Image Analysis"
    )

    st.write(
        "Upload multiple potato leaf images."
    )

    files = st.file_uploader(
        "Choose multiple images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
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

    st.write(
        """
        This section is reserved for Grad-CAM based
        visualization of the regions influencing the
        CNN prediction.
        """
    )

    st.info(
        """
        Grad-CAM requires access to the internal
        convolutional layer of your trained model.
        We will connect this section to your exact
        CNN architecture after confirming the model's
        convolutional layer name.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload image for explanation",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="explain_image"
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        (
            predicted_class,
            confidence,
            probabilities
        ) = predict_image(
            image
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

        st.warning(
            "Grad-CAM visualization will be enabled "
            "after connecting it to the exact convolutional "
            "layer of your trained model."
        )


# ============================================================
# DISEASE INFORMATION
# ============================================================

elif page == "🩺 Disease Information":

    st.title(
        "🩺 Potato Disease Information"
    )

    selected = st.selectbox(
        "Select a class",
        CLASS_NAMES,
        format_func=lambda x:
            DISPLAY_NAMES[x]
    )

    disease = DISEASE_INFO[
        selected
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
            "• " + symptom
        )

    st.subheader(
        "Recommended Actions"
    )

    for recommendation in (
        disease["recommendations"]
    ):

        st.write(
            "• " + recommendation
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
        Comparison of the architectures evaluated
        during the research.
        """
    )

    st.warning(
        "Replace the values below with the exact results "
        "from your Kaggle notebook."
    )

    data = [

        {
            "Model": "Custom CNN",
            "Test Accuracy": "From Kaggle",
            "Test Loss": "From Kaggle",
            "Parameters": "From Kaggle",
            "Remarks": "Lightweight custom model"
        },

        {
            "Model": "ResNet50",
            "Test Accuracy": "From Kaggle",
            "Test Loss": "From Kaggle",
            "Parameters": "From Kaggle",
            "Remarks": "Deep pretrained model"
        },

        {
            "Model": "MobileNetV2",
            "Test Accuracy": "From Kaggle",
            "Test Loss": "From Kaggle",
            "Parameters": "From Kaggle",
            "Remarks": "Efficient pretrained model"
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
        "📈 Model Performance Dashboard"
    )

    st.warning(
        "Insert your actual Kaggle evaluation values here."
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
        "✓ Confusion Matrix"
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
        "✓ Training / Validation Curves"
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Potato Plant Disease Detection System"
)

st.sidebar.caption(
    "Custom Lightweight CNN"
)
