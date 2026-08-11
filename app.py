import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
import os
import json
import time


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Potato Plant Disease Detection",
    page_icon="🥔",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "best_custom_cnn.keras"

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
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Potato___Early_blight": {

        "name": "Early Blight",

        "description":
            "Early blight is a common fungal disease of potato "
            "plants that mainly affects leaves and can reduce "
            "plant productivity.",

        "symptoms": [
            "Brown or dark circular spots on leaves",
            "Concentric ring-like patterns may appear",
            "Older leaves are usually affected first",
            "Severely infected leaves may turn yellow and die"
        ],

        "causes": [
            "Fungal infection",
            "Warm and humid environmental conditions",
            "Poor air circulation",
            "Infected plant debris"
        ],

        "solution": [
            "Remove severely infected leaves",
            "Keep proper spacing between plants",
            "Avoid unnecessary overhead irrigation",
            "Maintain good field sanitation",
            "Use appropriate fungicide according to local agricultural guidance"
        ],

        "prevention": [
            "Use healthy planting material",
            "Remove infected plant debris",
            "Maintain proper plant spacing",
            "Avoid prolonged leaf wetness",
            "Monitor plants regularly"
        ]
    },


    "Potato___Late_blight": {

        "name": "Late Blight",

        "description":
            "Late blight is a destructive potato disease that "
            "can spread rapidly under cool and humid conditions.",

        "symptoms": [
            "Dark brown or black irregular lesions",
            "Water-soaked appearance may occur",
            "Lesions can expand rapidly",
            "Severely affected leaves may collapse"
        ],

        "causes": [
            "Pathogen infection",
            "Cool and humid weather",
            "Extended leaf wetness",
            "Poor field sanitation"
        ],

        "solution": [
            "Remove severely infected plant material",
            "Improve air circulation",
            "Avoid prolonged leaf wetness",
            "Monitor surrounding plants carefully",
            "Use appropriate fungicide according to local agricultural guidance"
        ],

        "prevention": [
            "Use disease-free planting material",
            "Monitor plants frequently",
            "Maintain good field sanitation",
            "Avoid excessive irrigation",
            "Control disease early before rapid spread"
        ]
    },


    "Potato___healthy": {

        "name": "Healthy",

        "description":
            "The model classified the uploaded potato leaf "
            "as healthy.",

        "symptoms": [
            "No major visible disease symptoms detected",
            "Leaf structure appears generally healthy",
            "No strong disease-related pattern detected by the model"
        ],

        "causes": [
            "No major disease pattern detected"
        ],

        "solution": [
            "Continue regular monitoring",
            "Maintain proper irrigation",
            "Provide balanced plant nutrition",
            "Maintain good field sanitation"
        ],

        "prevention": [
            "Use healthy planting material",
            "Monitor leaves regularly",
            "Maintain proper spacing",
            "Control pests and weeds",
            "Maintain appropriate field hygiene"
        ]
    }
}


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        return None

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception as e:

        st.error(
            "❌ Model could not be loaded."
        )

        st.code(str(e))

        return None


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None


# ============================================================
# HEADER
# ============================================================

st.title("🥔 Potato Plant Disease Detection System")

st.markdown(
    """
    ### AI-powered Potato Leaf Analysis

    Upload **one potato leaf image** from the Home section.
    The same image will automatically be used for:

    **Disease Detection → Explainable AI → Disease Information
    → Causes & Solutions**
    """
)


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        f"""
        ❌ Model file not found.

        Please make sure this file exists in your GitHub
        repository:

        `{MODEL_PATH}`
        """
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def get_prediction(image):

    processed = preprocess_image(
        image
    )

    predictions = model.predict(
        processed,
        verbose=0
    )

    predictions = np.asarray(
        predictions
    )[0]

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


def find_last_conv_layer():

    for layer in reversed(model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            return layer.name

    return None


def make_gradcam_heatmap(
    image,
    predicted_index
):

    last_conv_layer_name = (
        find_last_conv_layer()
    )

    if last_conv_layer_name is None:

        raise ValueError(
            "No Conv2D layer found in model."
        )

    last_conv_layer = model.get_layer(
        last_conv_layer_name
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    img_array = preprocess_image(
        image
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(img_array)
        )

        class_score = predictions[
            :, predicted_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients could not be calculated."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs *
        pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_heatmap = tf.reduce_max(
        heatmap
    )

    if float(max_heatmap) > 0:

        heatmap = (
            heatmap / max_heatmap
        )

    return (
        heatmap.numpy(),
        last_conv_layer_name
    )


def create_heatmap_overlay(
    image,
    heatmap
):

    original = image.convert(
        "RGB"
    )

    original = original.resize(
        IMG_SIZE
    )

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        IMG_SIZE
    )

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    # Simple RGB heatmap
    red = heatmap_array

    green = np.sqrt(
        heatmap_array
    )

    blue = 1.0 - heatmap_array

    colored = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    colored = np.uint8(
        np.clip(
            colored * 255,
            0,
            255
        )
    )

    heatmap_rgb = Image.fromarray(
        colored
    )

    overlay = Image.blend(
        original,
        heatmap_rgb,
        0.45
    )

    return overlay


def calculate_metrics(
    y_true,
    y_pred,
    number_of_classes
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64
    )

    accuracy = float(
        np.mean(
            y_true == y_pred
        )
    )

    confusion = np.zeros(
        (
            number_of_classes,
            number_of_classes
        ),
        dtype=np.int64
    )

    for actual, predicted in zip(
        y_true,
        y_pred
    ):

        if (
            0 <= actual < number_of_classes
            and
            0 <= predicted < number_of_classes
        ):

            confusion[
                actual,
                predicted
            ] += 1

    precisions = []
    recalls = []
    f1_scores = []

    for i in range(
        number_of_classes
    ):

        tp = confusion[i, i]

        fp = (
            np.sum(
                confusion[:, i]
            ) - tp
        )

        fn = (
            np.sum(
                confusion[i, :]
            ) - tp
        )

        if (
            tp + fp
        ) > 0:

            precision = (
                tp /
                (tp + fp)
            )

        else:

            precision = 0.0

        if (
            tp + fn
        ) > 0:

            recall = (
                tp /
                (tp + fn)
            )

        else:

            recall = 0.0

        if (
            precision + recall
        ) > 0:

            f1 = (
                2 *
                precision *
                recall /
                (
                    precision +
                    recall
                )
            )

        else:

            f1 = 0.0

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
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1_scores)),
        confusion
    )


# ============================================================
# NAVIGATION
# ============================================================

pages = [
    "🏠 Home",
    "🔍 Disease Detection",
    "🔥 Explainable AI",
    "📚 Disease Information",
    "🩺 Causes & Solutions",
    "📦 Batch Analysis",
    "📊 Performance",
    "⚖️ Model Comparison"
]

page = st.sidebar.radio(
    "Navigation",
    pages
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.header(
        "📷 Upload Potato Leaf Image"
    )

    uploaded_file = st.file_uploader(
        "Choose a potato leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="home_uploader"
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.session_state.uploaded_image = (
                image
            )

            with st.spinner(
                "Analyzing potato leaf..."
            ):

                (
                    predicted_class,
                    confidence,
                    probabilities
                ) = get_prediction(
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

            st.success(
                "✓ Image analyzed successfully!"
            )

            col1, col2 = st.columns(
                2
            )

            with col1:

                st.image(
                    image,
                    caption="Uploaded Potato Leaf",
                    use_container_width=True
                )

            with col2:

                st.subheader(
                    "🎯 Quick Result"
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

                if predicted_class == (
                    "Potato___healthy"
                ):

                    st.success(
                        "🌱 The leaf appears healthy."
                    )

                else:

                    st.warning(
                        "⚠️ Disease pattern detected."
                    )

            st.divider()

            st.info(
                """
                Image uploaded successfully.

                You do NOT need to upload the image again.

                Open the other sections from the sidebar.
                The same image and prediction will be used.
                """
            )

        except Exception as e:

            st.error(
                "❌ Could not process image."
            )

            st.code(
                str(e)
            )

    else:

        st.info(
            "👆 Upload a potato leaf image to start."
        )


# ============================================================
# CHECK IMAGE FOR IMAGE-BASED PAGES
# ============================================================

elif page in [
    "🔍 Disease Detection",
    "🔥 Explainable AI",
    "📚 Disease Information",
    "🩺 Causes & Solutions"
]:

    if (
        st.session_state.uploaded_image
        is None
    ):

        st.warning(
            """
            ⚠️ No image uploaded.

            Please go to **Home** and upload
            a potato leaf image first.
            """
        )

        st.stop()


# ============================================================
# DISEASE DETECTION
# ============================================================

if page == "🔍 Disease Detection":

    st.header(
        "🔍 Disease Detection"
    )

    image = (
        st.session_state.uploaded_image
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

    col1, col2 = st.columns(
        2
    )

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

        st.subheader(
            "Class Probabilities"
        )

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = float(
                probabilities[i]
            )

            st.write(
                DISPLAY_NAMES[
                    class_name
                ]
            )

            st.progress(
                min(
                    max(
                        probability,
                        0.0
                    ),
                    1.0
                )
            )

            st.caption(
                f"{probability * 100:.2f}%"
            )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🔥 Explainable AI":

    st.header(
        "🔥 Explainable AI - Grad-CAM"
    )

    image = (
        st.session_state.uploaded_image
    )

    predicted_class = (
        st.session_state.prediction
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
        Grad-CAM highlights the image regions that
        contributed to the model's prediction.
        """
    )

    try:

        with st.spinner(
            "Generating Grad-CAM..."
        ):

            (
                heatmap,
                layer_name
            ) = make_gradcam_heatmap(
                image,
                predicted_index
            )

            overlay = (
                create_heatmap_overlay(
                    image,
                    heatmap
                )
            )

        st.success(
            "✓ Grad-CAM generated successfully."
        )

        st.write(
            f"**Prediction:** "
            f"{DISPLAY_NAMES[predicted_class]}"
        )

        st.write(
            f"**Grad-CAM Layer:** `{layer_name}`"
        )

        col1, col2 = st.columns(
            2
        )

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
                "Grad-CAM Heatmap"
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
            The highlighted regions indicate areas that
            contributed more strongly to the CNN prediction.

            This visualization can help determine whether
            the model is focusing on relevant leaf regions
            instead of unrelated background areas.
            """
        )

        max_activation = float(
            np.max(heatmap)
        )

        mean_activation = float(
            np.mean(heatmap)
        )

        c1, c2 = st.columns(
            2
        )

        with c1:

            st.metric(
                "Maximum Activation",
                f"{max_activation:.4f}"
            )

        with c2:

            st.metric(
                "Mean Activation",
                f"{mean_activation:.4f}"
            )

    except Exception as e:

        st.error(
            "❌ Grad-CAM could not be generated."
        )

        st.code(
            str(e)
        )

        st.info(
            """
            The model was successfully loaded, but this
            particular model architecture may not expose a
            compatible convolutional layer for Grad-CAM.
            """
        )


# ============================================================
# DISEASE INFORMATION
# ============================================================

elif page == "📚 Disease Information":

    st.header(
        "📚 Disease Information"
    )

    predicted_class = (
        st.session_state.prediction
    )

    info = DISEASE_INFO[
        predicted_class
    ]

    st.subheader(
        f"🌿 {info['name']}"
    )

    st.write(
        info["description"]
    )

    st.divider()

    st.subheader(
        "🔎 Common Symptoms"
    )

    for symptom in info[
        "symptoms"
    ]:

        st.write(
            f"• {symptom}"
        )

    st.divider()

    st.subheader(
        "🦠 Possible Causes"
    )

    for cause in info[
        "causes"
    ]:

        st.write(
            f"• {cause}"
        )


# ============================================================
# CAUSES & SOLUTIONS
# ============================================================

elif page == "🩺 Causes & Solutions":

    st.header(
        "🩺 Why Does This Problem Occur?"
    )

    predicted_class = (
        st.session_state.prediction
    )

    confidence = (
        st.session_state.confidence
    )

    info = DISEASE_INFO[
        predicted_class
    ]

    st.subheader(
        f"Detected Condition: {info['name']}"
    )

    st.metric(
        "Model Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.divider()

    st.subheader(
        "❓ Why Can This Problem Occur?"
    )

    for cause in info[
        "causes"
    ]:

        st.write(
            f"• {cause}"
        )

    st.divider()

    st.subheader(
        "⚡ What Should Be Done?"
    )

    for solution in info[
        "solution"
    ]:

        st.write(
            f"✅ {solution}"
        )

    st.divider()

    st.subheader(
        "🛡️ Prevention"
    )

    for prevention in info[
        "prevention"
    ]:

        st.write(
            f"• {prevention}"
        )

    st.warning(
        """
        ⚠️ This application provides AI-based preliminary
        classification and general information. For serious
        crop disease problems, consult a qualified
        agricultural professional.
        """
    )


# ============================================================
# BATCH ANALYSIS
# ============================================================

elif page == "📦 Batch Analysis":

    st.header(
        "📦 Batch Analysis"
    )

    st.write(
        """
        Upload multiple potato leaf images and analyze
        them together.
        """
    )

    files = st.file_uploader(
        "Upload multiple leaf images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if files:

        results = []

        progress = st.progress(
            0
        )

        total = len(files)

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
                    _
                ) = get_prediction(
                    image
                )

                results.append(
                    {
                        "Image":
                            file.name,

                        "Prediction":
                            DISPLAY_NAMES[
                                predicted_class
                            ],

                        "Confidence":
                            f"{confidence * 100:.2f}%"
                    }
                )

            except Exception as e:

                results.append(
                    {
                        "Image":
                            file.name,

                        "Prediction":
                            "Error",

                        "Confidence":
                            "-"
                    }
                )

            progress.progress(
                int(
                    ((index + 1) / total)
                    * 100
                )
            )

        st.success(
            f"✓ {total} images analyzed."
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        # Summary

        st.subheader(
            "📊 Batch Summary"
        )

        disease_count = 0
        healthy_count = 0

        for result in results:

            prediction = result[
                "Prediction"
            ]

            if prediction == "Healthy":

                healthy_count += 1

            elif prediction != "Error":

                disease_count += 1

        c1, c2 = st.columns(
            2
        )

        with c1:

            st.success(
                f"Healthy: {healthy_count}"
            )

        with c2:

            st.warning(
                f"Disease Detected: {disease_count}"
            )

    else:

        st.info(
            "Upload multiple images to start batch analysis."
        )


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "📊 Performance":

    st.header(
        "📊 Model Performance"
    )

    st.write(
        """
        This section evaluates the trained Custom CNN
        using the test dataset available in the repository.
        """
    )

    TEST_DIR = "test"

    # --------------------------------------------------------
    # TEST DATASET CHECK
    # --------------------------------------------------------

    if not os.path.exists(
        TEST_DIR
    ):

        st.warning(
            """
            ⚠️ Test dataset folder was not found.

            To calculate actual Accuracy, Precision, Recall,
            F1 Score and Confusion Matrix, add a `test` folder
            to your GitHub repository.

            Example:

            test/
            ├── Potato___Early_blight/
            ├── Potato___Late_blight/
            └── Potato___healthy/
            """
        )

    else:

        try:

            test_dataset = (
                tf.keras.utils.image_dataset_from_directory(

                    TEST_DIR,

                    image_size=IMG_SIZE,

                    batch_size=32,

                    shuffle=False
                )
            )

            dataset_classes = (
                test_dataset.class_names
            )

            y_true = []
            y_pred = []

            start_time = time.time()

            for images, labels in (
                test_dataset
            ):

                predictions = model.predict(
                    images,
                    verbose=0
                )

                predictions = np.asarray(
                    predictions
                )

                predicted_labels = (
                    np.argmax(
                        predictions,
                        axis=1
                    )
                )

                y_true.extend(
                    labels.numpy().tolist()
                )

                y_pred.extend(
                    predicted_labels.tolist()
                )

            inference_time = (
                time.time()
                - start_time
            )

            (
                accuracy,
                precision,
                recall,
                f1,
                confusion
            ) = calculate_metrics(

                y_true,

                y_pred,

                len(dataset_classes)
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            st.subheader(
                "🎯 Evaluation Components"
            )

            c1, c2, c3, c4 = st.columns(
                4
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

            # ------------------------------------------------
            # Confusion Matrix
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🔲 Confusion Matrix"
            )

            matrix_rows = []

            for i in range(
                len(dataset_classes)
            ):

                row = {
                    "Actual":
                        dataset_classes[i]
                }

                for j in range(
                    len(dataset_classes)
                ):

                    row[
                        dataset_classes[j]
                    ] = int(
                        confusion[i, j]
                    )

                matrix_rows.append(
                    row
                )

            st.dataframe(
                matrix_rows,
                use_container_width=True
            )

            # ------------------------------------------------
            # Correct / Incorrect
            # ------------------------------------------------

            correct = int(
                np.sum(
                    np.asarray(y_true)
                    ==
                    np.asarray(y_pred)
                )
            )

            incorrect = int(
                len(y_true)
                - correct
            )

            c1, c2 = st.columns(
                2
            )

            with c1:

                st.success(
                    f"✓ Correct Predictions: {correct}"
                )

            with c2:

                st.error(
                    f"✗ Incorrect Predictions: {incorrect}"
                )

            # ------------------------------------------------
            # Efficiency
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "⚡ Efficiency"
            )

            c1, c2 = st.columns(
                2
            )

            with c1:

                st.metric(
                    "Test Images",
                    len(y_true)
                )

            with c2:

                st.metric(
                    "Evaluation Time",
                    f"{inference_time:.2f} sec"
                )

        except Exception as e:

            st.error(
                "❌ Test dataset evaluation failed."
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

    HISTORY_PATH = (
        "training_history.json"
    )

    if not os.path.exists(
        HISTORY_PATH
    ):

        st.info(
            """
            `training_history.json` was not found.

            Therefore Training/Validation Accuracy and Loss
            cannot be reconstructed from the `.keras` model
            alone.

            Export the training history from Kaggle and place
            `training_history.json` beside `app.py`.
            """
        )

    else:

        try:

            with open(
                HISTORY_PATH,
                "r"
            ) as f:

                history = json.load(
                    f
                )

            accuracy_key = None
            validation_accuracy_key = None
            loss_key = None
            validation_loss_key = None

            possible_accuracy_keys = [
                "accuracy",
                "acc"
            ]

            possible_val_accuracy_keys = [
                "val_accuracy",
                "val_acc"
            ]

            for key in possible_accuracy_keys:

                if key in history:

                    accuracy_key = key

                    break

            for key in possible_val_accuracy_keys:

                if key in history:

                    validation_accuracy_key = key

                    break

            if "loss" in history:

                loss_key = "loss"

            if "val_loss" in history:

                validation_loss_key = "val_loss"

            # ----------------------------------------------
            # Accuracy
            # ----------------------------------------------

            if (
                accuracy_key is not None
                and
                validation_accuracy_key is not None
            ):

                accuracy_data = []

                total_epochs = len(
                    history[
                        accuracy_key
                    ]
                )

                for epoch in range(
                    total_epochs
                ):

                    accuracy_data.append(
                        {
                            "Epoch":
                                epoch + 1,

                            "Training Accuracy":
                                history[
                                    accuracy_key
                                ][epoch],

                            "Validation Accuracy":
                                history[
                                    validation_accuracy_key
                                ][epoch]
                        }
                    )

                st.write(
                    "Training vs Validation Accuracy"
                )

                st.line_chart(
                    accuracy_data,
                    x="Epoch",
                    y=[
                        "Training Accuracy",
                        "Validation Accuracy"
                    ]
                )

            # ----------------------------------------------
            # Loss
            # ----------------------------------------------

            if (
                loss_key is not None
                and
                validation_loss_key is not None
            ):

                loss_data = []

                total_epochs = len(
                    history[
                        loss_key
                    ]
                )

                for epoch in range(
                    total_epochs
                ):

                    loss_data.append(
                        {
                            "Epoch":
                                epoch + 1,

                            "Training Loss":
                                history[
                                    loss_key
                                ][epoch],

                            "Validation Loss":
                                history[
                                    validation_loss_key
                                ][epoch]
                        }
                    )

                st.write(
                    "Training vs Validation Loss"
                )

                st.line_chart(
                    loss_data,
                    x="Epoch",
                    y=[
                        "Training Loss",
                        "Validation Loss"
                    ]
                )

        except Exception as e:

            st.error(
                "❌ Training history could not be loaded."
            )

            st.code(
                str(e)
            )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "⚖️ Model Comparison":

    st.header(
        "⚖️ Model Comparison"
    )

    st.write(
        """
        Comparison of the three model approaches used in
        the research project.
        """
    )

    comparison = [

        {
            "Model":
                "Custom CNN",

            "Architecture":
                "3 Conv Blocks + GAP + Dense(256) + Dropout",

            "Type":
                "Proposed Lightweight CNN",

            "Input":
                "224 × 224 × 3",

            "Output":
                "3 Classes"
        },

        {
            "Model":
                "ResNet50",

            "Architecture":
                "Pre-trained ResNet50 + Custom Head",

            "Type":
                "Transfer Learning",

            "Input":
                "224 × 224 × 3",

            "Output":
                "3 Classes"
        },

        {
            "Model":
                "MobileNetV2",

            "Architecture":
                "Pre-trained MobileNetV2 + Custom Head",

            "Type":
                "Transfer Learning",

            "Input":
                "224 × 224 × 3",

            "Output":
                "3 Classes"
        }
    ]

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "🧠 Custom CNN Architecture"
    )

    st.write(
        """
        The proposed Custom CNN is designed as a
        lightweight architecture using convolutional blocks,
        Global Average Pooling, Dense(256), Dropout(0.5)
        and a 3-class softmax output.
        """
    )

    st.subheader(
        "🔄 ResNet50"
    )

    st.write(
        """
        ResNet50 is used as a transfer-learning baseline
        with a custom classification head.
        """
    )

    st.subheader(
        "📱 MobileNetV2"
    )

    st.write(
        """
        MobileNetV2 is used as a lightweight transfer-learning
        comparison model.
        """
    )

    st.info(
        """
        Actual Accuracy, Loss, Parameter Count and inference
        time should be loaded from the recorded experiment
        results. This app does not invent those values.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Potato Plant Disease Detection"
)

st.sidebar.caption(
    "Custom CNN • TensorFlow/Keras • Grad-CAM"
)
