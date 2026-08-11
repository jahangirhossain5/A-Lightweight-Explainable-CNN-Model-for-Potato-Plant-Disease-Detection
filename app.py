import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Potato Plant Disease Detection",
    page_icon="🥔",
    layout="wide"
)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "explainable_cnn_model.keras"
IMG_SIZE = (224, 224)

# ============================================================
# 38 PLANTVILLAGE CLASS NAMES
# ============================================================

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
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "cause": "Usually associated with the fungus Alternaria solani.",
        "symptoms": [
            "Dark brown to black spots on older leaves",
            "Spots may develop concentric ring patterns",
            "Leaves may gradually yellow and dry"
        ],
        "management": [
            "Remove severely infected leaves",
            "Maintain proper field sanitation",
            "Avoid prolonged leaf wetness",
            "Use recommended fungicides according to local agricultural guidance",
            "Maintain adequate plant nutrition"
        ],
        "prevention": [
            "Use healthy planting material",
            "Avoid excessive irrigation",
            "Maintain good spacing and airflow",
            "Remove infected plant debris"
        ]
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "cause": "Caused by the oomycete pathogen Phytophthora infestans.",
        "symptoms": [
            "Irregular dark brown or black lesions",
            "Rapid leaf deterioration",
            "White growth may appear under humid conditions",
            "Disease can spread rapidly during cool and wet weather"
        ],
        "management": [
            "Remove heavily infected plant material",
            "Improve air circulation",
            "Avoid unnecessary overhead irrigation",
            "Use locally recommended fungicide programs",
            "Monitor nearby plants frequently"
        ],
        "prevention": [
            "Use disease-free seed potatoes",
            "Avoid excessive moisture",
            "Provide adequate spacing",
            "Remove infected crop residues"
        ]
    },

    "Potato___healthy": {
        "name": "Healthy Potato Plant",
        "cause": "No major disease pattern was detected by the model.",
        "symptoms": [
            "Healthy green foliage",
            "No strong disease pattern detected"
        ],
        "management": [
            "Continue regular monitoring",
            "Maintain balanced irrigation",
            "Maintain proper nutrition",
            "Keep the field clean"
        ],
        "prevention": [
            "Use healthy planting material",
            "Maintain good field sanitation",
            "Regularly inspect leaves"
        ]
    }
}

# ============================================================
# GENERIC INFORMATION FOR OTHER PLANT DISEASES
# ============================================================

def get_disease_info(class_name):

    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]

    display_name = class_name.split("___")[-1].replace("_", " ")

    return {
        "name": display_name,
        "cause": "This class was detected from the PlantVillage disease classification dataset.",
        "symptoms": [
            "Visible symptoms may vary depending on disease severity.",
            "The model prediction should be confirmed with field observation."
        ],
        "management": [
            "Remove severely affected plant material where appropriate.",
            "Maintain proper irrigation and field sanitation.",
            "Improve airflow around plants.",
            "Consult an agricultural expert before applying treatment."
        ],
        "prevention": [
            "Use healthy planting material.",
            "Monitor plants regularly.",
            "Maintain good sanitation.",
            "Avoid unnecessary moisture on leaves."
        ]
    }


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model file not found: {MODEL_PATH}"
        )

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model, None

    except Exception as e:
        return None, str(e)


model, model_error = load_model()

# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

    return None


last_conv_layer = None

if model is not None:
    last_conv_layer = find_last_conv_layer(model)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    resized = image.resize(
        IMG_SIZE
    )

    array = np.asarray(
        resized,
        dtype=np.float32
    )

    array = array / 255.0

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    image_array,
    model,
    conv_layer
):

    if model is None:
        return None, "Model is not loaded."

    if conv_layer is None:
        return None, "No Conv2D layer was found."

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        # Build a gradient model from the actual loaded model.
        # ----------------------------------------------------

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                conv_layer.output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_output, predictions = grad_model(
                image_array,
                training=False
            )

            predicted_index = tf.argmax(
                predictions[0]
            )

            class_score = predictions[
                :, predicted_index
            ]

        gradients = tape.gradient(
            class_score,
            conv_output
        )

        if gradients is None:
            return None, "Gradients could not be calculated."

        # Global average pooling of gradients
        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2)
        )

        conv_output = conv_output[0]

        heatmap = tf.reduce_sum(
            conv_output *
            pooled_gradients,
            axis=-1
        )

        # ReLU
        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        if float(max_value.numpy()) <= 0:
            return None, "Grad-CAM heatmap contains no positive activation."

        heatmap = heatmap / max_value

        return heatmap.numpy(), None

    except Exception as e:

        return None, str(e)


# ============================================================
# SIMPLE HEATMAP COLORING
# ============================================================

def create_heatmap_image(heatmap):

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    h = heatmap.shape[0]
    w = heatmap.shape[1]

    # Simple blue -> cyan -> yellow -> red style mapping
    r = np.clip(
        2.0 * heatmap - 0.5,
        0,
        1
    )

    g = np.clip(
        2.0 * (1.0 - np.abs(heatmap - 0.5) * 2),
        0,
        1
    )

    b = np.clip(
        1.5 - 2.0 * heatmap,
        0,
        1
    )

    rgb = np.stack(
        [r, g, b],
        axis=-1
    )

    rgb = np.uint8(
        rgb * 255
    )

    return Image.fromarray(
        rgb
    ).resize(
        (w, h)
    )


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image,
    heatmap,
    alpha=0.45
):

    original = original_image.convert(
        "RGB"
    ).resize(
        IMG_SIZE
    )

    heatmap_img = create_heatmap_image(
        heatmap
    )

    heatmap_img = heatmap_img.resize(
        original.size
    )

    overlay = Image.blend(
        original,
        heatmap_img,
        alpha
    )

    return overlay


# ============================================================
# DISPLAY NAME
# ============================================================

def clean_class_name(name):

    return name.split(
        "___"
    )[-1].replace(
        "_",
        " "
    ).replace(
        "  ",
        " "
    )


# ============================================================
# MODEL PERFORMANCE INFORMATION
# ============================================================

MODEL_COMPARISON = {
    "Custom CNN": {
        "Architecture": "Lightweight Custom CNN",
        "Input": "224 × 224 × 3",
        "Purpose": "Lightweight disease classification"
    },

    "ResNet50": {
        "Architecture": "Transfer Learning",
        "Input": "224 × 224 × 3",
        "Purpose": "Deep residual feature extraction"
    },

    "MobileNetV2": {
        "Architecture": "Transfer Learning",
        "Input": "224 × 224 × 3",
        "Purpose": "Lightweight mobile-oriented classification"
    }
}


# ============================================================
# HEADER
# ============================================================

st.title(
    "🥔 Potato Plant Disease Detection System"
)

st.markdown(
    """
### AI-Based Plant Disease Analysis

Upload **one potato leaf image** from the Home page.
The same uploaded image is then used for:

- Disease Detection
- Explainable AI / Grad-CAM
- Disease Information
- Cause & Symptoms
- Management Recommendations
- Model Analysis
"""
)

# ============================================================
# MODEL STATUS
# ============================================================

if model is None:

    st.error(
        "❌ Model could not be loaded."
    )

    st.code(
        MODEL_PATH
    )

    st.info(
        "Make sure explainable_cnn_model.keras "
        "is in the same GitHub folder as app.py."
    )

    st.stop()

else:

    st.success(
        "✅ Explainable CNN model loaded successfully."
    )

    if last_conv_layer is not None:

        st.caption(
            f"Grad-CAM layer: {last_conv_layer.name}"
        )


# ============================================================
# HOME PAGE UPLOAD
# ============================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Disease Detection",
        "Explainable AI",
        "Disease Information",
        "Cause & Solution",
        "Batch Analysis",
        "Model Comparison",
        "Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload the image once from Home. "
    "The same image is reused throughout the app."
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.header(
        "🏠 Upload Potato Leaf"
    )

    uploaded_file = st.file_uploader(
        "Upload a potato leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="main_upload"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.session_state[
            "uploaded_image"
        ] = image

        st.session_state[
            "uploaded_filename"
        ] = uploaded_file.name

        st.success(
            "Image uploaded successfully!"
        )

        st.image(
            image,
            caption="Uploaded Potato Leaf",
            use_container_width=True
        )

        if st.button(
            "🔍 Analyze Leaf",
            type="primary"
        ):

            with st.spinner(
                "Analyzing the leaf..."
            ):

                processed = preprocess_image(
                    image
                )

                predictions = model.predict(
                    processed,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(predictions)
                )

                predicted_class = CLASS_NAMES[
                    predicted_index
                ]

                confidence = float(
                    predictions[
                        predicted_index
                    ]
                )

                st.session_state[
                    "predictions"
                ] = predictions

                st.session_state[
                    "predicted_class"
                ] = predicted_class

                st.session_state[
                    "confidence"
                ] = confidence

            st.success(
                "Analysis completed!"
            )

            st.metric(
                "Prediction",
                clean_class_name(
                    predicted_class
                )
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            if predicted_class == "Potato___healthy":

                st.success(
                    "🌱 The model detected a healthy potato leaf."
                )

            elif predicted_class.startswith(
                "Potato___"
            ):

                st.warning(
                    "⚠️ A potato disease pattern was detected."
                )

            else:

                st.info(
                    "The model predicted a non-potato PlantVillage class. "
                    "Please upload a potato leaf image."
                )

    else:

        st.info(
            "👆 Upload a potato leaf image to start the analysis."
        )


# ============================================================
# CHECK WHETHER IMAGE EXISTS
# ============================================================

has_image = (
    "uploaded_image" in st.session_state
)

if page != "Home" and not has_image:

    st.warning(
        "⚠️ Please upload and analyze an image from the Home page first."
    )

    st.stop()


# ============================================================
# GET STORED IMAGE / PREDICTION
# ============================================================

if has_image:

    image = st.session_state[
        "uploaded_image"
    ]

    processed_image = preprocess_image(
        image
    )

    # Recalculate prediction every page
    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[
            predicted_index
        ]
    )


# ============================================================
# DISEASE DETECTION
# ============================================================

if page == "Disease Detection":

    st.header(
        "🔍 Disease Detection"
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.image(
            image,
            caption="Analyzed Leaf",
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Prediction Result"
        )

        st.success(
            clean_class_name(
                predicted_class
            )
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.progress(
            min(
                max(confidence, 0.0),
                1.0
            )
        )

    st.divider()

    st.subheader(
        "📊 Class Probabilities"
    )

    # Show top 5 predictions
    top_indices = np.argsort(
        predictions
    )[::-1][:5]

    for idx in top_indices:

        probability = float(
            predictions[idx]
        )

        st.write(
            f"**{clean_class_name(CLASS_NAMES[idx])}** — "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            min(
                max(probability, 0.0),
                1.0
            )
        )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "Explainable AI":

    st.header(
        "🧠 Explainable AI — Grad-CAM"
    )

    st.markdown(
        """
Grad-CAM highlights the image regions that contributed
to the CNN prediction.
"""
    )

    st.image(
        image,
        caption="Original Image",
        use_container_width=True
    )

    if last_conv_layer is None:

        st.error(
            "No convolutional layer was found in the model."
        )

    else:

        with st.spinner(
            "Generating Grad-CAM explanation..."
        ):

            heatmap, grad_error = generate_gradcam(
                processed_image,
                model,
                last_conv_layer
            )

        if heatmap is not None:

            overlay = create_gradcam_overlay(
                image,
                heatmap
            )

            col1, col2 = st.columns(
                2
            )

            with col1:

                st.image(
                    create_heatmap_image(
                        heatmap
                    ),
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )

            with col2:

                st.image(
                    overlay,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

            st.success(
                "✅ Grad-CAM generated successfully."
            )

            st.info(
                f"Prediction: {clean_class_name(predicted_class)} "
                f"({confidence * 100:.2f}%)"
            )

        else:

            st.error(
                "❌ Grad-CAM could not be generated."
            )

            st.code(
                grad_error
            )

            st.info(
                f"Detected convolutional layer: "
                f"{last_conv_layer.name}"
            )


# ============================================================
# DISEASE INFORMATION
# ============================================================

elif page == "Disease Information":

    st.header(
        "🦠 Disease Information"
    )

    info = get_disease_info(
        predicted_class
    )

    st.subheader(
        info["name"]
    )

    st.write(
        f"**Model Confidence:** "
        f"{confidence * 100:.2f}%"
    )

    st.markdown(
        "### Cause"
    )

    st.write(
        info["cause"]
    )

    st.markdown(
        "### Symptoms"
    )

    for symptom in info["symptoms"]:

        st.write(
            f"• {symptom}"
        )


# ============================================================
# CAUSE & SOLUTION
# ============================================================

elif page == "Cause & Solution":

    st.header(
        "🌱 Why Did This Problem Occur?"
    )

    info = get_disease_info(
        predicted_class
    )

    st.subheader(
        "Possible Cause"
    )

    st.write(
        info["cause"]
    )

    st.subheader(
        "⚡ Recommended Management"
    )

    for item in info["management"]:

        st.write(
            f"✅ {item}"
        )

    st.subheader(
        "🛡️ Prevention"
    )

    for item in info["prevention"]:

        st.write(
            f"• {item}"
        )

    st.warning(
        "These recommendations are general educational guidance. "
        "For severe crop damage, confirm the diagnosis with "
        "an agricultural expert."
    )


# ============================================================
# BATCH ANALYSIS
# ============================================================

elif page == "Batch Analysis":

    st.header(
        "📁 Batch Image Analysis"
    )

    files = st.file_uploader(
        "Upload multiple leaf images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
    )

    if files:

        results = []

        progress = st.progress(
            0
        )

        total = len(files)

        for i, file in enumerate(files):

            try:

                batch_image = Image.open(
                    file
                ).convert("RGB")

                batch_input = preprocess_image(
                    batch_image
                )

                batch_prediction = model.predict(
                    batch_input,
                    verbose=0
                )[0]

                idx = int(
                    np.argmax(
                        batch_prediction
                    )
                )

                pred = CLASS_NAMES[
                    idx
                ]

                conf = float(
                    batch_prediction[
                        idx
                    ]
                )

                results.append(
                    {
                        "Image": file.name,
                        "Prediction": clean_class_name(pred),
                        "Confidence": f"{conf * 100:.2f}%"
                    }
                )

            except Exception as e:

                results.append(
                    {
                        "Image": file.name,
                        "Prediction": "Error",
                        "Confidence": str(e)
                    }
                )

            progress.progress(
                (i + 1) / total
            )

        st.dataframe(
            results,
            use_container_width=True
        )

        st.success(
            f"Analyzed {total} images."
        )

    else:

        st.info(
            "Upload multiple images to perform batch analysis."
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header(
        "⚖️ Model Comparison"
    )

    st.markdown(
        """
The project compares the lightweight Custom CNN
with transfer-learning architectures.
"""
    )

    st.table(
        {
            "Model": [
                "Custom CNN",
                "ResNet50",
                "MobileNetV2"
            ],
            "Architecture": [
                "Lightweight CNN",
                "Transfer Learning",
                "Transfer Learning"
            ],
            "Input Size": [
                "224×224×3",
                "224×224×3",
                "224×224×3"
            ],
            "Main Goal": [
                "Lightweight deployment",
                "High-level feature extraction",
                "Efficient mobile deployment"
            ]
        }
    )

    st.info(
        "For exact Accuracy, Loss, Parameters, Precision, "
        "Recall and F1 values, enter the values obtained "
        "from your Kaggle evaluation."
    )


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "Performance":

    st.header(
        "📈 Model Performance"
    )

    st.subheader(
        "Current Prediction"
    )

    st.metric(
        "Predicted Class",
        clean_class_name(
            predicted_class
        )
    )

    st.metric(
        "Prediction Confidence",
        f"{confidence * 100:.2f}%"
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

    st.warning(
        "These evaluation metrics cannot be calculated "
        "from a single uploaded image. They must be calculated "
        "using the original test dataset and true labels."
    )

    st.info(
        "Your Kaggle test results should be entered here "
        "if you want the final Streamlit app to display "
        "the exact research evaluation metrics."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Potato Plant Disease Detection | "
    "Explainable CNN | Grad-CAM | Streamlit"
)
