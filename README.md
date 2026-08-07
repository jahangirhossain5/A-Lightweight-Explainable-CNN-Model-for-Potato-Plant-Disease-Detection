# 🌿 Lightweight CNN for Plant Disease Detection: Transfer Learning and XAI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://plant-disease-detection-ogc4xelrfml9zsepjmgor5.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

## 📜 Overview
This project presents a Deep Learning-based solution for the automated detection of plant diseases. We developed a **Custom Lightweight CNN** and compared its performance against industry-standard Transfer Learning models (**ResNet50** and **MobileNetV2**) on the **PlantVillage dataset** (38 classes).

To ensure trust and transparency, we integrated **Explainable AI (Grad-CAM)** to visualize the decision-making process. The final model is deployed as a user-friendly **Streamlit Web Application**.

## 🚀 Live Demo
Check out the live web application here:
👉 **[Plant Disease Detection App](https://plant-disease-detection-ogc4xelrfml9zsepjmgor5.streamlit.app/)**

## ✨ Key Features
* **Multi-Class Classification:** Detects 38 different disease classes across 14 crop species.
* **Lightweight Architecture:** Proposed Custom CNN has only **0.65M parameters** (vs. 24M in ResNet50), making it suitable for mobile/edge deployment.
* **High Accuracy:** Achieved **98.71% test accuracy** with the Custom CNN.
* **Explainable AI (XAI):** Uses **Grad-CAM** heatmaps to highlight infected leaf regions, verifying the model's focus.
* **Interactive Web App:** Built with Streamlit for real-time inference.

## 📊 Model Comparison

| Model | Test Accuracy | Test Loss | Parameters | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| **Custom CNN** | **98.71%** | **0.7809** | **~0.65 M** | **Best Trade-off (Lightweight & Accurate)** |
| ResNet50 | 98.67% | 0.7833 | ~24 M | High computational cost |
| MobileNetV2 | 96.19% | 0.8603 | ~2.6 M | Standard lightweight baseline |

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** NumPy, Pandas, OpenCV
* **Visualization:** Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Cloud

## 📂 Dataset
The dataset used is the **PlantVillage Dataset**, originally collected by Pennsylvania State University.
* **Total Images:** 54,305
* **Classes:** 38 (Disease + Healthy)
* **Augmentation:** Rotation, Zoom, Flip (On-the-fly)

## ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/coderzaman/Plant-Disease-Detection-Streamlit-XAI.git
    cd your-repo-name
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App Locally**
    ```bash
    streamlit run app.py
    ```

## 🧠 Explainable AI (Grad-CAM)
We used **Grad-CAM** to validate our model. The heatmaps below show that the model correctly focuses on the **lesions and spots** on the leaves rather than the background.

*<img width="1452" height="1489" alt="download" src="https://github.com/user-attachments/assets/cb2f56ce-018c-4817-940e-52041418ba5e" />*


## 👥 Contributors
This project was developed by:
* **Aktaruzzaman** (Team Leader & Lead Developer)
* **Sagar Saha** (Data Engineer)
* **Md. Abdullah Al Mamun** (AI Analyst)
* **Md. Najmul Parves** (Technical Writer)
* **Siam Hossain** (Visual Designer)

---
*Developed for University Thesis / Final Year Project.*
