import streamlit as st
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
from PIL import Image

# -----------------------------
# 1. Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Image Classifier", layout="centered")

# -----------------------------
# 2. Title
# -----------------------------
st.title("🧠 Image Classification App (VGG16)")

# -----------------------------
# 3. Load Pretrained Model
# -----------------------------
@st.cache_resource
def load_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(3, activation='softmax')  # Example: 3 output classes (edit as needed)
    ])
    return model

model = load_model()

# -----------------------------
# 4. Image Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# -----------------------------
# 5. Prediction
# -----------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("")

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalization

    # Predict
    st.write("🔍 Classifying...")
    preds = model.predict(img_array)

    # Display prediction results
    st.write("### Prediction Probabilities:")
    st.write(preds)

    class_names = ["Class 1", "Class 2", "Class 3"]  # <-- Edit these names
    predicted_class = class_names[np.argmax(preds)]
    st.success(f"✅ Predicted: **{predicted_class}**")

# -----------------------------
# 6. Footer
# -----------------------------
st.markdown("---")
st.caption("Created with ❤️ using Streamlit and TensorFlow")
