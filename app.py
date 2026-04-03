import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# ================= CONFIG =================
MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 224
CONF_THRESHOLD = 0.5

# ================= LOAD =================
@st.cache_data
def load_classes():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

class_names = load_classes()

@st.cache_resource
def load_model(num_classes):
    # Rebuild architecture to bypass Keras 3 deserialization shape inference bugs
    base_model = tf.keras.applications.EfficientNetB3(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.load_weights(MODEL_PATH)
    return model

model = load_model(len(class_names))

# ================= PREPROCESS =================
def preprocess_image(img):
    # Force RGB (fix grayscale issue)
    img = img.convert("RGB")

    # Resize EXACTLY to training size
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array
    img = np.array(img)

    # Safety check: ensure 3 channels
    if len(img.shape) == 2 or img.shape[-1] != 3:
        img = np.stack((img,) * 3, axis=-1)

    # Preprocess (EfficientNet)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

# ================= UI =================
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect disease")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_array = preprocess_image(image)

        # Debug (optional)
        st.write("Image shape:", img_array.shape)

        # Predict
        preds = model.predict(img_array)[0]

        # Top 3 predictions
        top_indices = preds.argsort()[-3:][::-1]

        st.subheader("🔍 Top Predictions:")
        for i in top_indices:
            st.write(f"{class_names[i]} : {preds[i]*100:.2f}%")

        # Final prediction with confidence check
        confidence = preds[top_indices[0]]
        final_class = class_names[top_indices[0]]

        st.subheader("🌱 Final Result:")

        if confidence < CONF_THRESHOLD:
            st.warning("⚠️ Unable to confidently detect disease")
        else:
            st.success(f"✅ {final_class} ({confidence*100:.2f}%)")

    except Exception as e:
        st.error(f"⚠️ Error loading image: The file might be hidden, corrupt, or unsupported. Please upload a real image. Details: {e}")