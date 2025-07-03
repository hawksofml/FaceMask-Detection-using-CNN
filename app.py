import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mask_detector_final.keras')
    return model

model = load_model()
class_names = ['Mask', 'No Mask']

# Sidebar
st.sidebar.title("🔧 Settings")
image_size = st.sidebar.selectbox("Image Resize Size", [224, 128, 96], index=0)

# Main Title
st.title("😷 Face Mask Detection App")
st.markdown("Upload a face image to detect whether the person is wearing a **mask** or **no mask**.")

# File Uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)
    
    # Progress feedback
    with st.spinner("🔍 Analyzing Image..."):
        time.sleep(1)  # Simulated delay for realism

        # Preprocess image
        img = image.resize((image_size, image_size))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        result_label = class_names[class_index]

        # Color-coded output
        if result_label == 'Mask':
            st.success(f"✅ **Prediction:** {result_label} ({confidence:.2f}%)")
        else:
            st.error(f"❌ **Prediction:** {result_label} ({confidence:.2f}%)")

        # Details
        st.markdown("📊 **Prediction Probabilities:**")
        st.write(f"- Mask: {prediction[0][0]*100:.2f}%")
        st.write(f"- No Mask: {prediction[0][1]*100:.2f}%")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and CNN")

