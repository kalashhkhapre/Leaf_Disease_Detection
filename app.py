import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array 
from gtts import gTTS
import os
import random
import time

# Load your trained model
MODEL_PATH = "plant_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# All class names from your final model
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus'
]

# Eco-friendly treatment suggestions
TREATMENTS = {
    "Pepper__bell___Bacterial_spot": "Use copper-based sprays. Remove infected leaves. Avoid overhead watering.",
    "Pepper__bell___healthy": "Healthy pepper! Keep an eye for changes and water regularly.",
    "Potato___Early_blight": "Use neem oil spray and remove affected leaves. Rotate crops.",
    "Potato___healthy": "Potato plant looks great! Keep soil moist but not waterlogged.",
    "Potato___Late_blight": "Use organic fungicide. Remove severely infected plants to stop spread.",
    "Tomato_Bacterial_spot": "Avoid working with wet plants. Use copper-based organic treatments.",
    "Tomato_Early_blight": "Mulch around base. Apply compost tea or neem oil.",
    "Tomato_healthy": "Tomato looking juicy and clean! Maintain proper airflow and pruning.",
    "Tomato_Late_blight": "Use certified seeds. Apply Bacillus subtilis or other biocontrol sprays.",
    "Tomato_Leaf_Mold": "Ensure good ventilation. Apply sulfur dust as a preventive.",
    "Tomato_Septoria_leaf_spot": "Remove lower infected leaves. Use potassium bicarbonate spray.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray with insecticidal soap or garlic-pepper tea.",
    "Tomato__Target_Spot": "Apply organic copper fungicide and prune lower leaves.",
    "Tomato__Tomato_mosaic_virus": "No cure, but remove infected plants. Disinfect tools.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies naturally with sticky traps or neem oil."
}

# Green tip of the day suggestions
SUSTAINABLE_TIPS = [
    "Practice crop rotation to prevent soil nutrient depletion.",
    "Use compost instead of chemical fertilizers.",
    "Collect rainwater for irrigation.",
    "Plant native crops that require less water.",
    "Use companion planting to deter pests naturally."
]

# Set up Streamlit UI
st.set_page_config(page_title="🌿 CropDoctor", layout="centered")
st.title("🌿 CropDoctor")
st.markdown("**Your AI-powered eco-friendly plant disease detector.**")
st.markdown("Upload a leaf image and get instant diagnosis + green treatment tips. 🌱")

# File uploader
uploaded_file = st.file_uploader("📷 Upload a plant leaf image", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    # Display the uploaded image
    pil_image = Image.open(uploaded_file).convert('RGB')
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model
    img = pil_image.resize((128, 128))  # ✅ Resize the image manually
    img_array = img_to_array(img)       # ✅ Use the correct utility function
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0       # ✅ Normalize

    # Now img_array is ready for prediction


    with st.spinner("🔍 Analyzing leaf..."):
        preds = model.predict(img_array)
        score = tf.nn.softmax(preds[0])
        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)
        time.sleep(2)

    st.success("✅ Diagnosis complete!")
    st.balloons()

    # 🧼 Clean label formatting
    clean_label = predicted_class.replace("_", " ").replace("  ", " ")
    st.markdown(f"### 🦠 Predicted Disease: `{clean_label}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    # Show eco-friendly treatment
    treatment = TREATMENTS.get(predicted_class, "No treatment info available.")
    st.markdown("### 🌱 Eco-Friendly Treatment")
    st.markdown(f"`{treatment}`")

    # Voice suggestion using gTTS
    if st.button("🔊 Hear Advice"):
        tts = gTTS(text=treatment, lang='en')
        tts.save("treatment.mp3")
        st.audio("treatment.mp3", format="audio/mp3")

    # Green farming tip of the day
    tip = random.choice(SUSTAINABLE_TIPS)
    st.markdown("---")
    st.markdown("💡 **Sustainable Farming Tip of the Day**")
    st.info(tip)
