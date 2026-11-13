# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(

    page_title="Healthcare ANN",

    page_icon="🏥",

    layout="centered"

)


# ==========================================
# 3. APPLICATION TITLE
# ==========================================

st.title("🏥 Healthcare Chest X-Ray Classification")

st.subheader("Artificial Neural Network (ANN)")

st.write(
    "Upload a chest X-ray image to see the "
    "model's educational classification prediction."
)


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource

def load_model():

    model = tf.keras.models.load_model(
        "model/ann_chest_xray.keras"
    )

    return model


try:

    model = load_model()

    st.success("ANN Model Loaded Successfully!")

except Exception as e:

    st.error(f"Model loading failed: {e}")

    st.stop()


# ==========================================
# 5. IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(

    "Upload Chest X-Ray Image",

    type=["jpg", "jpeg", "png"]

)


# ==========================================
# 6. IMAGE PREPROCESSING
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    image = image.resize((128, 128))

    st.image(

        image,

        caption="Uploaded Chest X-Ray",

        width=300

    )


    # Convert image into NumPy array

    img_array = np.array(image)

    # Add batch dimension and channel dimension

    img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)


    # ======================================
    # 7. MAKE PREDICTION
    # ======================================

    if st.button("Predict"):

        prediction = model.predict(img_array, verbose=0)

        probability = float(prediction[0][0])


        # ==================================
        # 8. DISPLAY RESULTS
        # ==================================

        st.subheader("Prediction Result")

        st.metric(

            "Estimated PNEUMONIA probability",

            f"{probability * 100:.2f}%"

        )


        if probability >= 0.5:

            st.error("Model Prediction: PNEUMONIA")

        else:

            st.success("Model Prediction: NORMAL")


        st.info(
            "This is an educational ANN prediction, "
            "not a medical diagnosis. Consult a qualified "
            "healthcare professional for medical evaluation."
        )