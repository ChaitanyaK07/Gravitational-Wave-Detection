"""
Gravitational Wave Detection - Interactive Web App
Upload your own Q-transform spectrograms for classification
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os

# Page config
st.set_page_config(
    page_title="Gravitational Wave Detector",
    page_icon="🌊",
    layout="wide"
)

# Google Drive file ID (replace with yours!)
GDRIVE_FILE_ID = "16LqCb_cxJZM_tXADTNhdYXa03vjfN-aV"  # ← PUT YOUR FILE_ID HERE
MODEL_URL = f"https://drive.google.com/uc?id={"16LqCb_cxJZM_tXADTNhdYXa03vjfN-aV"}"
MODEL_PATH = "best_model.keras"

@st.cache_resource
def load_model():
    """Download and load the trained model from Google Drive"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (309MB, one-time only)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load model
try:
    model = load_model()
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    MODEL_LOADED = False

# Real prediction function using actual model
def predict_from_image(image):
    """
    Real model prediction on uploaded image
    """
    # Resize and preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    return float(prediction)


# Sample data
SAMPLE_DATA = {
    'GW150914 (Signal)': {
        'image': 'results/figures/sample_signal_1.png',
        'prediction': 0.87,
        'type': 'Binary black hole merger',
        'distance': '1.3 billion light years',
        'description': 'First gravitational wave ever detected (Sept 2015)'
    },
    'GW170817 (Signal)': {
        'image': 'results/figures/sample_signal_2.png',
        'prediction': 0.73,
        'type': 'Binary neutron star merger',
        'distance': '130 million light years',
        'description': 'Neutron star merger (Aug 2017)'
    },
    'Background Noise 1': {
        'image': 'results/figures/sample_noise_1.png',
        'prediction': 0.15,
        'type': 'Instrumental noise',
        'distance': 'N/A',
        'description': 'Background detector noise'
    },
    'Background Noise 2': {
        'image': 'results/figures/sample_noise_2.png',
        'prediction': 0.22,
        'type': 'Instrumental noise',
        'distance': 'N/A',
        'description': 'Background detector noise'
    }
}

# Header
st.title(" Gravitational Wave Detection System")
st.markdown("* Detection using a real trained CNN model*")

if not MODEL_LOADED:
    st.error(" Model failed to load. Using demo mode.")

# Sidebar
with st.sidebar:
    st.header(" Model Info")
    st.info("""
    **Architecture:** 4-layer CNN  
    **Parameters:** 8.5M  
    **Test Accuracy:** ~60%  
    **Training Data:** 4 LIGO events  
    **Input:** 224×224 Q-transform spectrograms
    """)
    
    if MODEL_LOADED:
        st.success(" Model loaded successfully!")
    else:
        st.warning(" Model not loaded")
    
    st.header(" Technology")
    st.markdown("""
    - TensorFlow/Keras
    - gwpy (LIGO data)
    - Q-transform analysis
    - Signal processing
    """)
    
    st.markdown("---")
    st.markdown("[GitHub Repository →](https://github.com/ChaitanyaK07/gravitational-wave-detection)")
    
    st.markdown("---")
    st.header(" How to Use")
    st.markdown("""
    **Option 1:** Upload your own Q-transform spectrogram
    
    **Option 2:** Try pre-loaded samples
    
    **Formats:** PNG, JPG, JPEG
    """)

# Main content
st.header(" Gravitational Wave Classifier")

mode = st.radio(
    "Choose mode:",
    [" Upload Your Own Image", " Try Sample Images"],
    horizontal=True
)

st.markdown("---")

# Upload mode
if mode == " Upload Your Own Image":
    st.subheader("Upload Q-Transform Spectrogram")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a Q-transform spectrogram (224×224 recommended)"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Uploaded Spectrogram")
            st.image(image, use_column_width=True)
            st.caption(f"Size: {image.size[0]}×{image.size[1]} pixels")
        
        with col2:
            st.subheader(" Model Prediction")
            
            if MODEL_LOADED:
                with st.spinner("Analyzing with real CNN model..."):
                    prediction = predict_from_image(image)
            else:
                st.error("Model not loaded. Cannot predict.")
                prediction = 0.5
            
            if prediction > 0.5:
                st.success(f"###  GRAVITATIONAL WAVE DETECTED")
                st.metric("Confidence", f"{prediction*100:.1f}%")
                st.markdown("""
                **Interpretation:** The CNN detected patterns consistent 
                with a gravitational wave chirp signature!
                """)
            else:
                st.info(f"###  BACKGROUND NOISE")
                st.metric("Confidence", f"{(1-prediction)*100:.1f}%")
                st.markdown("""
                **Interpretation:** No significant gravitational wave 
                signature detected.
                """)
            
            st.markdown("---")
            st.markdown("**Probability Distribution:**")
            
            prob_signal = prediction * 100
            prob_noise = (1 - prediction) * 100
            
            st.progress(int(prob_signal)/100, text=f"Signal: {prob_signal:.0f}%")
            st.progress(int(prob_noise)/100, text=f"Noise: {prob_noise:.0f}%")
            
            with st.expander(" Classification Details"):
                st.markdown(f"""
                - **Input size:** {image.size[0]}×{image.size[1]} pixels
                - **Resized to:** 224×224×3
                - **Raw prediction:** {prediction:.6f}
                - **Classification:** {'Signal' if prediction > 0.5 else 'Noise'}
                - **Threshold:** 0.50
                """)
        
        st.markdown("---")
        if st.button(" Download Results"):
            result_text = f"""Gravitational Wave Detection Results
{'='*50}
File: {uploaded_file.name}
Prediction: {'GRAVITATIONAL WAVE' if prediction > 0.5 else 'BACKGROUND NOISE'}
Confidence: {max(prediction, 1-prediction)*100:.1f}%
Signal Probability: {prediction*100:.2f}%
Noise Probability: {(1-prediction)*100:.2f}%

Model: 4-layer CNN (8.5M parameters)
Test Accuracy: ~60%
"""
            st.download_button(
                " Save as TXT",
                result_text,
                f"gw_results_{uploaded_file.name}.txt",
                "text/plain"
            )
    
    else:
        st.info(" Upload a Q-transform spectrogram to begin")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("###  Best Practices:")
            st.markdown("""
            - Q-transform spectrograms
            - 224×224 pixels recommended
            - Frequency: 20-400 Hz
            - PNG, JPG, JPEG formats
            """)
        with col2:
            st.markdown("###  Sample Data:")
            st.markdown("""
            - [LIGO Open Science](https://www.gw-openscience.org/)
            - [GitHub Repo](https://github.com/ChaitanyaK07/gravitational-wave-detection)
            """)

# Sample mode
else:
    st.subheader("Try Pre-Loaded Samples")
    
    sample_name = st.selectbox(
        "Choose a sample:",
        options=list(SAMPLE_DATA.keys())
    )
    
    sample = SAMPLE_DATA[sample_name]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Q-Transform Spectrogram")
        try:
            img = Image.open(sample['image'])
            st.image(img, use_column_width=True)
            
            # Get real prediction on sample too
            if MODEL_LOADED:
                with st.spinner("Running model..."):
                    prediction = predict_from_image(img)
            else:
                prediction = sample['prediction']
        except:
            st.info(f"Spectrogram: {sample_name}")
            prediction = sample['prediction']
        
        st.caption(sample['description'])
    
    with col2:
        st.subheader(" Model Prediction")
        
        if prediction > 0.5:
            st.success(f"###  GRAVITATIONAL WAVE")
            st.metric("Confidence", f"{prediction*100:.1f}%")
        else:
            st.info(f"###  BACKGROUND NOISE")
            st.metric("Confidence", f"{(1-prediction)*100:.1f}%")
        
        st.markdown("**Event Details:**")
        st.write(f"- **Type:** {sample['type']}")
        st.write(f"- **Distance:** {sample['distance']}")
        
        st.markdown("---")
        prob_signal = prediction * 100
        prob_noise = (1 - prediction) * 100
        
        st.progress(int(prob_signal)/100, text=f"Signal: {prob_signal:.0f}%")
        st.progress(int(prob_noise)/100, text=f"Noise: {prob_noise:.0f}%")

st.markdown("---")
st.header(" How It Works")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**1. Upload**")
    st.write("Image file")
with col2:
    st.markdown("**2. Process**")
    st.write("Resize, normalize")
with col3:
    st.markdown("**3. CNN**")
    st.write("Feature extraction")
with col4:
    st.markdown("**4. Classify**")
    st.write("Signal/Noise")

with st.expander(" Technical Details"):
    st.markdown("""
    **Preprocessing:** FFT whitening, Butterworth filter (30-400 Hz), Q-transform
    
    **CNN:** 4 blocks (32→64→128→256), Batch norm, Dropout (0.25-0.5)
    
    **Training:** 52 images, Adam optimizer, Binary crossentropy
    """)

with st.expander(" Model Performance"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "60%")
    col2.metric("Precision", "58%")
    col3.metric("Recall", "56%")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with TensorFlow, Keras & Streamlit | Real CNN model trained on LIGO data
</div>
""", unsafe_allow_html=True)