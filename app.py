"""
Gravitational Wave Detection - Interactive Web App
Upload your own Q-transform spectrograms for classification using a REAL trained CNN
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
    page_icon="",
    layout="wide"
)

# Google Drive Model Configuration
GDRIVE_FILE_ID = "16LqCb_cxJZM_tXADTNhdYXa03vjfN-aV"
MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
MODEL_PATH = "best_model.keras"

@st.cache_resource
def load_model():
    """Download and load the trained model from Google Drive"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner(" Downloading trained model (309MB, one-time only)..."):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Load failed: {e}")
        return None

# Load model
model = load_model()
MODEL_LOADED = model is not None

def predict_from_image(image):
    """
    Real CNN prediction with proper preprocessing
    """
    try:
        # Convert to RGB if needed (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to exact model input size
        img_resized = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Normalize to [0, 1] range
        img_array = img_array / 255.0
        
        # Ensure shape is (224, 224, 3)
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Unexpected shape after preprocessing: {img_array.shape}")
        
        # Add batch dimension: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)
        
        # Extract scalar value
        return float(prediction[0][0])
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.5  # Return neutral prediction on error


# Sample data
SAMPLE_DATA = {
    'GW150914 (Signal)': {
        'image': 'results/figures/sample_signal_1.png',
        'type': 'Binary black hole merger',
        'distance': '1.3 billion light years',
        'description': 'First gravitational wave ever detected (Sept 14, 2015)'
    },
    'GW170817 (Signal)': {
        'image': 'results/figures/sample_signal_2.png',
        'type': 'Binary neutron star merger',
        'distance': '130 million light years',
        'description': 'Neutron star merger (Aug 17, 2017)'
    },
    'Background Noise 1': {
        'image': 'results/figures/sample_noise_1.png',
        'type': 'Instrumental noise',
        'distance': 'N/A',
        'description': 'Background detector noise'
    },
    'Background Noise 2': {
        'image': 'results/figures/sample_noise_2.png',
        'type': 'Instrumental noise',
        'distance': 'N/A',
        'description': 'Background detector noise'
    }
}

# Header
st.title(" Gravitational Wave Detection System")
st.markdown("*Real-time classification using a trained CNN model*")

if MODEL_LOADED:
    st.success(" Model loaded and ready!")
else:
    st.error(" Model not loaded - using demo mode")

# Sidebar
with st.sidebar:
    st.header(" Model Info")
    st.info("""
    **Architecture:** 4-layer CNN  
    **Parameters:** 8.5M  
    **Test Accuracy:** ~60%  
    **Training Data:** 4 LIGO events  
    **Input:** 224×224×3
    """)
    
    if MODEL_LOADED:
        st.success(" Status: Active")
    else:
        st.warning(" Status: Demo Mode")
    
    st.markdown("---")
    st.header(" Tech Stack")
    st.markdown("""
    - TensorFlow/Keras
    - gwpy (LIGO data)
    - Q-transform analysis
    - Streamlit
    """)
    
    st.markdown("---")
    st.markdown("[GitHub →](https://github.com/ChaitanyaK07/gravitational-wave-detection)")

# Main content
st.header(" Gravitational Wave Classifier")

mode = st.radio(
    "Choose mode:",
    [" Upload Your Own Image", " Try Sample Images"],
    horizontal=True
)

st.markdown("---")

# UPLOAD MODE
if mode == " Upload Your Own Image":
    st.subheader("Upload Q-Transform Spectrogram")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a Q-transform spectrogram"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Uploaded Spectrogram")
            st.image(image, use_column_width=True)
            st.caption(f"Size: {image.size[0]}×{image.size[1]} | Mode: {image.mode}")
        
        with col2:
            st.subheader(" Prediction")
            
            if MODEL_LOADED:
                with st.spinner("Analyzing..."):
                    prediction = predict_from_image(image)
                
                if prediction > 0.5:
                    st.success("###  GRAVITATIONAL WAVE")
                    st.metric("Confidence", f"{prediction*100:.1f}%")
                    st.markdown("The CNN detected gravitational wave patterns!")
                else:
                    st.info("###  BACKGROUND NOISE")
                    st.metric("Confidence", f"{(1-prediction)*100:.1f}%")
                    st.markdown("No gravitational wave signature detected.")
                
                st.markdown("---")
                prob_signal = prediction * 100
                prob_noise = (1 - prediction) * 100
                
                col_a, col_b = st.columns(2)
                col_a.metric("Signal", f"{prob_signal:.1f}%")
                col_b.metric("Noise", f"{prob_noise:.1f}%")
                
                st.progress(prob_signal/100)
                
                with st.expander(" Details"):
                    st.write(f"- Input: {image.size[0]}×{image.size[1]}, {image.mode}")
                    st.write(f"- Resized: 224×224×3")
                    st.write(f"- Prediction: {prediction:.6f}")
                    st.write(f"- Threshold: 0.5")
            else:
                st.error("Model not loaded")
        
        st.markdown("---")
        if st.button(" Download Results"):
            result_text = f"""Gravitational Wave Detection Results
{'='*50}
File: {uploaded_file.name}
Prediction: {'GRAVITATIONAL WAVE' if prediction > 0.5 else 'BACKGROUND NOISE'}
Confidence: {max(prediction, 1-prediction)*100:.1f}%
Signal: {prediction*100:.2f}%
Noise: {(1-prediction)*100:.2f}%

Model: 4-layer CNN (8.5M parameters)
"""
            st.download_button(
                " Save",
                result_text,
                f"gw_result_{uploaded_file.name}.txt",
                "text/plain"
            )
    else:
        st.info(" Upload a spectrogram to begin")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("###  Tips:")
            st.markdown("""
            - Q-transform spectrograms
            - 224×224 recommended
            - PNG, JPG, JPEG
            """)
        with col2:
            st.markdown("###  Get Data:")
            st.markdown("""
            - [LIGO Open Science](https://www.gw-openscience.org/)
            - [GitHub Repo](https://github.com/ChaitanyaK07/gravitational-wave-detection)
            """)

# SAMPLE MODE
else:
    st.subheader("Try Pre-Loaded Samples")
    
    sample_name = st.selectbox(
        "Choose a sample:",
        options=list(SAMPLE_DATA.keys())
    )
    
    sample = SAMPLE_DATA[sample_name]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Spectrogram")
        try:
            img = Image.open(sample['image'])
            st.image(img, use_column_width=True)
            
            if MODEL_LOADED:
                prediction = predict_from_image(img)
            else:
                prediction = 0.5
        except:
            st.error("Image not found")
            prediction = 0.5
        
        st.caption(sample['description'])
    
    with col2:
        st.subheader(" Prediction")
        
        if prediction > 0.5:
            st.success("###  GRAVITATIONAL WAVE")
            st.metric("Confidence", f"{prediction*100:.1f}%")
        else:
            st.info("###  BACKGROUND NOISE")
            st.metric("Confidence", f"{(1-prediction)*100:.1f}%")
        
        st.markdown(f"**Type:** {sample['type']}")
        st.markdown(f"**Distance:** {sample['distance']}")
        
        st.markdown("---")
        prob_signal = prediction * 100
        prob_noise = (1 - prediction) * 100
        
        col_a, col_b = st.columns(2)
        col_a.metric("Signal", f"{prob_signal:.0f}%")
        col_b.metric("Noise", f"{prob_noise:.0f}%")
        
        st.progress(prob_signal/100)

# Footer
st.markdown("---")
st.header(" How It Works")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**1. Upload**")
with col2:
    st.markdown("**2. Preprocess**")
with col3:
    st.markdown("**3. CNN**")
with col4:
    st.markdown("**4. Classify**")

with st.expander(" Technical Details"):
    st.markdown("""
    **Preprocessing:** FFT whitening, Butterworth filter, Q-transform
    
    **CNN:** 4 blocks (32→64→128→256 filters)
    
    **Training:** 52 images, Adam optimizer
    """)

with st.expander(" Performance"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "60%")
    col2.metric("Precision", "58%")
    col3.metric("Recall", "56%")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with TensorFlow & Streamlit | Real CNN trained on LIGO data
</div>
""", unsafe_allow_html=True)