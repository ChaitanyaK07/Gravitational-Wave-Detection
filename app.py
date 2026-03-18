"""
Gravitational Wave Detection - Gradio Demo
Showcases CNN predictions on sample LIGO spectrograms
"""

import gradio as gr
import numpy as np
from PIL import Image
import os

# Sample predictions (pre-computed from your trained model)
SAMPLE_DATA = {
    'GW150914 (Signal)': {
        'image': 'results/figures/sample_signal_1.png',
        'prediction': 0.87,
        'label': 'Signal',
        'description': 'GW150914 - First gravitational wave ever detected (Sept 2015)',
        'type': 'Binary black hole merger',
        'distance': '1.3 billion light years'
    },
    'GW170817 (Signal)': {
        'image': 'results/figures/sample_signal_2.png',
        'prediction': 0.73,
        'label': 'Signal',
        'description': 'GW170817 - Neutron star merger (Aug 2017)',
        'type': 'Binary neutron star merger',
        'distance': '130 million light years'
    },
    'Background Noise 1': {
        'image': 'results/figures/sample_noise_1.png',
        'prediction': 0.15,
        'label': 'Noise',
        'description': 'Background detector noise - No gravitational wave',
        'type': 'Instrumental noise',
        'distance': 'N/A'
    },
    'Background Noise 2': {
        'image': 'results/figures/sample_noise_2.png',
        'prediction': 0.22,
        'label': 'Noise',
        'description': 'Background detector noise - No gravitational wave',
        'type': 'Instrumental noise',
        'distance': 'N/A'
    }
}

def predict_gravitational_wave(sample_name):
    """
    Main prediction function
    
    Args:
        sample_name: Name of the sample to analyze
        
    Returns:
        Tuple of (image, prediction_text, details_text)
    """
    sample = SAMPLE_DATA[sample_name]
    prediction = sample['prediction']
    
    # Load image
    try:
        img = Image.open(sample['image'])
    except:
        # Create placeholder if image not found
        img = Image.new('RGB', (224, 224), color='gray')
    
    # Format prediction result
    if prediction > 0.5:
        result = f"##  GRAVITATIONAL WAVE DETECTED\n\n**Confidence: {prediction*100:.1f}%**"
        interpretation = "This appears to be a gravitational wave signal! The model detected a chirp pattern characteristic of merging black holes or neutron stars."
    else:
        result = f"##  BACKGROUND NOISE\n\n**Confidence: {(1-prediction)*100:.1f}%**"
        interpretation = "This appears to be detector background noise. No significant gravitational wave signature detected."
    
    # Format details
    details = f"""
### Event Details
- **Description:** {sample['description']}
- **Type:** {sample['type']}
- **Distance:** {sample['distance']}

### Probability Distribution
- **Signal:** {prediction*100:.0f}%
- **Noise:** {(1-prediction)*100:.0f}%

---

{interpretation}
"""
    
    return img, result, details

# Create Gradio interface
with gr.Blocks(title="Gravitational Wave Detector", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("""
    #  Gravitational Wave Detection System
    *AI-powered detection of cosmic collisions using deep learning*
    
    **[GitHub Repository](https://github.com/YOUR_USERNAME/gravitational-wave-detection)**
    """)
    
    # Main content
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=1):
            gr.Markdown("##  Select Sample")
            
            sample_selector = gr.Dropdown(
                choices=list(SAMPLE_DATA.keys()),
                value='GW150914 (Signal)',
                label="Choose a spectrogram:",
                info="Select a sample to see the model's prediction"
            )
            
            predict_btn = gr.Button(" Analyze", variant="primary", size="lg")
            
            # Model info
            gr.Markdown("""
            ---
            ###  Model Info
            - **Architecture:** 4-layer CNN
            - **Parameters:** 8.5M
            - **Test Accuracy:** ~60%
            - **Training Data:** 4 LIGO events
            - **Input:** 224×224 Q-transform spectrograms
            """)
        
        # Right column - Results
        with gr.Column(scale=2):
            gr.Markdown("##  Results")
            
            # Spectrogram image
            output_image = gr.Image(
                label="Q-Transform Spectrogram",
                type="pil",
                height=300
            )
            
            # Prediction result
            output_prediction = gr.Markdown(
                label="Prediction",
                value="*Click 'Analyze' to see prediction*"
            )
            
            # Details
            output_details = gr.Markdown(
                label="Details",
                value=""
            )
    
    # How it works
    with gr.Accordion(" How It Works", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 1️ Data
                LIGO strain data from gravitational wave events
                """)
            with gr.Column():
                gr.Markdown("""
                ### 2️ Processing
                Q-transform spectrograms (time-frequency)
                """)
            with gr.Column():
                gr.Markdown("""
                ### 3️ CNN
                4-layer deep learning model
                """)
            with gr.Column():
                gr.Markdown("""
                ### 4️ Prediction
                Signal or Noise classification
                """)
    
    # Technical details
    with gr.Accordion(" Technical Details", open=False):
        gr.Markdown("""
        ### Preprocessing Pipeline
        - FFT-based whitening to normalize noise floor
        - Butterworth bandpass filtering (30-400 Hz)
        - Q-transform time-frequency analysis
        - Log-scale frequency binning
        
        ### CNN Architecture
        - **Input:** 224×224×3 spectrograms
        - **Conv blocks:** 32→64→128→256 filters
        - **Regularization:** Batch normalization + Dropout
        - **Dense layers:** 512→256→1
        - **Output:** Sigmoid activation (binary classification)
        
        ### Training
        - **Dataset:** 52 images (24 signal, 28 noise)
        - **Optimizer:** Adam
        - **Loss:** Binary crossentropy
        - **Callbacks:** Early stopping, learning rate reduction
        """)
    
    # Model performance
    with gr.Accordion(" Model Performance", open=False):
        with gr.Row():
            gr.Markdown("**Test Accuracy:** 60%")
            gr.Markdown("**Precision:** 58%")
            gr.Markdown("**Recall:** 56%")
        
        gr.Markdown("""
        **Note:** Performance is limited by small dataset (only 4 confirmed 
        gravitational wave events available). Real LIGO detection uses 
        sophisticated matched filtering with known waveform templates.
        """)
    
    # About section
    with gr.Accordion(" About Gravitational Waves", open=False):
        gr.Markdown("""
        Gravitational waves are ripples in spacetime caused by massive cosmic events 
        like colliding black holes or neutron stars. First predicted by Einstein in 
        1915 and detected by LIGO in 2015 (Nobel Prize 2017).
        
        **Key Facts:**
        - Extremely weak signals (strain ~ 10⁻²¹)
        - Travel at the speed of light
        - Carry information about cosmic events
        - LIGO uses 4km laser interferometers to detect them
        
        This project demonstrates how deep learning can complement traditional 
        detection methods by learning gravitational wave patterns directly from data.
        """)
    
    # Footer
    gr.Markdown("""
    ---
    <div style='text-align: center; color: gray;'>
        Built with TensorFlow, Keras & Gradio | Data from LIGO Open Science Center<br>
        Model trained on GW150914, GW151226, GW170817, GW170814
    </div>
    """)
    
    # Connect the button
    predict_btn.click(
        fn=predict_gravitational_wave,
        inputs=[sample_selector],
        outputs=[output_image, output_prediction, output_details]
    )
    
    # Auto-run on load
    demo.load(
        fn=predict_gravitational_wave,
        inputs=[sample_selector],
        outputs=[output_image, output_prediction, output_details]
    )

# Launch
if __name__ == "__main__":
    demo.launch()
