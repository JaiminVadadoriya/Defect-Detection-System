import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
import base64

# Configure page
st.set_page_config(
    page_title="NEU-DET Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #1b1b1b;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .probability-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.2rem 0;
    }
    .probability-fill {
        height: 20px;
        background-color: #1f77b4;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_defect_model():
    try:
        model = load_model('neu_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure 'neu_model.keras' is in the same directory as this app.")
        return None

# Define class names
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted', 'rolled', 'scratches']

def preprocess_image(img):
    """Preprocess image for prediction"""
    # Resize image to model's expected input size
    img_resized = img.resize((200, 200))
    
    # Convert to array and normalize
    img_array = np.array(img_resized) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_defect(model, img_array):
    """Make prediction on preprocessed image"""
    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def display_predictions(predicted_class, confidence, all_predictions):
    """Display prediction results with probabilities"""
    st.markdown(f"""
    <div class="prediction-box">
        <h3>🎯 Predicted Defect: <span style="color: #1f77b4;">{CLASS_NAMES[predicted_class]}</span></h3>
        <h4>Confidence: {confidence:.2%}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Display all class probabilities
    st.subheader("📊 All Class Probabilities")
    
    # Sort predictions by probability (descending)
    sorted_indices = np.argsort(all_predictions)[::-1]
    
    for i in sorted_indices:
        prob = all_predictions[i]
        class_name = CLASS_NAMES[i]
        
        # Create color based on probability
        color = "#1f77b4" if i == predicted_class else "#95a5a6"
        
        # Create HTML for probability bar
        bar_html = f"""
        <div style="margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; color: {color};">{class_name}</span>
                <span style="color: {color};">{prob:.2%}</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: {prob*100}%; background-color: {color};">
                </div>
            </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 NEU-DET Defect Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_defect_model()
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("🛠️ Options")
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Upload Image", "Camera Capture", "Real-time Camera"]
    )
    
    # Display class information
    with st.sidebar.expander("ℹ️ Defect Classes"):
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{i+1}. **{class_name}**")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input")
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload an image to detect defects"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                
                # Process image
                if st.button("🔍 Detect Defects", type="primary"):
                    with st.spinner("Processing image..."):
                        img_array = preprocess_image(img)
                        predicted_class, confidence, all_predictions = predict_defect(model, img_array)
                        
                        if predicted_class is not None:
                            with col2:
                                st.header("📊 Results")
                                display_predictions(predicted_class, confidence, all_predictions)
        
        elif input_method == "Camera Capture":
            camera_image = st.camera_input("Take a picture")
            
            if camera_image is not None:
                # Display captured image
                img = Image.open(camera_image)
                st.image(img, caption="Captured Image", use_column_width=True)
                
                # Process image automatically
                with st.spinner("Processing image..."):
                    img_array = preprocess_image(img)
                    predicted_class, confidence, all_predictions = predict_defect(model, img_array)
                    
                    if predicted_class is not None:
                        with col2:
                            st.header("📊 Results")
                            display_predictions(predicted_class, confidence, all_predictions)
        
        elif input_method == "Real-time Camera":
            st.info("Real-time camera processing")
            
            # Create a placeholder for the camera feed
            camera_placeholder = st.empty()
            results_placeholder = st.empty()
            
            # Start/Stop buttons
            start_button = st.button("📹 Start Camera")
            stop_button = st.button("⏹️ Stop Camera")
            
            if start_button:
                # Initialize camera
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Cannot open camera")
                    return
                
                # Real-time processing
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Process every few frames to reduce computation
                    if hasattr(st.session_state, 'frame_count'):
                        st.session_state.frame_count += 1
                    else:
                        st.session_state.frame_count = 0
                    
                    if st.session_state.frame_count % 30 == 0:  # Process every 30 frames
                        # Convert to PIL Image
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Predict
                        img_array = preprocess_image(pil_image)
                        predicted_class, confidence, all_predictions = predict_defect(model, img_array)
                        
                        if predicted_class is not None:
                            with col2:
                                st.header("📊 Real-time Results")
                                display_predictions(predicted_class, confidence, all_predictions)
                    
                    # Check for stop condition
                    if stop_button:
                        break
                
                cap.release()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>🏭 NEU-DET Steel Surface Defect Detection System</p>
            <p>Powered by TensorFlow & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()