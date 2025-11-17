import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
import base64
import time

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

# Load model with caching and performance optimizations
@st.cache_resource
def load_defect_model():
    """Load TensorFlow Lite model optimized for edge devices"""
    try:
        interpreter = tf.lite.Interpreter(model_path='neu_model.tflite')
        interpreter.allocate_tensors()
        
        # Enable performance optimizations
        # Use GPU delegate if available (for edge devices with GPU)
        try:
            interpreter.set_num_threads(4)  # Optimize thread usage
        except:
            pass
        
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure 'neu_model.tflite' is in the same directory as this app.")
        return None

# Define class names
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted', 'rolled', 'scratches']

def preprocess_image(img, optimize=True):
    """Preprocess image for prediction with performance optimizations"""
    # Resize image to model's expected input size
    # Using LANCZOS for better quality, but can switch to NEAREST for speed
    if optimize:
        img_resized = img.resize((200, 200), Image.NEAREST)  # Faster resizing
    else:
        img_resized = img.resize((200, 200), Image.LANCZOS)  # Better quality
    
    # Convert to array and normalize in one step (optimized)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_defect(interpreter, img_array):
    """Make prediction on preprocessed image with optimized inference pipeline"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Ensure correct input type and shape (optimized - avoid unnecessary copies)
        if img_array.dtype != np.float32:
            input_data = img_array.astype(np.float32)
        else:
            input_data = img_array
        
        # Optimized inference pipeline
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = np.argmax(output_data[0])
        confidence = output_data[0][predicted_class]
        return predicted_class, confidence, output_data[0]
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
            st.info("Real-time camera processing with optimized inference pipeline")
            
            # Performance settings
            with st.sidebar.expander("⚙️ Performance Settings"):
                frame_skip = st.slider("Process every N frames", min_value=5, max_value=60, value=15, 
                                      help="Higher values = better performance, lower latency")
                show_fps = st.checkbox("Show FPS", value=True)
            
            # Create placeholders for the camera feed and results
            camera_placeholder = st.empty()
            results_placeholder = st.empty()
            fps_placeholder = st.empty()
            
            # Initialize session state
            if 'camera_running' not in st.session_state:
                st.session_state.camera_running = False
            if 'frame_count' not in st.session_state:
                st.session_state.frame_count = 0
            if 'last_fps_time' not in st.session_state:
                st.session_state.last_fps_time = time.time()
            if 'fps_counter' not in st.session_state:
                st.session_state.fps_counter = 0
            
            # Start/Stop buttons
            col_start, col_stop = st.columns(2)
            with col_start:
                start_button = st.button("📹 Start Camera", disabled=st.session_state.camera_running)
            with col_stop:
                stop_button = st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_running)
            
            if start_button and not st.session_state.camera_running:
                st.session_state.camera_running = True
                st.session_state.frame_count = 0
                st.session_state.last_fps_time = time.time()
                st.session_state.fps_counter = 0
                
                # Initialize camera with optimized settings
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Optimize resolution
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not cap.isOpened():
                    st.error("Cannot open camera")
                    st.session_state.camera_running = False
                    return
                
                # Real-time processing with optimized pipeline
                last_prediction = None
                last_prediction_time = 0
                
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    # Convert BGR to RGB (optimized)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Update frame counter
                    st.session_state.frame_count += 1
                    st.session_state.fps_counter += 1
                    
                    # Calculate FPS
                    current_time = time.time()
                    if current_time - st.session_state.last_fps_time >= 1.0:
                        fps = st.session_state.fps_counter / (current_time - st.session_state.last_fps_time)
                        if show_fps:
                            fps_placeholder.metric("FPS", f"{fps:.1f}")
                        st.session_state.fps_counter = 0
                        st.session_state.last_fps_time = current_time
                    
                    # Process frames with optimized skipping
                    should_process = (st.session_state.frame_count % frame_skip == 0) or \
                                   (current_time - last_prediction_time > 0.5)  # Force process every 0.5s
                    
                    if should_process:
                        # Convert to PIL Image (optimized)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Predict with optimized preprocessing
                        img_array = preprocess_image(pil_image, optimize=True)
                        predicted_class, confidence, all_predictions = predict_defect(model, img_array)
                        
                        if predicted_class is not None:
                            last_prediction = (predicted_class, confidence, all_predictions)
                            last_prediction_time = current_time
                            
                            # Display results
                            with col2:
                                st.header("📊 Real-time Results")
                                display_predictions(predicted_class, confidence, all_predictions)
                    
                    # Display last prediction if available
                    elif last_prediction is not None:
                        with col2:
                            st.header("📊 Real-time Results")
                            display_predictions(last_prediction[0], last_prediction[1], last_prediction[2])
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)
                
                cap.release()
                st.session_state.camera_running = False
                st.rerun()
            
            if stop_button:
                st.session_state.camera_running = False
                st.rerun()
    
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