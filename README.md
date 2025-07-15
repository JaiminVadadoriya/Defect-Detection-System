

---

# Defect Detection with Streamlit

## Project Overview

This project uses a pre-trained deep learning model to detect defects in images. The model is hosted in a Streamlit app where users can either upload an image, capture a photo using their camera, or process a live camera feed. The app provides detailed results, including predicted classes, confidence scores, and a clean, modern interface.

---

## Installation Instructions

### Option 1: Using `pip` (Recommended)

1. **Create and Activate a Virtual Environment**:

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

2. **Install Required Dependencies**:

   Install the dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using `conda` (For GPU Support)

If you're setting up the project with GPU support on Windows, follow these steps to install the required packages using `conda`:

1. **Create and Activate a Conda Environment**:

   Create a new conda environment for the project:

   ```bash
   conda create -n defect-detection python=3.8
   conda activate defect-detection
   ```

2. **Install CUDA and cuDNN (GPU support)**:

   If you plan to use TensorFlow with GPU support, install the necessary versions of `cudatoolkit` and `cudnn`. **Note:** Only CUDA 11.2 and cuDNN 8.1.0 are supported with TensorFlow 2.10 on Windows.

   ```bash
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```

3. **Install TensorFlow (with GPU support)**:

   Install the compatible version of TensorFlow (anything below 2.11):

   ```bash
   python -m pip install "tensorflow<2.11"
   ```

4. **Verify GPU Support**:

   After installing, you can verify that TensorFlow is using the GPU by running the following command:

   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

   If everything is set up correctly, it should list your GPU as a physical device.

---

## Ensure Model File is Present

Make sure the following files are in the same directory as your `app.py` file:

```
project/
├── app.py             (Streamlit app)
├── neu_model.keras    (Your trained model file)
└── requirements.txt   (Optional, but recommended)
```

---

## Running the App

Once everything is set up, you can run the app with Streamlit:

```bash
streamlit run app.py
```

This will start a local web server and open the app in your browser.

---

## Features

### 📤 **Image Upload**

Upload an image to detect defects.

### 📷 **Camera Capture**

Take a picture directly from your device's camera to detect defects in real-time.

### 📹 **Real-time Processing**

Detect defects continuously with live camera feed and processing every 30 frames.

### 📊 **Detailed Results**

Display predicted class, confidence, and probabilities for each possible class.

### 🎨 **Modern UI**

A clean and responsive interface with custom styling to make interactions intuitive.

---

## Key Components

### 1. **Model Loading**

* The model is cached on the first load for faster performance.
* If the model file is missing, the app will show an error message with instructions to add the model.

### 2. **Image Processing**

* Automatically resizes uploaded or captured images to the model's input size (200x200).
* Proper normalization (dividing by 255) is applied to the image data.
* Handles batch dimension for prediction.

### 3. **Prediction Display**

* The primary prediction is highlighted, showing the predicted class and its confidence level.
* All possible classes are displayed with their corresponding probabilities.
* Color-coded probability bars to visually represent class confidence.

### 4. **Three Input Methods**

* **Upload**: Upload an image file from your local machine.
* **Camera**: Take a single photo using your device's camera.
* **Real-time**: Continuously process frames from your camera (every 30 frames).

---

## Optional Enhancements

* **Model Re-training**: You can replace the `neu_model.keras` file with your own trained model.
* **Custom Styling**: Modify the app’s styling by updating the CSS within the Streamlit components.

---

## Troubleshooting

1. **Model File Missing**:
   Ensure that the `neu_model.keras` file is in the same directory as `app.py`.

2. **Dependencies**:
   If you encounter errors related to missing libraries, make sure you’ve installed the required dependencies using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Camera Issues**:
   If the camera feed doesn't work, check that your browser has permissions to access the camera. Additionally, make sure that the camera is properly connected and working.

---
