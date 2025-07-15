import tensorflow as tf

# Load your original model
model = tf.keras.models.load_model("neu_model.keras")

# Convert to TFLite with float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Use float16 quantization
tflite_model = converter.convert()

# Save the smaller model
with open("neu_model.tflite", "wb") as f:
    f.write(tflite_model)
