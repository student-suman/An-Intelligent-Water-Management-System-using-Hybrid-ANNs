# convert_to_tflite.py

import tensorflow as tf
from model import HybridSwishReLU # Import your custom activation function

# Define the path to your current Keras model
keras_model_path = 'irrigation_model.keras'

# Define the path for the new TFLite model
tflite_model_path = 'irrigation_model.tflite'

print(f"Loading Keras model from: {keras_model_path}")

# Load the Keras model, making sure to include your custom object
model = tf.keras.models.load_model(
    keras_model_path,
    custom_objects={'HybridSwishReLU': HybridSwishReLU}
)

# Initialize the TFLite converter from the loaded Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply default optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting model to TensorFlow Lite format...")

# Perform the conversion
tflite_model = converter.convert()

# Save the new .tflite model to a file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Model successfully converted and saved to: {tflite_model_path}")