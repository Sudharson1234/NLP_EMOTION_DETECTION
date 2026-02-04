import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model("emotion_model.h5", compile=False)
print("Model loaded.")

# Convert to TFLite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved emotion_model.tflite successfully!")
