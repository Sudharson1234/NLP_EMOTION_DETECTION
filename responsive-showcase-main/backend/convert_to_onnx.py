"""
Convert Keras model to ONNX format for stable inference
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tf2onnx
import onnx

print("Loading Keras model...")
model = tf.keras.models.load_model("emotion_model_pretrained.h5", compile=False)

print("Converting to ONNX...")
input_signature = [tf.TensorSpec([1, 64, 64, 1], tf.float32, name='input')]

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
onnx.save(onnx_model, "emotion_model.onnx")

print("[OK] Saved emotion_model.onnx successfully!")
