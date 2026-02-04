import tensorflow as tf
import numpy as np

try:
    model = tf.keras.models.load_model("emotion_model.h5", compile=False)
    print("Model loaded.")
    
    # Test with the shape we just implemented
    # (1, 48, 48, 1)
    input_shape = (1, 48, 48, 1)
    fake_input = np.zeros(input_shape)
    
    print(f"Testing prediction with shape {input_shape}...")
    prediction = model.predict(fake_input, verbose=0)
    print("Prediction successful:", prediction)

except Exception as e:
    print("Error:", e)
