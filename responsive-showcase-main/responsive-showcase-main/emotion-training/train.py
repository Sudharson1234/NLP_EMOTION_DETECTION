import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("fer2013.csv")

pixels = data["pixels"].tolist()
emotions = data["emotion"].tolist()

X = []
y = []

for pix in pixels:
    img = np.array(pix.split(), dtype="float32")
    img = img.reshape(48, 48, 1)
    X.append(img)

X = np.array(X) / 255.0
y = tf.keras.utils.to_categorical(emotions, 7)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.1
)

# Save model
model.save("emotion_model.h5")

print("✅ Model saved as emotion_model.h5")
