import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Build a clean model for 48x48 grayscale images
model = Sequential([
    Input(shape=(48,48,1)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(6, activation="softmax")   # 6 emotions
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Save clean working model
model.save("emotion_model.h5")

print("✅ New clean emotion_model.h5 created successfully!")
