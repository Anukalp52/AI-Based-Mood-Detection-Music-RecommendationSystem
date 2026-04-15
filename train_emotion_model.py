# train_emotion_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Adjust path if needed
FER_CSV = "C:/Users/anuka/OneDrive/Desktop/hello/fer2013.csv"

def load_fer(csv_path):
    df = pd.read_csv(csv_path)
    pixels = df['pixels'].tolist()
    faces = []
    for px in pixels:
        arr = np.fromstring(px, dtype=int, sep=' ').reshape(48,48)
        faces.append(arr)
    X = np.stack(faces).astype('float32') / 255.0
    X = np.expand_dims(X, -1)  # (N,48,48,1)
    y = df['emotion'].values  # 0-6 labels
    return X, y

print("Loading FER dataset (this may take some minutes)...")
X, y = load_fer(FER_CSV)
print("Shape:", X.shape, y.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.12, random_state=42, stratify=y)

# simple CNN
def build_model():
    inp = layers.Input(shape=(48,48,1))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(7, activation='softmax')(x)

    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("emotion_model.h5", save_best_only=True, monitor='val_accuracy', mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]



history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=64,
    callbacks=callbacks
)

print("Training complete. Best model saved as emotion_model.h5")
