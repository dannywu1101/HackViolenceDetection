from typing import Sequence
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Load dataset
neutral_df = pd.read_csv("neutral.txt")
violent_df = pd.read_csv("violent.txt")

# Inspect dataset
print(violent_df.head())
print(f"Dataset shape: {violent_df.shape}")

# Check if the dataset has enough columns and rows
if violent_df.shape[1] <= 1:
    raise ValueError("Dataset must contain more than one column.")

X = []
y = []
no_of_timesteps = 20

datasets = neutral_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(0)

datasets = violent_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(1)

# Extract the dataset values (ignoring the first column)
datasets = violent_df.iloc[:, 1:].values
n_samples = len(datasets)

# Check if the dataset has enough samples
if n_samples <= no_of_timesteps:
    raise ValueError(f"Dataset too small. Need more than {no_of_timesteps} samples, but got {n_samples}.")

# Create sequences
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(0)  # Adjust this according to your labeling logic

# Convert to numpy arrays
X, y = np.array(X), np.array(y)
print(f"Processed dataset shapes - X: {X.shape}, y: {y.shape}")

# No train-test split due to small dataset size
X_train, y_train = X, y

# Build and compile the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))  # Adjust based on your output classes

model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Save the model
model.save("lstm-hand-grasping.h5")

