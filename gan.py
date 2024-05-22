import sys
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Conv1D,
    LeakyReLU,
    Dropout,
    BatchNormalization,
)
import numpy as np
from dataIO import (
    load_channel_data,
    get_channel_signal_without_seizures,
    get_active_channels,
)


# Generator
def build_generator(input_shape):
    inputs = Input(shape=input_shape)

    x = Dense(64)(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    outputs = Dense(input_shape[-1])(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Discriminator
def build_discriminator(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = LSTM(64)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# LSTM-GAN Anomaly Detector
class LSTMGANAnomalyDetector:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.generator = build_generator(input_shape)
        self.discriminator = build_discriminator(input_shape)

    def compile(self, g_optimizer, d_optimizer):
        self.generator.compile(loss="mse", optimizer=g_optimizer)
        self.discriminator.compile(loss="binary_crossentropy", optimizer=d_optimizer)

    def train(self, X_train, epochs, batch_size):
        for epoch in range(epochs):
            for _ in range(X_train.shape[0] // batch_size):
                # Train the discriminator
                real_batch = X_train[
                    np.random.randint(0, X_train.shape[0], size=batch_size)
                ]
                noise = np.random.normal(
                    0, 1, (batch_size, self.input_shape[0], self.input_shape[1])
                )
                generated_batch = self.generator.predict(noise)

                X = np.concatenate([real_batch, generated_batch])
                y_dis = np.zeros(2 * batch_size)
                y_dis[:batch_size] = 0.9  # One-sided label smoothing

                self.discriminator.trainable = True
                self.discriminator.train_on_batch(X, y_dis)

                # Train the generator
                noise = np.random.normal(
                    0, 1, (batch_size, self.input_shape[0], self.input_shape[1])
                )
                y_gen = np.ones(batch_size)

                self.discriminator.trainable = False
                self.generator.train_on_batch(noise, y_gen)

            # Print progress
            print(f"Epoch: {epoch + 1}/{epochs}")

    def detect_anomalies(self, X_test, beta, threshold):
        generated_data = self.generator.predict(X_test)
        discriminator_output = self.discriminator.predict(X_test)

        reconstruction_loss = np.mean(np.square(X_test - generated_data), axis=1)
        discrimination_loss = np.abs(
            discriminator_output - np.ones_like(discriminator_output)
        ).flatten()

        anomaly_scores = beta * reconstruction_loss + (1 - beta) * discrimination_loss
        anomalies = (anomaly_scores > threshold).astype(int)

        return anomalies


# Usage example
window_size = 100  # Specify the window size for segmenting the signal
num_features = 1  # Specify the number of features in the signal
stride = 50  # Specify the stride for sliding the window
batch_size = 32

input_shape = (window_size, num_features)  # Specify your input shape
anomaly_detector = LSTMGANAnomalyDetector(input_shape)

g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
anomaly_detector.compile(g_optimizer, d_optimizer)

# Load and preprocess your data
data = {}  # Load your data using the provided code
file_path = "./mv_data/Slice2_0Mg_13-9-20_resample_100_channel_data.h5"
active_channels = get_active_channels(file_path)

# Preprocess the data
X_train = []
y_train = []

for active_channel in active_channels:
    signal, seizures, _ = load_channel_data(file_path, *active_channel)
    no_seizure_signal = get_channel_signal_without_seizures(signal, seizures)
    for i in range(0, len(signal) - window_size + 1, stride):
        segment = signal[i : i + window_size]
        X_train.append(segment)
        y_train.append(0)  # Assuming all segments are non-seizure (normal) for training

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape the data if necessary
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))

# Ensure there are enough samples to train
if X_train.shape[0] < batch_size:
    print("Not enough training data.")
    sys.exit(1)

# Train the model
epochs = 10
batch_size = 32
anomaly_detector.train(X_train, epochs, batch_size)

# Perform anomaly detection on test data
file_path = "./mv_data/Slice3_0Mg_13-9-20_resample_100_channel_data.h5"
active_channels = get_active_channels(file_path)

X_test = []
for active_channel in active_channels:
    signal, seizures, _ = load_channel_data(file_path, *active_channel)
    for i in range(0, len(signal) - window_size + 1, stride):
        segment = signal[i : i + window_size]
        X_test.append(segment)

X_test = np.array(X_test)

# Check if X_test is empty before reshaping
if X_test.shape[0] > 0:
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))
else:
    print("No test data available.")
    sys.exit(1)

beta = (
    0.5  # Specify the beta value for combining reconstruction and discrimination loss
)
threshold = 0.5  # Specify the threshold for anomaly detection
print("Performing anomaly detection...")
anomalies = anomaly_detector.detect_anomalies(X_test, beta, threshold)

# Analyze the anomalies
print("Anomalies detected:", anomalies.sum())
