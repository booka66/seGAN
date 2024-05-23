import tensorflow as tf
import numpy as np
import os
import h5py
from tqdm import tqdm
from dataIO import (
    load_channel_data,
    get_channel_signal_without_seizures,
    get_active_channels,
)

# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Generator
def build_generator(latent_dim, max_length):
    inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128)(inputs)
    x = tf.keras.layers.RepeatVector(max_length)(x)
    x = tf.keras.layers.LSTM(256, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(512, return_sequences=True)(x)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation="tanh")
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Discriminator
def build_discriminator(max_length):
    inputs = tf.keras.layers.Input(shape=(max_length, 1))
    x = tf.keras.layers.LSTM(512, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(256)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_max_length(folder):
    files = os.listdir(folder)[:5]
    max_length = 0
    for file in tqdm(files, desc="Checking max length", unit="file"):
        if file.endswith(".h5"):
            active_channels = get_active_channels(os.path.join(folder, file))
            signal, _, _ = load_channel_data(
                os.path.join(folder, file), *active_channels[0]
            )
            max_length = max(max_length, len(signal))
    # print("Max length:", max_length)
    return max_length

def generate_signal(generator, latent_dim, length):
    random_latent_vector = tf.random.normal(shape=(1, latent_dim))
    generated_signal = generator(random_latent_vector)
    generated_signal = tf.reshape(generated_signal, (length, 1))
    return generated_signal

def load_data(folder, segment_length=1000, downsample_factor=1):
    files = os.listdir(folder)[:5]
    for file in files:
        if file.endswith(".h5"):
            active_channels = get_active_channels(os.path.join(folder, file))
            for channel in active_channels:
                signal, _, _ = load_channel_data(os.path.join(folder, file), *channel)
                signal = signal[::downsample_factor]  # Downsample the signal
                for i in range(0, len(signal), segment_length):
                    segment = signal[i : i + segment_length]
                    if len(segment) < segment_length:
                        pad_width = segment_length - len(segment)
                        segment = np.pad(segment, (0, pad_width), mode="constant")
                    yield segment


# GAN
class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim, max_length):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.max_length = max_length

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @tf.function(reduce_retracing=True)
    def train_step(self, real_data):
        if not real_data:
            return {"d_loss": 0.0, "g_loss": 0.0}

        real_data = real_data[0]  # Extract the first element from the tuple
        real_data = tf.cast(real_data, tf.float16)  # Cast real_data to float16
        batch_size = tf.shape(real_data)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # print("Generating data...")
        generated_data = self.generator(random_latent_vectors)

        combined_data = tf.concat([generated_data, real_data], axis=0)

        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_data)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}


# Training
data_folder = "./mv_data/"
segment_length = 100
downsample_factor = 50
data_generator = load_data(data_folder, segment_length, downsample_factor)
latent_dim = 2
epochs = 10
batch_size = 32

discriminator = build_discriminator(segment_length)
generator = build_generator(latent_dim, segment_length)

gan = GAN(discriminator, generator, latent_dim, segment_length)

gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=tf.keras.losses.BinaryCrossentropy(),
)

for epoch in range(epochs):
    # print(f"Epoch {epoch+1}/{epochs}")
    batch_data = []
    for signal in tqdm(data_generator, desc=f"Epoch {epoch+1}", unit="signal", leave=False):
        batch_data.append(signal)
        if len(batch_data) == batch_size:
            batch_data = np.array(batch_data)
            batch_data = np.reshape(
                batch_data, (batch_size, segment_length, 1)
            )  # Reshape the data
            loss = gan.train_on_batch(batch_data)
            # if isinstance(loss, dict):
            #     # print("Discriminator Loss:", loss.get("d_loss", 0.0))
            #     # print("Generator Loss:", loss.get("g_loss", 0.0))
            # elif isinstance(loss, list):
            #     # print("Discriminator Loss:", loss[0])
            #     # print("Generator Loss:", loss[1])
            # else:
            #     # print("Unexpected loss format:", loss)
            batch_data = []
    if len(batch_data) > 0:
        batch_data = np.array(batch_data)
        if batch_data.shape[0] > 0:
            batch_data = np.reshape(
                batch_data, (-1, segment_length, 1)
            )  # Reshape the data
            loss = gan.train_on_batch(batch_data)
    #         if isinstance(loss, dict):
    #             # print("Discriminator Loss:", loss.get("d_loss", 0.0))
    #             # print("Generator Loss:", loss.get("g_loss", 0.0))
    #         elif isinstance(loss, list):
    #             # print("Discriminator Loss:", loss[0])
    #             # print("Generator Loss:", loss[1])
    #         else:
    #             # print("Unexpected loss format:", loss)
    #     else:
    #         # print("Empty batch data, skipping training step.")
    # else:
    #     # print("No batch data, skipping training step.")


# Save the generator model
generator.save("generator_model.h5")

# Save the discriminator model
discriminator.save("discriminator_model.h5")

# Create a new trace of n length
n = 500  # Desired length of the generated signal
generated_signal = generate_signal(generator, latent_dim, n)


# # Load the saved generator model
# loaded_generator = tf.keras.models.load_model("generator_model.h5")
#
# # Load the saved discriminator model
# loaded_discriminator = tf.keras.models.load_model("discriminator_model.h5")
# # Generate a new signal using the loaded generator model
# generated_signal = generate_signal(loaded_generator, latent_dim, n)
