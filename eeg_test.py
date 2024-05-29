import torch
from tqdm import tqdm
from eeg import (
    EEGformerAutoencoder,
    load_model,
    detect_seizures,
)
from dataIO import get_active_channels, load_channel_data
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps")

file_path = "./mv_data/Slice3_0Mg_13-9-20_resample_100_channel_data.h5"
model_save_path = "eegformer_autoencoder_v1.pth"

active_channels = get_active_channels(file_path)
num_channels = len(active_channels)
print(f"Number of active channels: {num_channels}")

segment_length = 50
seizure_threshold = 0.1

loaded_model = load_model(EEGformerAutoencoder, model_save_path, segment_length)

# Create a figure and axis for the graph
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.set_title("Real-time Seizure Detection")
plt.ion()  # Enable interactive mode

# Iterate over active channels and detect seizures
for channel_idx, (row, col) in enumerate(active_channels[2:]):
    print(f"Processing channel {channel_idx + 1}/{num_channels}")
    signal, seizures, se, recording_length, _ = load_channel_data(file_path, row, col)
    signal = np.array(signal)
    if signal is not None:
        num_segments = len(signal) // segment_length
        time = np.arange(len(signal)) / recording_length

        # Plot the trace data
        ax.clear()
        ax.plot(time, signal, label=f"Channel ({row}, {col})")

        # Detect seizures and highlight them on the graph in real-time
        for i in tqdm(range(num_segments)):
            segment = signal[i * segment_length : (i + 1) * segment_length]
            if len(segment) < segment_length:
                segment = np.pad(
                    segment, (0, segment_length - len(segment)), "constant"
                )
            is_seizure = detect_seizures(loaded_model, segment, seizure_threshold)
            if is_seizure:
                start_time = i * segment_length / recording_length
                end_time = (i + 1) * segment_length / recording_length
                ax.axvspan(start_time, end_time, alpha=0.3, color="red")
                plt.draw()
                plt.pause(0.001)  # Pause for a short duration to allow graph update

        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Real-time Seizure Detection - Channel ({row}, {col})")
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)  # Pause for 1 second before moving to the next channel

plt.ioff()  # Disable interactive mode
plt.show()  # Keep the final graph displayed until closed
