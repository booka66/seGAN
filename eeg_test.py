import torch
from eeg import (
    EEGformerAutoencoder,
    load_model,
    load_data,
    create_dataset,
    pad_signals,
    detect_anomalies,
)
from dataIO import get_active_channels

device = torch.device("mps")


file_path = "./mv_data/Slice3_0Mg_13-9-20_resample_100_channel_data.h5"
model_save_path = "eegformer_autoencoder.pth"

active_channels = get_active_channels(file_path)
num_channels = len(active_channels)
print(f"Number of active channels: {num_channels}")
segment_length = 50


new_signals = load_data(
    file_path,
    active_channels,
)

new_signals, max_length = pad_signals(new_signals, segment_length)

new_data = create_dataset(new_signals, segment_length, max_length).to(device)
anomaly_threshold = 0.1
loaded_model = load_model(
    EEGformerAutoencoder, model_save_path, len(new_signals), segment_length
)
anomalies = detect_anomalies(loaded_model, new_data, anomaly_threshold)

print(f"Anomalies detected: {anomalies}")
