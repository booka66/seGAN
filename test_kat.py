import torch
import numpy as np
from tqdm import tqdm
from dataIO import (
    get_active_channels,
    remove_first_n_seconds,
    load_channel_data,
    downsample_signal,
)
from kat import EEGformerAutoencoder, cluster_latent_representations, device
import matplotlib.pyplot as plt


def load_test_data(file_path, active_channels):
    signals = []
    for row, col in tqdm(active_channels, desc="Loading test data", unit="channels"):
        signal, _, _, recording_length, _ = load_channel_data(file_path, row, col)
        signal = remove_first_n_seconds(signal, 20, recording_length)
        if signal is not None:
            downsampled_signal = downsample_signal(signal, 7)
            signals.append(downsampled_signal)
    return signals


def plot_signals_with_labels(signals, labels, segment_length, n_clusters):
    plt.figure(figsize=(12, 8))
    for i, signal in enumerate(signals):
        plt.subplot(len(signals), 1, i + 1)
        num_segments = len(signal) // segment_length
        for j in range(num_segments):
            start = j * segment_length
            end = (j + 1) * segment_length
            label = labels[i * num_segments + j]
            color = plt.cm.get_cmap("viridis")(label / (n_clusters - 1))
            plt.plot(np.arange(start, end), signal[start:end], color=color)
        plt.ylabel(f"Signal {i+1}")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


def main():
    model_path = "kat_v1.pth"
    segment_length = 50
    n_clusters = 2

    # Load the saved model
    print("Loading model...")
    model = EEGformerAutoencoder(segment_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.")

    test_file = "./mv_data/Slice3_0Mg_13-9-20_resample_100_channel_data.h5"
    active_channels = get_active_channels(test_file)
    print("Loading test data...")
    test_signals = load_test_data(test_file, active_channels[:5])
    print("Test data loaded.")

    labels = cluster_latent_representations(
        model, test_signals, segment_length, n_clusters
    )

    for i, label in enumerate(labels):
        print(f"Segment {i+1}: Cluster {label}")

    plot_signals_with_labels(test_signals, labels, segment_length, n_clusters)


if __name__ == "__main__":
    main()
