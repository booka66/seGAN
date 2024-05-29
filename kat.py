import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from tqdm import tqdm
from dataIO import (
    get_active_channels,
    keep_trace,
    load_channel_data,
    downsample_signal,
    remove_first_n_seconds,
)
from visualize_signals import plot_signal
import numpy as np
from progress_table import ProgressTable

device = torch.device("mps")


class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseConv1d, self).__init__()
        # Ensure out_channels is divisible by in_channels
        self.out_channels = (out_channels // in_channels) * in_channels
        self.depthwise = nn.Conv1d(
            in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, x):
        out = self.depthwise(x)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        return self.encoder(x)


class EEGformerAutoencoder(nn.Module):
    def __init__(self, segment_length):
        super(EEGformerAutoencoder, self).__init__()
        hidden_channels = 128
        nhead = 8
        print(f"Hidden channels: {hidden_channels}, nhead: {nhead}")
        assert (
            hidden_channels % nhead == 0
        ), "hidden_channels must be divisible by nhead"

        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_channels, 10),
            DepthwiseConv1d(hidden_channels, hidden_channels, 10),
            DepthwiseConv1d(hidden_channels, hidden_channels, 10),
        )

        self.flatten = nn.Flatten(start_dim=2)
        self.transformers = nn.Sequential(
            TransformerEncoder(
                d_model=hidden_channels,
                nhead=nhead,
                dim_feedforward=hidden_channels,
                num_layers=3,
            ),
            TransformerEncoder(
                d_model=hidden_channels,
                nhead=nhead,
                dim_feedforward=hidden_channels,
                num_layers=3,
            ),
            TransformerEncoder(
                d_model=hidden_channels,
                nhead=nhead,
                dim_feedforward=hidden_channels,
                num_layers=3,
            ),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels, hidden_channels, 10),
            nn.ConvTranspose1d(hidden_channels, hidden_channels, 10),
            nn.ConvTranspose1d(hidden_channels, 1, 10),
        )

    def forward(self, x):
        x = self.encoder(x)
        latent = self.flatten(x)
        x = latent.permute(0, 2, 1)
        x = self.transformers(x)
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        return x, latent


def find_factor(n):
    for i in range(2, n):
        if n % i == 0:
            return i
    return n


def load_data(file_path, active_channels):
    signals = []
    for row, col in tqdm(active_channels, desc="Loading data"):
        signal, _, _, recording_length, _ = load_channel_data(file_path, row, col)
        signal = remove_first_n_seconds(signal, 20, recording_length)
        if signal is not None:
            downsampled_signal = downsample_signal(signal, 7)
            signals.append(downsampled_signal)
    return signals


def pad_signals(signals, segment_length):
    max_length = max(len(signal) for signal in signals)
    padded_signals = []
    for signal in signals:
        padded_signal = np.pad(signal, (0, max_length - len(signal)), "constant")
        padded_signals.append(padded_signal)
    return np.array(padded_signals, dtype=np.float32), max_length


def normalize_signals(signals):
    normalized_signals = []
    for signal in signals:
        normalized_signal = (signal - np.min(signal)) / (
            np.max(signal) - np.min(signal)
        )
        normalized_signals.append(normalized_signal)
    return normalized_signals


def create_dataset(signals, segment_length, max_length):
    X = []
    for signal in signals:
        num_segments = max_length // segment_length
        for i in range(num_segments):
            segment = signal[i * segment_length : (i + 1) * segment_length]
            if len(segment) < segment_length:
                segment = np.pad(
                    segment, (0, segment_length - len(segment)), "constant"
                )
            X.append(segment)
    X = np.array(X, dtype=np.float32)
    X = torch.from_numpy(X)
    return X


def train_model(
    all_active_channels,
    segment_length,
    num_epochs,
    batch_size,
    learning_rate,
):
    print(f"Training model with {len(all_active_channels)} active channels")
    print(f"Segment length: {segment_length}, Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    signals = []
    for file_path, active_channels in all_active_channels.items():
        signals.extend(load_data(file_path, active_channels))
    num_signals = len(signals)

    signals, max_length = pad_signals(signals, segment_length)
    X = create_dataset(signals, segment_length, max_length)
    X = X.unsqueeze(1)  # Add a channel dimension
    X = X.to(device)
    print(f"Number of signals: {num_signals}")

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EEGformerAutoencoder(segment_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training model...")
    table = ProgressTable(
        pbar_embedded=False,
        pbar_style="angled alt green red",
    )
    table.column_width = 10
    for epoch in table(num_epochs, show_throughput=False, show_eta=True):
        table["Epoch"] = epoch
        running_loss = 0.0
        for i, (inputs,) in table(
            enumerate(dataloader),
            total=len(dataloader),
            show_throughput=False,
            show_eta=True,
        ):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            table.update("Current loss", loss.item(), color="red")
            table.update("Average loss", loss.item(), aggregate="mean", color="blue")
            table.update(
                "Running loss",
                running_loss / (i + 1),
                aggregate="mean",
            )
            table.update("Step", f"{i+1}/{len(dataloader)}", color="yellow")

        table.next_row()
    table.close()

    return model


def cluster_latent_representations(model, signals, segment_length, n_clusters):
    latent_representations = []
    for signal in signals:
        num_segments = len(signal) // segment_length
        for i in range(num_segments):
            segment = signal[i * segment_length : (i + 1) * segment_length]
            segment = (
                torch.from_numpy(segment).float().unsqueeze(0).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                _, latent = model(segment)
                latent_representations.append(latent.cpu().numpy().flatten())

    latent_representations = np.array(latent_representations)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(latent_representations)
    labels = kmeans.labels_
    return labels


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model_class, model_path, segment_length):
    model = model_class(segment_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def detect_seizures(model, segment, threshold):
    segment = torch.from_numpy(segment).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed_segment = model(segment)
        reconstruction_error = torch.mean((segment - reconstructed_segment) ** 2)
        is_seizure = reconstruction_error > threshold
    return is_seizure.item()


def main():
    folder = "./mv_data"
    active_channels = {}
    for file in os.listdir(folder)[:1]:
        print(file)
        if "100" not in file:
            print("skip")
            continue
        full_path = folder + "/" + file
        active_channels[full_path] = get_active_channels(full_path)
    print(f"Number of active channels: {len(active_channels)}")
    segment_length = 500
    num_epochs = 3
    batch_size = 16
    learning_rate = 0.0001
    n_clusters = 4

    trained_model = train_model(
        active_channels,
        segment_length,
        num_epochs,
        batch_size,
        learning_rate,
    )

    model_save_path = "kat_v1.pth"
    save_model(trained_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    signals = []
    for file_path, active_channels in active_channels.items():
        signals.extend(load_data(file_path, active_channels))
    labels = cluster_latent_representations(
        trained_model, signals, segment_length, n_clusters
    )
    print(f"Clustering labels: {labels}")


if __name__ == "__main__":
    main()
