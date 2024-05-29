import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dataIO import (
    get_active_channels,
    keep_trace,
    load_channel_data,
    get_baseline,
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
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        x = self.transformers(x)
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        return x


def find_factor(n):
    for i in range(2, n):
        if n % i == 0:
            return i
    return n


def load_data(file_path, active_channels):
    signals = []
    for row, col in tqdm(active_channels, desc="Loading data"):
        signal, seizures, se, recording_length, _ = load_channel_data(
            file_path, row, col
        )
        if signal is not None:
            baseline_signal, new_recording_length = get_baseline(
                signal, seizures, recording_length
            )
            if keep_trace(baseline_signal, 0.065):
                signals.append(baseline_signal)
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

    signals = normalize_signals(signals)
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
    table = ProgressTable(["Epoch", "Step"])
    table.add_column("Loss")
    for epoch in range(num_epochs):
        table["Epoch"] = f"{epoch+1}/{num_epochs}"
        running_loss = 0.0
        for i, (inputs,) in enumerate(dataloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            table["Step"] = f"{i+1}/{len(dataloader)}"
            table["Loss"] = loss.item()

        table.next_row()
    table.close()

    return model


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
    test_file = "Slice3_0Mg_13-9-20_resample_100_channel_data.h5"
    folder = "./mv_data"
    active_channels = {}
    for file in os.listdir(folder)[:5]:
        print(file)
        if file == test_file.split("/")[-1]:
            print("skip")
            continue
        full_path = folder + "/" + file
        active_channels[full_path] = get_active_channels(full_path)
    print(f"Number of active channels: {len(active_channels)}")
    segment_length = 50
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.0001

    trained_model = train_model(
        active_channels,
        segment_length,
        num_epochs,
        batch_size,
        learning_rate,
    )

    model_save_path = "eegformer_autoencoder_v3.pth"
    save_model(trained_model, model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
