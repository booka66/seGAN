import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dataIO import load_channel_data, get_active_channels

device = torch.device("mps")


class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            out_channels,
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
    def __init__(self, num_channels, segment_length):
        super(EEGformerAutoencoder, self).__init__()
        hidden_channels = num_channels * 2
        nhead = find_factor(hidden_channels)
        print(f"Hidden channels: {hidden_channels}, nhead: {nhead}")
        assert (
            hidden_channels % nhead == 0
        ), "hidden_channels must be divisible by nhead"

        self.encoder = nn.Sequential(
            DepthwiseConv1d(num_channels, hidden_channels, 10),
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
            nn.ConvTranspose1d(hidden_channels, num_channels, 10),
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
    for row, col in active_channels:
        signal, seizures, se = load_channel_data(file_path, row, col)
        if signal is not None:
            signals.append(signal)
    return signals


def create_dataset(signals, segment_length):
    X = []
    for signal in signals:
        for i in range(0, len(signal), segment_length):
            segment = signal[i : i + segment_length]
            if len(segment) == segment_length:
                X.append(segment)
    X = torch.tensor(X, dtype=torch.float32)
    X = X.view(
        -1, len(signals), segment_length
    )  # Reshape to (batch_size, num_channels, segment_length)
    return X


def train_model(
    file_path, active_channels, segment_length, num_epochs, batch_size, learning_rate
):
    print(f"Training model with {len(active_channels)} active channels")
    print(f"Segment length: {segment_length}, Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print("Loading data...")
    signals = load_data(file_path, active_channels)
    num_channels = len(active_channels)

    X = create_dataset(signals, segment_length)
    X = X.to(device)
    print(f"Data shape: {X.shape}")

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EEGformerAutoencoder(num_channels, segment_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training model...")
    pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs,) in enumerate(dataloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f} Progress: {i+1}/{len(dataloader)}"
            )

        epoch_loss = running_loss / len(dataloader)
        pbar.set_postfix({"Loss": epoch_loss})
        pbar.update(1)
    pbar.close()

    return model


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model_class, model_path, num_channels, segment_length):
    model = model_class(num_channels, segment_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def detect_anomalies(model, data, threshold):
    data = data.to(device)
    with torch.no_grad():
        reconstructed_data = model(data)
        reconstruction_errors = torch.mean((data - reconstructed_data) ** 2, dim=(1, 2))
        anomalies = reconstruction_errors > threshold
    return anomalies


def main():
    file_path = (
        "/Users/booka66/seGAN/mv_data/Slice3_0Mg_13-9-20_resample_100_channel_data.h5"
    )
    active_channels = get_active_channels(file_path)
    num_channels = len(active_channels)
    print(f"Number of active channels: {num_channels}")
    segment_length = 50
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    trained_model = train_model(
        file_path,
        active_channels,
        segment_length,
        num_epochs,
        batch_size,
        learning_rate,
    )

    model_save_path = "eegformer_autoencoder.pth"
    save_model(trained_model, model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
