import h5py
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import os


def get_active_channels(file_path):
    active_channels = []
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        for key in keys:
            key_split = key.split("_")
            row, col = int(key_split[0]), int(key_split[1])
            active_channels.append((row, col))
    return active_channels


def load_channel_data(file_path, row, col):
    with h5py.File(file_path, "r") as f:
        channel_key = f"{row}_{col}"
        if channel_key in f:
            channel_group = f[channel_key]
            signal = channel_group["signal"][:]
            seizures = channel_group["SzEventsTimes"][()]
            # check if seizures is a scalar
            if len(seizures.shape) == 0:
                seizures = [seizures]
            return signal, len(seizures)
        else:
            print(f"Channel ({row}, {col}) not found in the dataset.")
            return None


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(initialdir="/Users/booka66/seGAN/")
    data = {}
    total_seizures = 0

    for file_name in tqdm(os.listdir(folder_path), desc="Processing files"):
        if file_name.endswith(".h5"):
            file_path = os.path.join(folder_path, file_name)
            for channel in get_active_channels(file_path):
                signal, seizure_count = load_channel_data(file_path, *channel)
                total_seizures += seizure_count
                if signal is not None:
                    data[f"{file_name}_{channel}"] = signal
                else:
                    print("Error occurred during analysis")

    print("Data loaded successfully.")
    print("Number of channels:", len(data))
    print("Total seizures:", total_seizures)
    print("Example data:")
    print(list(data.items())[:5])
