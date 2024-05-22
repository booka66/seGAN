import numpy as np
import matlab
import matlab.engine
from time import perf_counter
import h5py
import sys
import tkinter as tk
from tkinter import filedialog
import pandas as pd


def get_channels(file_path):
    with h5py.File(file_path, "r") as f:
        recElectrodeList = f["/3BRecInfo/3BMeaStreams/Raw/Chs"]
        rows = recElectrodeList["Row"][()]
        cols = recElectrodeList["Col"][()]
        return rows, cols


def get_data(file_path):
    try:
        eng = matlab.engine.start_matlab()
        recording_length = None
        time_vector = None
        data = None
        active_channels = []

        start_time_1 = perf_counter()
        print("Beginning analysis...")

        data_cell, total_channels, sampling_rate, num_rec_frames = (
            eng.vectorized_state_extractor(file_path, nargout=4)
        )

        end_time_1 = perf_counter()
        print(f"State extraction complete in {end_time_1 - start_time_1} seconds")

        total_channels = int(total_channels)
        sampling_rate = float(sampling_rate)
        num_rec_frames = int(num_rec_frames)

        data_np = np.array(data_cell)
        data = data_np.reshape((64, 64))

        recording_length = (1 / sampling_rate) * (num_rec_frames - 1)
        time_vector = [i / sampling_rate for i in range(num_rec_frames)]

        rows, cols = get_channels(file_path)
        active_channels = list(zip(rows, cols))
        zero_indexed_active_channels = [(r - 1, c - 1) for r, c in active_channels]

        return (
            data,
            zero_indexed_active_channels,
            time_vector,
            recording_length,
            sampling_rate,
        )

    except Exception as e:
        print(e)
        return None, None, None, None, None


def load_channel_data(data, active_channels):
    channel_data = []

    for row, col in active_channels:
        channel_info = data[row, col]
        name = channel_info["name"]
        signal = channel_info["signal"]
        sz_events_times = channel_info["SzEventsTimes"]
        se_list = channel_info["SE_List"]

        channel_data.append(
            {
                "name": name,
                "signal": signal,
                "SzEventsTimes": sz_events_times,
                "SE_List": se_list,
            }
        )

    return pd.DataFrame(channel_data)


def save_channel_data(channel_dataset, file_path):
    file_name = file_path.split("/")[-1].split(".")[0]
    output_file = f"{file_name}_channel_data.csv"
    channel_dataset.to_csv(output_file, index=False)
    print(f"Channel dataset saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # Allow the user to input the file path with a file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(
            initialdir="/Users/booka66/seGAN/"
        )  # Show the file dialog
    else:
        file_path = sys.argv[1]

    data, active_channels, time_vector, recording_length, sampling_rate = get_data(
        file_path
    )

    if data is not None:
        print("Active channels:", active_channels)
        print("Recording length:", recording_length)
        print("Sampling rate:", sampling_rate)

        channel_dataset = load_channel_data(data, active_channels)
        print("Channel dataset:")
        print(channel_dataset.head())

        save_channel_data(channel_dataset, file_path)
    else:
        print("Error occurred during analysis")
