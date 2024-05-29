import numpy as np
import os
import matlab
import matlab.engine
from time import perf_counter
import h5py
import tkinter as tk
from tkinter import filedialog


def get_channels(file_path):
    with h5py.File(file_path, "r") as f:
        recElectrodeList = f["/3BRecInfo/3BMeaStreams/Raw/Chs"]
        rows = recElectrodeList["Row"][()]
        cols = recElectrodeList["Col"][()]
    return rows, cols


def keep_trace(signal, threshold):
    """
    Returns true if all of the data in the signal is below the threshold.

    Args:
        signal (list): The signal to clean.
        threshold (float): The threshold value to ignore.

    Returns:
        bool: True if all of the data in the signal is below the threshold, False otherwise.
    """

    # Check that threshold is a number
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a number")

    # Check if all values in the signal are within the absolute value of the threshold don't use numpy
    for value in signal:
        if abs(value) > threshold:
            return False

    return True


def get_baseline(signal, seizures, recording_length):
    # print(f"Signal length: {len(signal)}")
    # print("Seizures:", seizures)
    if type(seizures) == np.float64:
        return signal, len(signal)
    if len(seizures) == 0:
        return signal, len(signal)

    samples_per_second = len(signal) / recording_length
    buffer = 30 * samples_per_second

    if len(seizures[0]) == 1:
        # print("YOU FOUND IT")
        # print("Seizures:", seizures)
        start = max(0, int(seizures[0][0] * samples_per_second - buffer))
        stop = min(len(signal), int(seizures[1][0] * samples_per_second + buffer))
        new_signal = np.concatenate((signal[:start], signal[stop:]))
        return new_signal, len(new_signal)

    signal_segments = []
    prev_stop = 0

    for seizure in seizures:
        start = max(0, int(seizure[0] * samples_per_second - buffer))
        stop = min(len(signal), int(seizure[1] * samples_per_second + buffer))
        signal_segments.append(signal[prev_stop:start])
        prev_stop = stop

    signal_segments.append(signal[prev_stop:])
    # print(f"Adjusted signal length: {sum(len(s) for s in signal_segments)}")
    new_signal = np.concatenate(signal_segments)

    start_index = int(20 * samples_per_second)
    new_signal = new_signal[start_index:]

    return new_signal, len(new_signal)


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
            active_channels,
            time_vector,
            recording_length,
            sampling_rate,
        )

    except Exception as e:
        print(e)
        return None, None, None, None, None


def save_channel_data(
    data, sampling_rate, recording_length, active_channels, file_path
):
    file_name = file_path.split("/")[-1].split(".")[0]
    output_folder = "mea_data_survey"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, f"{file_name}_channel_data.h5")

    with h5py.File(output_file, "w") as f:
        for row, col in active_channels:
            channel_info = data[row - 1, col - 1]
            channel_key = f"{row}_{col}"  # Use row and col as the key
            channel_group = f.create_group(channel_key)
            channel_group.create_dataset("signal", data=channel_info["signal"])
            channel_group.create_dataset(
                "SzEventsTimes", data=channel_info["SzEventsTimes"]
            )
            channel_group.create_dataset("SE_List", data=channel_info["SE_List"])
            channel_group.create_dataset("samplingRate", data=sampling_rate)
            channel_group.create_dataset("recordingLength", data=recording_length)

    print(f"Channel dataset saved to {output_file}")


def get_active_channels(file_path):
    active_channels = []
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        for key in keys:
            key_split = key.split("_")
            row, col = int(key_split[0]), int(key_split[1])
            active_channels.append((row, col))
    return active_channels


def load_channel_data(file_path: str, row: int, col: int):
    """
    Loads channel data from a given file path for a specific row and column.

    This function opens the file at `file_path` and reads the data for the channel specified by `row` and `col`.
    It returns the signal, recording length, sampling rate, number of seizures, and seizure event list for the channel.
    If the channel is not found in the dataset, it prints an error message and returns None.

    Args:
        file_path (str): The path to the file to read.
        row (int): The row number of the channel to load.
        col (int): The column number of the channel to load.

    Returns:
        tuple: A tuple containing the signal, seizures event list (start, stop, strength), SE event list (start, stop), recording length, and sampling rate for the channel.
        None: If the channel is not found in the dataset.

    Raises:
        IOError: If the file cannot be read.
    """

    with h5py.File(file_path, "r") as f:
        channel_key = f"{row}_{col}"
        if channel_key in f:
            channel_group = f[channel_key]
            signal = channel_group["signal"][:]
            signal = [x[0] for x in signal]
            seizures = channel_group["SzEventsTimes"][()]
            se = channel_group["SE_List"][()]
            recording_length = channel_group["recordingLength"][()]
            sampling_rate = channel_group["samplingRate"][()]
            return signal, seizures, se, recording_length, sampling_rate
        else:
            # print(f"Channel ({row}, {col}) not found in the dataset.")
            return None, None, None, None, None


if __name__ == "__main__":
    # Allow the user to select a folder with a file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(
        initialdir="/Users/booka66/seGAN/"
    )  # Show the folder dialog

    # Iterate over .brw files in the selected folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".brw"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")

            data, active_channels, time_vector, recording_length, sampling_rate = (
                get_data(file_path)
            )

            if data is not None:
                print("Active channels:", active_channels)
                print("Recording length:", recording_length)
                print("Sampling rate:", sampling_rate)

                save_channel_data(
                    data, sampling_rate, recording_length, active_channels, file_path
                )

                # Example usage of loading channel data
                row = 0
                col = 0
                signal, recording_length, sampling_rate, seizure_count, se = (
                    load_channel_data(file_path, row, col)
                )
                if signal is not None:
                    print(f"Signal for channel ({row}, {col}):")
                    print(signal)
            else:
                print("Error occurred during analysis")
