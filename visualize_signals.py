import matplotlib.pyplot as plt
import numpy as np
from dataIO import (
    get_baseline,
    get_active_channels,
    load_channel_data,
)


def plot_signal(signal, seizures, recording_length):
    # Create the plot
    plt.figure("Signal with seizures")
    plt.title("Signal with seizures")

    time_vector = np.linspace(0, recording_length, len(signal))

    # Plot the signal
    plt.plot(signal)

    # for seizure in seizures:
    #     if len(seizure) == 3:
    #         start = seizure[0]
    #         stop = seizure[1]
    #
    #     plt.axvspan(start, stop, color="red", alpha=0.5)

    # Show the plot
    plt.show()


def plot_signal_without_seizures(signal, seizures, recording_length):
    # Get the signal without seizures
    signal_without_seizures, new_signal_length = get_baseline(
        signal, seizures, recording_length
    )
    time_vector = np.linspace(0, recording_length, new_signal_length)

    # Create the plot
    plt.figure("Signal without seizures")
    plt.title("Signal without seizures")

    # Plot the signal
    plt.plot(time_vector, signal_without_seizures)

    # Show the plot
    plt.show()
