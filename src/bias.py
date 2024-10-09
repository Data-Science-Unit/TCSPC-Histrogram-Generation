import numpy as np
import matplotlib.pyplot as plt
from Intensity_PDF import Wavebounds
from probe_config import (
    NUM_CHANNELS,
    NUM_BINS,
    CHANNEL_RANGE,
    CHANNEL_WIDTH,
    BIAS_PHOTON_RANGE,
)

from typing import List


class Bias:
    def __init__(
        self,
        intensity_range: Wavebounds,
        num_channels: int = NUM_CHANNELS,
        num_bins: int = NUM_BINS,
        channel_range: Wavebounds = CHANNEL_RANGE,
        channel_width: float = CHANNEL_WIDTH,
    ):
        self.intensity_range = intensity_range
        self.num_channels = num_channels
        self.num_bins = num_bins
        self.channel_range = channel_range
        self.channel_width = channel_width

    def get_intensity_matrix_indicies(self):
        lower_bound = round(
            (self.intensity_range[0] - self.channel_range[0]) / self.channel_width
        )
        upper_bound = round(
            (self.intensity_range[1] - self.channel_range[0]) / self.channel_width
        )

        return lower_bound, upper_bound

    def get_bias_matrix(self, verbose: bool = False) -> np.ndarray:
        background_noise_histogram = np.zeros((self.num_channels, self.num_bins))
        matrix_indicies = self.get_intensity_matrix_indicies()
        lower_bound = matrix_indicies[0]
        upper_bound = matrix_indicies[1]
        num_channels_covered = upper_bound - lower_bound

        poission_bias_mean_per_channel = np.random.uniform(
            *BIAS_PHOTON_RANGE, num_channels_covered
        )

        # map on a poission distribution of size num_bins per channel using the channel value
        samples_per_bin = np.array(
            list(
                map(
                    lambda pois: np.random.poisson(pois, NUM_BINS),
                    poission_bias_mean_per_channel,
                )
            )
        )

        background_noise_histogram[lower_bound:upper_bound, :] = samples_per_bin

        if verbose:
            print(f"shape: {poission_bias_mean_per_channel.shape}")
            print(samples_per_bin)
            print(f"shape: {samples_per_bin.shape}")

        return background_noise_histogram

    def get_metadata(self) -> dict:
        return {
            "intensity_range": self.intensity_range.to_list(),
        }


if __name__ == "__main__":
    x = Wavebounds(300, 1000)
    b = Bias(x)
    b.get_bias_matrix()

    print(f"Shape of bias matrix: {b.get_bias_matrix().shape}")

    bias_indicies = b.get_intensity_matrix_indicies()
    # Plot the bias at the bias indices of the first channel
    # plt.plot(b.get_bias_matrix()[bias_indicies[0]])
    # plt.title("Bias at the first channel")
    # plt.xlabel("Bins")
    # plt.ylabel("Photons")
    # plt.show()

    # Plot across the first bin for all channels for spectral range
    bias_data = b.get_bias_matrix()[bias_indicies[0] : bias_indicies[1]]
    wavelength_range = np.linspace(*CHANNEL_RANGE, b.num_channels)
    wavelength_range = wavelength_range[bias_indicies[0] : bias_indicies[1]]
    plt.figure(figsize=(10, 5))
    plt.plot(wavelength_range, bias_data[:, 0])
    plt.title("Bias at the first bin")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Photons")
    plt.xlim(400, 600)
    plt.show()
