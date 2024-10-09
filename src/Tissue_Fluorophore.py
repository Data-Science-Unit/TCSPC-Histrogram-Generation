import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import warnings
import scipy.interpolate as interp
import typing
from scipy.integrate import quad
from scipy.signal import convolve2d
from PIL import Image
from tqdm import tqdm
import json
from spectral_sensitivity import SpectralSensitivity
from probe_config import (
    CHANNEL_RANGE,
    NUM_CHANNELS,
    CHANNEL_WIDTH,
    EPISODE_TIME,
    TIME_BIN_WIDTH,
    NUM_BINS,
    get_probe_metadata,
)
from Intensity_PDF import Wavebounds, Fluorophore_Intensity_PDF
from irf_function import IRF, DEFAULT_SIGMA_PER_CHANNEL
from bias import Bias
import torch
from time import time
import os
import psutil
from memory_profiler import profile


class Tissue_Fluorophore:
    """
    This class will be used to generate data for Endogenous fluorophores:

    intensity_distribution: should be an interpolation or spline function of the normalise PDF of the fluorophores intensity across its emission spectra
    intensity_range: the range of intensities that the fluorophore can have

    average_lifetime : the average lifetime of the fluorophore

    irf_function : Takes a wavelength (tau) and outputs the irf for given channel
        |-> Use the IRF class

    name : the name of the fluorophore

    ------------------------------ These settings by default will come from probe_config.py file ------------------------------
    num_channels : the number of channels in the simulated data
    channel_range : the range of the channels

    total_time : the total time of the simulation in ns
    num_bins : the number of bins in the simulation


    wave_samples_ratio : the will be the split between the wavelength and time samples i.e. (0.8 wavelength and (1 - 0.8) lifetime samples)
        |->  this is due to the nature of the inverse transform sampling

    bias_ratio : this will be the ratio between the number of samples used to for the fluorophore generation and the background noise
        |-> e.g. a value of 0.1 means that 10% of the number of samples entered will be background noise
        |-> If using a different bias function than the standard -> ensure it returns a matrix of the same size as the histogram
    """

    def __init__(
        self,
        intensity_distribution: interp.UnivariateSpline,
        intensity_range: Wavebounds,
        average_lifetime: float,
        spectral_sensitivity: interp.interp1d,
        spectral_sensitivity_range: Wavebounds,
        irf_function: typing.Callable[[float], float],
        irf_mu_lookup: typing.Callable[[int], float],
        name: str,
        num_channels: int = NUM_CHANNELS,
        channel_range: Wavebounds = CHANNEL_RANGE,
        total_time: float = EPISODE_TIME,
        num_bins: int = NUM_BINS,
    ):
        self.name = name

        # CONFIG
        # Intensity Parameters
        self.intensity_distribution = intensity_distribution
        self.intensity_range = intensity_range
        self.inverse_cdf_intensity = None

        # Lifetime Parameters
        self.average_lifetime = average_lifetime

        # Sensitivity Parameters
        self.spectral_sensitivity = spectral_sensitivity
        self.spectral_sensitivity_range = spectral_sensitivity_range
        self.sensitivity_matrix = None

        # IRF Parameters
        self.irf_function = irf_function
        self.irf_mu_lookup = irf_mu_lookup
        self.irf_wave = None

        # Hisorgram parameters
        #   | Wavelength parameters
        self.num_channels = num_channels
        self.channel_range = channel_range
        self.channel_width = CHANNEL_WIDTH

        #   | Time parameters
        self.total_time = total_time
        self.num_bins = num_bins
        self.bin_width = TIME_BIN_WIDTH

        # Initalising the empty histogram
        self.data = np.zeros((self.num_channels, self.num_bins))

        # Data generation parameters
        self.wave_samples_ratio = 0.8

        # Bias paramters

        self.bias = Bias(
            self.intensity_range,
            num_channels=num_channels,
            num_bins=num_bins,
        )
        self.bias_matrix = None

        # Multiprocessing paramters
        self.num_threads = mp.cpu_count()

        # END CONFIG
        # Generate the inverse intensity CDF
        self.obtain_inverse_cdf()

    def intensity_cdf(self, intensity: float) -> float:
        """
        The CDF of the intensity distribution, acts as a lookup table for the CDF
        """
        cdf, err = quad(self.intensity_distribution, self.intensity_range[0], intensity)
        return cdf

    def obtain_inverse_cdf(self):
        """
        Obtain the inverse CDF of the intensity distribution
        """
        x = np.linspace(self.intensity_range[0], self.intensity_range[1], 1000)
        y = np.array([self.intensity_cdf(i) for i in x])
        cdf_values = (y - np.min(y)) / (np.max(y) - np.min(y))
        self.inverse_cdf_intensity = interp.interp1d(
            cdf_values, x, bounds_error=False, fill_value="extrapolate"
        )

        # plt.plot(cdf_values, x)
        # plt.show()

    # @profile
    def extract_irf_wave(
        self,
        verbose: bool = False,
        use_sensitivity_matrix: bool = False,
        use_bias_matrix: bool = False,
        bounds=Wavebounds(300, 1000),
        irf_lifetime: float = 0,
    ):
        """
        This function will extract the IRF for the fluorophore, and return the IRF for the fluorophore
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # irf_lifetime = 0.05  # ns -> meaning 50 ps
        num_wavelength_samples_per_channel = 5_000
        num_wavelength_samples = (
            self.num_channels
        ) * num_wavelength_samples_per_channel
        # irf_emission_specra = np.linspace(*self.channel_range, self.num_channels)[:-1]
        irf_emission_specta = torch.linspace(
            *self.channel_range, self.num_channels, device=device
        )[:-1]

        num_lifetime_samples_per_emission = 1
        inverse_cdf_lifetime_samples = torch.rand(
            num_wavelength_samples * num_lifetime_samples_per_emission,
            device=device,
            dtype=torch.float64,
        )
        # inverse_cdf_lifetime_samples = torch.linspace(0.001, 1, num_lifetime_samples_per_emission, device=device).repeat(num_wavelength_samples)

        inverse_cdf_lifetime_samples = torch.log(1 - inverse_cdf_lifetime_samples)
        inverse_cdf_lifetime_samples = (-irf_lifetime) * inverse_cdf_lifetime_samples

        irf_emission_samples = irf_emission_specta.repeat_interleave(
            num_wavelength_samples_per_channel
        )

        irf_wavelengths = torch.tensor(irf_emission_samples, device=device)
        # irf_wavelength_indicies = torch.round(
        #     (irf_wavelengths - self.channel_range[0]) / self.channel_width
        # ).to(torch.int64)
        irf_wavelength_indicies = (
            torch.arange(0, self.num_channels, device=device)
            .repeat_interleave(num_wavelength_samples_per_channel)
            .to(torch.int64)
        )
        irf_mu_values = torch.as_tensor(
            self.irf_mu_lookup(irf_wavelength_indicies), device=device
        ).repeat_interleave(num_lifetime_samples_per_emission)
        if verbose:
            print(f"IRF mu values: {irf_mu_values}")
        irf_delay_times = torch.normal(irf_mu_values, DEFAULT_SIGMA_PER_CHANNEL)

        print(f"Shape of irf delay: {irf_delay_times.shape}")
        print(
            f"Shape of inverse cdf lifetime samples: {inverse_cdf_lifetime_samples.shape}"
        )

        # Testing the irf convolution theory: for now will turn off irf delay

        inverse_cdf_lifetime_samples += irf_delay_times
        wavelength_samples = irf_emission_samples
        lifetime_samples = inverse_cdf_lifetime_samples

        # CONVERTING GENEREATE DATA TO HISTOGRAM INDICES
        wavelength_channel_indicies = torch.round(
            (wavelength_samples - self.channel_range[0]) / self.channel_width
        ).to(torch.int64)
        possible_wavelength_indicies = set(range(0, self.num_channels))

        if verbose:
            plt.figure()
            plt.hist(wavelength_channel_indicies.cpu().numpy(), bins=1375)
            plt.title("Wavelength samples")
            plt.show()

        wavelength_channel_indicies = wavelength_channel_indicies.repeat_interleave(
            num_lifetime_samples_per_emission
        )
        wavelength_channel_indicies = torch.arange(
            0, self.num_channels, device=device
        ).repeat_interleave(num_wavelength_samples_per_channel)
        lifetime_bin_indices = torch.round(lifetime_samples / self.bin_width).to(
            torch.int64
        )
        missing_indexes = possible_wavelength_indicies - set(
            wavelength_channel_indicies.cpu().numpy()
        )
        print(f"Missing indexes: {missing_indexes}")
        print(f"Wavelength channel indicies: {wavelength_channel_indicies}")
        print(
            f"Unique wavelength indcies {len(torch.unique(wavelength_channel_indicies))}"
        )

        flaten_indices = (
            wavelength_channel_indicies * self.num_bins
        ) + lifetime_bin_indices
        histogram_indices = torch.bincount(
            flaten_indices, minlength=self.num_channels * self.num_bins
        )
        irf_wave = (
            histogram_indices.view(self.num_channels, self.num_bins).cpu().numpy()
        )

        # Extracting the IRF

        if use_sensitivity_matrix and self.sensitivity_matrix is not None:
            irf_wave = np.round(irf_wave * self.sensitivity_matrix).astype(int)
        if use_bias_matrix and self.bias_matrix is not None:
            irf_wave = irf_wave + self.bias_matrix

        self.irf_wave = irf_wave
        return self.irf_wave

    # @profile
    def generate_data(
        self,
        total_samples: int,
        use_spectral_sensitivity: bool = True,
        use_bias: bool = True,
        add_irf: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        This function is used to generate the data for the fluorophore, when entering the desired number of samples
        I would suggest having a higher proportion of wavelength samples than time samples

        The data generation process uses INVERSE TRANSFORM SAMPLING

        The function will generate the data, and update the objects internal histogram to always have the most
        up-to-date histogram, it will return the histogram generated

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # BIAS
        if use_bias:
            bias_matrix = np.round(self.bias.get_bias_matrix()).astype(int)
            self.bias_matrix = bias_matrix

        # SPILTTING SAMPLES
        NUM_TIME_SAMPLES_PER_CHANNEL_SAMPLE = 1

        # DATA GENERATION PROCESS
        # total_samples = num_wavelength_samples * num_time_samples
        samples_shape = (total_samples, NUM_TIME_SAMPLES_PER_CHANNEL_SAMPLE)

        # Generating wavelength samples i.e. this will give the intensity characterist of the histogram
        inverse_cdf_wave_samples = np.random.uniform(0, 1, total_samples)
        wavelength_samples = self.inverse_cdf_intensity(1 - inverse_cdf_wave_samples)
        wavelength_samples = np.clip(wavelength_samples, *self.intensity_range)

        # Generating the lifetime samples
        # F^-1(U) = - 1/tau ln(1 - U) where U is a uniform random variable
        # There lifetime samples is of size num_wavelength_samples * num_time_samples as it
        # samples the lifetime for each wavelength sample
        inverse_cdf_lifetime_samples = np.random.uniform(0, 1, total_samples)
        inverse_cdf_lifetime_samples = np.log(1 - inverse_cdf_lifetime_samples)
        tau_values = self.average_lifetime

        # Generating IRF

        irf_wavelengths = torch.tensor(wavelength_samples, device=device)
        irf_wavelength_indicies = torch.round(
            (irf_wavelengths - self.channel_range[0]) / self.channel_width
        ).to(torch.int64)
        irf_mu_values = torch.as_tensor(
            self.irf_mu_lookup(irf_wavelength_indicies), device=device
        ).repeat_interleave(NUM_TIME_SAMPLES_PER_CHANNEL_SAMPLE)
        if verbose:
            print(f"IRF mu values: {irf_mu_values}")
        irf_delay_times = torch.normal(irf_mu_values, DEFAULT_SIGMA_PER_CHANNEL)

        # Calculating the lifetime values for each of the uniform numbers
        lifetime_samples = (-tau_values) * inverse_cdf_lifetime_samples

        if add_irf:
            lifetime_samples += irf_delay_times.cpu().numpy()

        wavelength_samples = torch.tensor(wavelength_samples, device=device)
        lifetime_samples = torch.tensor(lifetime_samples, device=device)

        # CONVERTING GENEREATE DATA TO HISTOGRAM INDICES
        wavelength_channel_indicies = torch.round(
            (wavelength_samples - self.channel_range[0]) / self.channel_width
        ).to(torch.int64)
        wavelength_channel_indicies = wavelength_channel_indicies.repeat_interleave(
            NUM_TIME_SAMPLES_PER_CHANNEL_SAMPLE
        )

        lifetime_bin_indices = (
            torch.floor((lifetime_samples / self.total_time) * self.num_bins)
            .to(torch.int64)
            .view(-1)
        )

        # Flatten the indices
        flat_indices = (
            wavelength_channel_indicies * self.num_bins
        ) + lifetime_bin_indices
        histogram_indices = torch.bincount(
            flat_indices, minlength=self.num_channels * self.num_bins
        )
        if verbose:
            print(f"Max index: {torch.max(flat_indices)}")
        data = histogram_indices.view(self.num_channels, self.num_bins).cpu().numpy()

        # LOGGING INFO
        if verbose:
            print(
                f"Config for {self.name} : lifetime_samples: {NUM_TIME_SAMPLES_PER_CHANNEL_SAMPLE:_}, wavelength_samples: {total_samples:_} with lifetime of {self.average_lifetime}ns and std of {self.std_lifetime}ns"
            )
            print(f"Total samples: {total_samples:_}")

        # CONVERTING TO HISTOGRAM
        self.data = data

        if use_spectral_sensitivity:
            sensitivity_matrix = self.get_spectral_sensitivity_matrix()
            self.sensitivity_matrix = sensitivity_matrix
            self.data = np.round(self.data * sensitivity_matrix).astype(int)
        self.data = self.data + bias_matrix if use_bias else self.data
        return self.data

    def sample_emission_spectral(self, num_samples: int) -> np.ndarray:
        """
        This function will sample the emission spectra of the fluorophore

        Parameters:
            num_samples : the number of samples to take
        Return:
            histogram of number of channels with samples aggregated
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        uniform_numbers = np.random.uniform(0, 1, num_samples)
        wavelength_samples = self.inverse_cdf_intensity(1 - uniform_numbers)
        wavelength_samples = torch.as_tensor(wavelength_samples, device=device)

        wavelength_channel_indicies = torch.round(
            (wavelength_samples - self.channel_range[0]) / self.channel_width
        ).to(torch.int64)

        histogram_indices = torch.bincount(
            wavelength_channel_indicies, minlength=self.num_channels
        )
        return histogram_indices.cpu().numpy()

    def sample_life(
        self,
        num_samples: int,
        lifetime: float,
        with_irf: bool = False,
        with_bias: bool = False,
    ) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bias_matrix = np.round(self.bias.get_bias_matrix()).astype(int)
        # self.bias_matrix = bias_matrix

        wavesample = np.random.uniform(0, 1, 1)
        wavelength = self.inverse_cdf_intensity(1 - wavesample)
        wavelength_index = np.round(
            (wavelength - self.channel_range[0]) / self.channel_width
        ).astype(np.int8)

        irf_delay = self.irf_function(wavelength, num_samples=1)

        uniform_samples = np.random.uniform(0, 1, num_samples)
        inverse_cdf_lifetime_samples = np.log(1 - uniform_samples)
        lifetime_samples = (-lifetime) * inverse_cdf_lifetime_samples
        if with_irf:
            lifetime_samples += irf_delay

        lifetime_samples = torch.as_tensor(lifetime_samples, device=device)
        lifetime_bin_indices = (
            torch.floor((lifetime_samples / self.total_time) * self.num_bins)
            .to(torch.int64)
            .view(-1)
        )
        data = torch.bincount(lifetime_bin_indices, minlength=self.num_bins)[
            : self.num_bins
        ]
        # crop data to num bins
        # data = data[:self.num_bins]
        data = data.cpu().numpy()

        if with_bias:
            data = data + bias_matrix[wavelength_index].flatten()

        return data

    def get_spectral_sensitivity_matrix(self) -> np.ndarray:
        """
        This function is used to obtain a matrix, of the same size as the data histogram,
        describing the probability of a photon being detect at a given wavelength
        """
        print(f"Sensitivity Range: {self.spectral_sensitivity_range}")
        print(f"Channel Range: {self.channel_range}")
        # assert self.spectral_sensitivity_range[0] <= self.channel_range[0], "the spectral sensitivty (LOWER BOUND) range does not cover the fluorohpores range"
        # assert self.spectral_sensitivity_range[1] >= self.channel_range[1], "the spectral sensitivty (UPPER BOUND) range does not cover the fluorophores range"
        if (self.spectral_sensitivity_range[0] > self.channel_range[0]) or (
            self.spectral_sensitivity_range[1] < self.channel_range[1]
        ):
            warnings.warn(
                "The channel range bounds are greater than the spectral sensitivty range, the bounds of the spectral sensitivity will be extended"
            )

        sensor_range = np.linspace(*self.channel_range, self.num_channels)
        sensor_sensitivity = np.array(
            [self.spectral_sensitivity(i) for i in sensor_range]
        )
        sensitivity_matrix = sensor_sensitivity.repeat(self.num_bins).reshape(
            self.num_channels, self.num_bins
        )
        return sensitivity_matrix

    def plot_data(
        self,
        data_used: np.ndarray = np.array([]),
        time_range: typing.Tuple[float, float] = (0, 30),
        view_entire_spectral_range: bool = False,
        name: str = None,
        block: bool = False,
    ):
        spectral_range = np.linspace(
            self.channel_range[0], self.channel_range[1], self.num_channels
        )
        data_used = self.data.copy() if len(data_used) == 0 else data_used
        name = self.name if name is None else name
        wavelength_range = np.linspace(*self.channel_range, self.num_channels)
        if not view_entire_spectral_range:
            lower_bound = np.where(
                np.isclose(spectral_range, self.intensity_range[0], atol=1)
            )[0][0]
            upper_bound = np.where(
                np.isclose(spectral_range, self.intensity_range[1], atol=1)
            )[0][-1]
            # cropping data
            data_used = data_used[lower_bound:upper_bound, :]
            # wavelength_range
            wavelength_range = np.array(
                [
                    0.509 * i + self.intensity_range[0]
                    for i in np.arange(1, upper_bound - lower_bound + 1, 1)
                ]
            )

        starting_time = int(np.floor((time_range[0] / self.total_time) * self.num_bins))
        ending_time = int(np.floor((time_range[1] / self.total_time) * self.num_bins))

        # print(f"Starting time: {starting_time}, Ending time: {ending_time}")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        X = np.linspace(0, self.total_time, self.num_bins)[starting_time:ending_time]
        # X = np.arange(0, self.num_bins)
        Y = wavelength_range
        X, Y = np.meshgrid(X, Y)

        ax.plot_surface(X, Y, data_used[:, starting_time:ending_time], cmap="coolwarm")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_zlabel("Counts")
        ax.set_title(f"Histogram for {name}")
        # ax.set_title(f"Bias matrix for {name}")
        # plt.xlim(0, 30)

        plt.show(block=block)

    def save_data(self, file_name: str, num_samples: int, irf: IRF):
        """
        This function will save the histogram data, along with the lifetime, emission spectra and IRF function as a csv file
        """
        # write the histogram data as csv
        # np.savetxt(f"{file_name}.csv", self.data, delimiter=",")
        np.savez_compressed(f"{file_name}.npz", self.data)
        print(f"Data saved as {file_name}")
        # write the lifetime, emission spectra and IRF function as a csv
        irf_meta = irf.get_irf_metadata()
        probe_metadata = get_probe_metadata()
        metadata = {
            "num_samples": int(num_samples),
            "probe_config": probe_metadata,
            "fluorophore": self.n_fluorophore_get_metadata(),
            "irf": irf_meta,
        }
        print(num_samples)
        with open(f"{file_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=3)
        print("Metadata saved")

    def n_fluorophore_get_metadata(self) -> dict:
        """
        This function will return the metadata for the fluorophore
        """
        metadata = {
            "name": self.name,
            "avg_lifetime": self.average_lifetime,
            "lifetime_std": self.std_lifetime,
            "emission_spectra": self.intensity_range.to_list(),
        }
        return metadata


if __name__ == "__main__":
    biomaker_path = (
        "../data/simulation-requirements/reference-endo-fluoros-marcu-2014-fig-3-2.png"
    )
    image_wavebounds = (300, 700)

    elastin_colour = (183, 91, 107)  # roughly red
    nadh_colour = (0, 0, 0)
    elastin_intensity = Fluorophore_Intensity_PDF(
        "Elastin", biomaker_path, image_wavebounds, elastin_colour
    )

    # print(f"Elastin intensity bounds: {elastin_intensity.wavebounds}")
    nadh_intensity_pdf = Fluorophore_Intensity_PDF(
        "NADH", biomaker_path, image_wavebounds, nadh_colour
    )

    nadh_avg_lifetime = 2.3
    nadh_std_lifetime = 0.3

    elastin_avg_lifetime = 5.8
    elastin_std_lifetime = 0.5

    spectral_sensitivity = SpectralSensitivity(
        blue_pde_range=(0, 1), red_pde_range=(0, 1)
    )

    irf = IRF()

    elastin = Tissue_Fluorophore(
        elastin_intensity.intensity_pdf,
        elastin_intensity.wavebounds,
        elastin_avg_lifetime,
        spectral_sensitivity.red_spad_sensitivity,
        spectral_sensitivity.blue_spad_range,
        irf.lookup,
        irf.mu_lookup,
        "Elastin",
    )

    nadh = Tissue_Fluorophore(
        nadh_intensity_pdf.intensity_pdf,
        nadh_intensity_pdf.wavebounds,
        nadh_avg_lifetime,
        spectral_sensitivity.red_spad_sensitivity,
        spectral_sensitivity.blue_spad_range,
        irf.lookup,
        irf.mu_lookup,
        "NADH",
    )

    print(f"Emission spectra for NAHD: {nadh_intensity_pdf.wavebounds}")

    # time the data generation

    num_photon = 10_000_000
    nadh.generate_data(num_photon, use_bias=True)
    nadh.plot_data(view_entire_spectral_range=False, block=True)
