import numpy as np
from typing import Tuple
import os
import matplotlib.pyplot as plt 
from scipy import signal
from probe_config import CHANNEL_WIDTH, CHANNEL_RANGE




def save_data(
    name: str,
    data: np.ndarray,
    irf_data: np.ndarray,
    data_directory: str = "single-fluoro-test-data",
):
    # data_directory = "single-fluoro-test-data"
    os.makedirs(f"{data_directory}/{name}", exist_ok=True)
    np.savetxt(f"{data_directory}/{name}/{name}_data.csv", data, delimiter=",")
    np.savetxt(f"{data_directory}/{name}/{name}_irf_data.csv", irf_data, delimiter=",")

def save_peak_intensities(name: str, data : np.ndarray, data_directory: str = "single-fluoro-test-data"):
    os.makedirs(f"{data_directory}/", exist_ok=True)
    np.savetxt(f"{data_directory}/{name}_peak_intensities.csv", data, delimiter=",")

def save_bias_data(name: str, bias_data: np.ndarray, data_directory: str = "single-fluoro-test-data"):
    os.makedirs(f"{data_directory}/", exist_ok=True)
    np.savetxt(f"{data_directory}/{name}_bias_data.csv", bias_data, delimiter=",")


def plot_peak_intensity_per_channel(
    data, indicies: Tuple[int, int] = None, name: str = "", show_convole: bool = False, for_irf : bool = False
):
    cropped_data = data
    if indicies:
        cropped_data = data[indicies[0] : indicies[1], :]

    # print(cropped_data)

    peak_intensitys = []
    for i in range(len(cropped_data)):
        row_max_index = np.argmax(cropped_data[i])
        peak_intensitys.append(cropped_data[i][row_max_index])

    peak_intensitys = np.array(peak_intensitys)
    # print(peak_intensitys)
    peak_intensitys_normalised = peak_intensitys / peak_intensitys.max()

    # make a rolling average with window of 3
    peak_intensitys_avg = (
        np.convolve(peak_intensitys_normalised, np.ones(3), "valid") / 3
    )
    spectral_range = np.linspace(300, 1000, 1375)
    if indicies:
        spectral_range = spectral_range[indicies[0] : indicies[1]]
    clipped_spectral_range = spectral_range[:-2]

    plt.figure()
    plt.scatter(spectral_range, peak_intensitys_normalised, s=1)
    if show_convole:
        plt.plot(clipped_spectral_range, peak_intensitys_avg, color="red")
    # plt.xlim(300, 600)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Peak Intensity")
    plt.title(f"Peak Intensity for {name} per Wavelength")
    if for_irf:
        plt.ylim(0, 1)
    plt.show()

def get_max_and_average_peak_intensity_per_channel(data: np.ndarray, indicies: Tuple[int, int] = None) -> Tuple[float, float]:
    """
    This function will return the max value of each channel in the data, and average them whilst returning the highest value
    Returns: Tuple[float, float] -> (max_peak_intensity, average_peak_intensity)
    """
    cropped_data = data
    if indicies:
        cropped_data = data[indicies[0] : indicies[1], :]

    peak_intensitys = []
    for i in range(len(cropped_data)):
        row_max_index = np.argmax(cropped_data[i])
        peak_intensitys.append(cropped_data[i][row_max_index])

    peak_intensitys = np.array(peak_intensitys)
    max_peak_intensity = peak_intensitys.max()
    average_peak_intensity = peak_intensitys.mean()

    return max_peak_intensity, average_peak_intensity

def get_peak_intensity_per_channel(data: np.ndarray, indicies: Tuple[int, int] = None, with_wavelengths : bool = False) -> np.ndarray:
    """
    This function will return the peak value of each channel in the data
    Returns: np.ndarray -> peak_intensitys
    """
    cropped_data = data
    wavelengths_lookup = np.arange(CHANNEL_RANGE[0], CHANNEL_RANGE[1], CHANNEL_WIDTH)
    if indicies:
        cropped_data = data[indicies[0] : indicies[1], :]
        wavelengths_lookup = wavelengths_lookup[indicies[0] : indicies[1]]

    wavelengths = []
    peak_intensitys = []
    for i in range(len(cropped_data)):
        row_max_index = np.argmax(cropped_data[i])
        peak_intensitys.append(cropped_data[i][row_max_index])
        wavelengths.append(wavelengths_lookup[i])

    if with_wavelengths:
        return np.array(wavelengths), np.array(peak_intensitys)    

    return np.array(peak_intensitys)

def data_and_irf_inspection(data: np.ndarray, irf_data : np.ndarray, name: str, index_bounds : Tuple[int, int]):
    indicies_to_inspect = np.linspace(index_bounds[0] + 1, index_bounds[-1] - 1, 4, dtype=int)
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle(f"Data and IRF inspection for {name}")
    for i, ax in enumerate(axs.flat):
        ax.plot(data[indicies_to_inspect[i]], label = name, )
        ax.plot(irf_data[indicies_to_inspect[i]], label = "IRF", alpha=0.5)
        ax.legend()
        ax.set_title(f"Index {indicies_to_inspect[i]}")
        # ax.xlim(100, 200)
        #set ax x limits
        ax.set_xlim(0, 300)

    plt.show()

def single_data_and_irf_inspection(data: np.ndarray, irf_data : np.ndarray, name: str, index : int):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(data[index], label = name)
    ax.plot(irf_data[index], label = "IRF", alpha=0.5)
    ax.legend()
    ax.set_title(f"Index {index}")
    ax.set_xlim(0, 300)
    plt.show()

def find_data_indices(data: np.ndarray) -> Tuple[int, int]:
    """
    Find the start and end indices of the data in the matrix
    """
    #Using the sum of each channel to find the start and end indices
    start_index = 0
    end_index = 0
    for i, row in enumerate(data):
        if np.sum(row) > 0:
            start_index = i
            break
    for i, row in enumerate(data[::-1]):
        if np.sum(row) > 0:
            end_index = len(data) - i
            break
    return start_index, end_index


def convolve_data_with_irf(data: np.ndarray, irf: np.ndarray) -> np.ndarray:
    """
    Convolve the data with the irf
    """
    # conv = []
    # for i, pixel in enumerate(data):
    #     conv.append(
    #         signal.convolve(pixel, irf[i], mode="full", method="direct")[:1200]
    #     )
    # return np.array(conv)
    conv = signal.fftconvolve(data, irf, mode='full', axes=1)[:, :1200]
    return conv

def get_peak_values(data: np.ndarray) -> np.ndarray:
    """
    Get the peak value of each channel
    """
    return np.max(data, axis=1)

def normalise_per_channel(data: np.ndarray) -> np.ndarray:
    """
    Normalise the data along each row
    """
    return data / np.max(data, axis=1)[:, None]

def create_data_using_irf_convole(data: np.ndarray, irf: np.ndarray) -> np.ndarray:
    indices = find_data_indices(data)
    convolved_data = convolve_data_with_irf(data, irf)
    original_peak_values = get_peak_values(data)[indices[0] : indices[1]]
    repeated_peak_values = original_peak_values.repeat(1200).reshape(original_peak_values.shape[0], 1200)
    normalised_data = normalise_per_channel(convolved_data[indices[0] : indices[1]])
    scaled_data = normalised_data * repeated_peak_values

    # plt.figure()
    # plt.plot(scaled_data[210])
    # plt.show()
    convolution = np.zeros((1375, 1200))
    convolution[indices[0] : indices[1]] = scaled_data
    return convolution

def mix_data(fluo_1: np.ndarray, fluo_2: np.ndarray, fluo_3: np.ndarray, irf: np.ndarray,  sensitivity_matrix : np.ndarray) -> np.ndarray:
   mix = fluo_1 + fluo_2 + fluo_3
   convolved_mix = create_data_using_irf_convole(mix, irf)
   convolved_mix = np.round(convolved_mix * sensitivity_matrix)
   return convolved_mix