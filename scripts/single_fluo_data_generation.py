from Tissue_Fluorophore import *
from probe_config import *
from random_intensity_distrbution import IntensityPDFGenerator
from irf_function import IRF
from enum import Enum
from datetime import datetime
from tqdm import tqdm
import time
from argparse import ArgumentParser

SINGLE_LOW_PHOTONS = 50_000_000
SINGLE_MEDIAN_PHOTONS = 100_000_000

SINGLE_HIGH_PHOTONS = 200_000_000

SPECTRAL_SENSITIVITY_TYPE = Enum("SPECTRAL_SENSITIVITY_TYPE", ["RED", "BLUE"])
SPECTRAL_SENSITIVITY = SpectralSensitivity()

base_path = "samples/single_fluoro/"


def choose_photon_count() -> int:
    """
    This function is used to choose the photon count for the fluorophore
    :return: int
    """
    return np.random.choice(
        [SINGLE_LOW_PHOTONS, SINGLE_MEDIAN_PHOTONS, SINGLE_HIGH_PHOTONS]
    )


def choose_spectral_sensitivity() -> tuple[Wavebounds, interp.interp1d]:
    """
    This function is used to choose the spectral sensitivity for the fluorophore
    :return: SPECTRAL_SENSITIVITY_TYPE
    """
    choice = np.random.choice(
        [SPECTRAL_SENSITIVITY_TYPE.RED, SPECTRAL_SENSITIVITY_TYPE.BLUE]
    )
    if choice == SPECTRAL_SENSITIVITY_TYPE.RED:
        return (
            SPECTRAL_SENSITIVITY.red_spad_range,
            SPECTRAL_SENSITIVITY.red_spad_sensitivity,
        )
    else:
        return (
            SPECTRAL_SENSITIVITY.blue_spad_range,
            SPECTRAL_SENSITIVITY.blue_spad_sensitivity,
        )


def generate_random_lifetime() -> float:
    """
    This function is used to choose the lifetime for the fluorophore
    for now this should be between 0.01 and 15ns
    :return: float
    """
    return np.random.uniform(0.01, 15)  # ns


def generate_fluoro_name() -> str:
    """
    This function is used to generate the name of the fluorophore
    :return: str
    """
    # make id from date and time
    date = datetime.now()
    return f"Fluorophore_{date.year}_{date.month}_{date.day}_{date.hour}_{date.minute}_{date.second}_{date.microsecond}"


def tracking_file(file_name: str, photon_count: int):
    with open("samples/single_fluoro/tracking.csv", "a") as file:
        meta_data = f"{file_name}_metadata.json"
        emission_data = f"{file_name}_emission.npz"
        photon_sample_type = (
            "low"
            if photon_count == SINGLE_LOW_PHOTONS
            else "median" if photon_count == SINGLE_MEDIAN_PHOTONS else "high"
        )
        file.write(
            f"{file_name}.npz,{meta_data},{emission_data},{photon_sample_type},{photon_count}\n"
        )


def generate_single_fluoro_data(verbose: bool = False):
    """
    This function is used to generate a single fluorophore data set
    :return: Fluorophore
    """
    # ------------------------------ Fluorophore Setting ------------------------------
    # Choose the photon count
    photon_count = choose_photon_count()
    # Choose the spectral sensitivity
    spectral_sensitivity_bounds, spectral_sensitivity_function = (
        choose_spectral_sensitivity()
    )
    # Generate the lifetime
    lifetime = generate_random_lifetime()
    # Generate the name of the fluorophore
    name = generate_fluoro_name()
    # Generate the intensity distribution
    intensity_distribution = IntensityPDFGenerator()
    # Generate the IRF
    irf = IRF()

    fluo = Tissue_Fluorophore(
        name=name,
        spectral_sensitivity_range=spectral_sensitivity_bounds,
        spectral_sensitivity=spectral_sensitivity_function,
        average_lifetime=lifetime,
        intensity_distribution=intensity_distribution.get_intensity_distribution(),
        intensity_range=intensity_distribution.get_intensity_bounds(),
        irf_function=irf.lookup,
    )

    file_path = base_path + name

    fluo.generate_data(photon_count, use_bias=True)
    if verbose:
        fluo.plot_data(block=True)
    fluo.save_data(file_path, photon_count, irf)
    intensity_distribution.save_intensity_distribution(f"{file_path}_emission.npz")
    tracking_file(name, photon_count)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--verbose",
        help="This flag is used to enable verbose mode",
        default=False,
    )
    parser.add_argument(
        "--test", help="This flag is used to enable test mode", default=False
    )
    args = parser.parse_args()
    verbose = args.verbose
    test = True if args.test == "True" else False
    # Genereate 10_000 single fluorophore data sets
    if not test:
        for i in tqdm(range(200_000)):
            try:
                generate_single_fluoro_data(verbose=verbose)
            except Exception as e:
                continue
            finally:
                time.sleep(1)
    else:
        generate_single_fluoro_data(verbose=verbose)
