import sys

sys.path.append("../src/")
from Tissue_Fluorophore import *
from probe_config import *
from random_intensity_distrbution import IntensityPDFGenerator
from irf_function import IRF
from enum import Enum
from datetime import datetime
from tqdm import tqdm
import time
from argparse import ArgumentParser

low_photons = 1_000_000
median_photons = 5_000_000
high_photons = 20_000_000

fluoros = ["elastin", "nadh", "flavin"]

biomaker_path = "../data/reference-endo-fluoros-marcu-2014-fig-3-2.png"
nadh_colour = (0, 0, 0)  # roughly black
flavin_colour = (227, 220, 135)  # roughly yellow
elastin_colour = (183, 91, 107)  # roughly red
image_wavebounds = (300, 700)
use_verbose = False

nadh_intensity_pdf = Fluorophore_Intensity_PDF(
    "NADH", biomaker_path, image_wavebounds, nadh_colour, verbose=use_verbose
)

flavin_intensity_pdf = Fluorophore_Intensity_PDF(
    "Flavin", biomaker_path, image_wavebounds, flavin_colour, verbose=use_verbose
)
elastin_intensity_pdf = Fluorophore_Intensity_PDF(
    "Elastin", biomaker_path, image_wavebounds, elastin_colour, verbose=use_verbose
)

fluoro_intensity_pdfs = {
    "nadh": nadh_intensity_pdf,
    "flavin": flavin_intensity_pdf,
    "elastin": elastin_intensity_pdf,
}

SPECTRAL_SENSITIVITY_TYPE = Enum("SPECTRAL_SENSITIVITY_TYPE", ["RED", "BLUE"])
SPECTRAL_SENSITIVITY = SpectralSensitivity()

base_path = "genereated_data/single_fluoro/"


def choose_photon_count() -> int:
    """
    This function is used to choose the photon count for the fluorophore
    :return: int
    """
    return np.random.choice([low_photons, median_photons, high_photons])


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
    return np.random.uniform(0.1, 12)  # ns


def choose_fluoro() -> str:
    return np.random.choice(fluoros)


def generate_fluoro_name(fluoro: str):
    """
    Adds date and time onto fluoro
    """
    date = datetime.now()
    return f"{fluoro}_{date.year}_{date.month}_{date.day}_{date.hour}_{date.minute}_{date.second}_{date.microsecond}"


def tracking_file(file_name: str, photon_count: int):
    with open("samples/single_fluoro/tracking.csv", "a") as file:
        meta_data = f"{file_name}_metadata.json"
        emission_data = f"{file_name}_emission.npz"
        photon_sample_type = (
            "low"
            if photon_count == low_photons
            else "median" if photon_count == median_photons else "high"
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
    # Choose fluoro
    fluoro = choose_fluoro()
    # Choose the photon count
    photon_count = choose_photon_count()
    # Choose the spectral sensitivity
    spectral_sensitivity_bounds, spectral_sensitivity_function = (
        choose_spectral_sensitivity()
    )
    # Generate the lifetime
    lifetime = generate_random_lifetime()
    # Generate the name of the fluorophore
    name = generate_fluoro_name(fluoro)
    # Generate the IRF
    irf = IRF()

    fluo = Tissue_Fluorophore(
        name=name,
        spectral_sensitivity_range=spectral_sensitivity_bounds,
        spectral_sensitivity=spectral_sensitivity_function,
        average_lifetime=lifetime,
        intensity_distribution=fluoro_intensity_pdfs[fluoro].intensity_pdf,
        intensity_range=fluoro_intensity_pdfs[fluoro].wavebounds,
        irf_function=irf.lookup,
        irf_mu_lookup=irf.mu_lookup,
    )

    file_path = base_path + name

    fluo.generate_data(photon_count, use_bias=True)
    if verbose:
        fluo.plot_data(block=True)
    fluo.save_data(file_path, photon_count, irf)
    fluoro_intensity_pdfs[fluoro].save_emission(f"{file_path}_emission.npz")
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
