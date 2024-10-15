import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from Intensity_PDF import Wavebounds
from probe_config import CHANNEL_WIDTH, CHANNEL_RANGE
import os

DEFAULT_IRF_BOUNDS = Wavebounds(300, 1000)
DEFAULT_WAVE_STEP = 50

DEFAULT_STARTING_LIFETIME = 2  # ns orignal -> 2
DEFAULT_LIFETIME_STEP = 0.001  # ns original -> 0.001
DEFAULT_LIFETIME_STD = 0.05  # ns original -> 0.05

DEFAULT_SIGMA_PER_CHANNEL = 0.1  # ns

# Assuming this file is located in the src directory
IRF_PATH = "../data/simulation-requirements/wavelengths_mu.csv"


class IRF:
    """
    This class will be used to represent the IRF as a piecewise function of the
    probe


    starting_lifetime : float = the starting lifetime
    wavebounds : Wavebounds = the bounds to the piecewise function
    wavelength_step : float = the step value between the irf jump of the wavelength
    lifetime_step : float = the step in lifetime the irf will take after each step in the wavelength
    lifetime_std : float the std deviation of the lifetime

         y
        ^
        |
     7.5|
        |
     7.0|
        |
     6.5|                                 ------|
        |                            -----|
     6.0|                     -------|
        |               ------|
     5.5|          -----|
        |    -----|
     5.0|___|
        +----------------------------------------> x
     300   400   500   600   700   800   900   1000


    """

    def __init__(
        self,
        wavebounds: Wavebounds = DEFAULT_IRF_BOUNDS,
        wavelength_step: float = DEFAULT_WAVE_STEP,
        starting_lifetime: float = DEFAULT_STARTING_LIFETIME,
        lifetime_step: float = DEFAULT_LIFETIME_STEP,
        lifetime_std: float = DEFAULT_LIFETIME_STD,
        path: str = IRF_PATH,
    ) -> None:
        self.starting_lifetime = starting_lifetime
        self.wavebounds = wavebounds
        self.wavelength_step = wavelength_step
        self.lifetime_step = lifetime_step
        self.lifetime_std = lifetime_std

        self.wavelengths_steps = np.arange(
            self.wavebounds[0],
            self.wavebounds[-1] + self.wavelength_step,
            self.wavelength_step,
        )

        self.channel_wavelengths = np.arange(
            self.wavebounds[0], self.wavebounds[-1], CHANNEL_WIDTH
        )
        # print(self.channel_wavelengths.shape)
        self.path = path

        self.initalise_lifetimes()

    def initalise_lifetimes(self):
        """
        This function will initialise the mu lifetimes of the IRF, if there is no file contain the mu values is found
        it will generate them randomly

        """
        if os.path.exists(self.path):
            self.mu_lifetimes = np.genfromtxt(self.path, delimiter=",")
        else:
            self.mu_lifetimes = np.array(
                [
                    np.random.normal(self.starting_lifetime, self.lifetime_std, 1)[0]
                    + (self.get_step(i) * self.lifetime_step)
                    for i in range(len(self.channel_wavelengths))
                ]
            )
            # self.mu_lifetimes = np.random.normal(self.starting_lifetime, 0.5, len(self.channel_wavelengths))
            np.savetxt(self.path, self.mu_lifetimes, delimiter=",")

    def get_step(self, wavelength: float) -> float:
        step = int(math.floor(wavelength / self.wavelength_step)) * self.wavelength_step
        return step

    def set_irf_intensity_distribution(self, verbose: bool = True):
        """
        This function will set give an intensity distribution to the IRF, return a spline of essentially a straight line

        """

        # Let's evenly space the wavelengths
        x = np.arange(
            self.wavebounds[0],
            self.wavebounds[-1] + self.wavelength_step,
            self.wavelength_step,
        )

        y = np.array([1 / (self.wavebounds[-1] - self.wavebounds[0])] * len(x))
        # y = y - y.mean() / y.std()
        # Now let's create a spline of the intensity
        self.spline = interp.UnivariateSpline(x, y, k=1, s=0)
        if verbose:
            plt.plot(x, y, "ro", ms=5)
            plt.plot(x, self.spline(x), "b-")
            plt.show()

    def mu_lookup(self, index: int) -> float:
        """
        This function will look up the mu value of the IRF at a given wavelength
        """
        return self.mu_lifetimes[index.cpu().numpy()]

    def lookup(self, wavelength: float, num_samples: int = 2_000) -> float:
        wavelength_index = np.round(
            (wavelength - CHANNEL_RANGE[0]) / CHANNEL_WIDTH
        ).astype(int)
        mu = self.mu_lifetimes[wavelength_index]
        return np.random.normal(mu, DEFAULT_SIGMA_PER_CHANNEL, num_samples)

    def get_irf_metadata(self):
        return {
            "starting_lifetime": self.starting_lifetime,
            "wavebounds": self.wavebounds.to_list(),
            "wavelength_step": self.wavelength_step,
            "lifetime_step": self.lifetime_step,
            "lifetime_std": self.lifetime_std,
            "irf": list(
                zip(
                    self.channel_wavelengths.astype(float),
                    self.mu_lifetimes.astype(float),
                )
            ),
        }

    def get_irf_lifetimes(self):
        return list(
            zip(self.channel_wavelengths.astype(float), self.mu_lifetimes.astype(float))
        )

    def plot_irf(self):
        plt.plot(self.channel_wavelengths, self.mu_lifetimes)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Mu to be used for Guassian sampling (ns)")
        plt.show()


if __name__ == "__main__":
    irf = IRF()
    irf.plot_irf()
    irf.set_irf_intensity_distribution()
