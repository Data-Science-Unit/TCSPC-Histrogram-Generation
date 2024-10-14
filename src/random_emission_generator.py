import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline
from scipy.integrate import quad
from Intensity_PDF import Wavebounds
from probe_config import CHANNEL_RANGE, CHANNEL_WIDTH
from typing import Tuple


class Emission_Generator:
    """
    This class will be used to generate random B-Splines from the wavebounds of the TRFS device used in simulation (comes from the probe configuration)
    """

    def __init__(
        self, bounds: Wavebounds = CHANNEL_RANGE, verbose: bool = False
    ) -> None:
        self.bounds = bounds

        self.emission_bounds = None

        # B-Spline, knots, and coefficents
        self.spline = None
        self.t = None
        self.c = None

        self.xs = None
        self.ys = None

        # normalised B-Spline, knots, and coefficents
        self.norm_spline = None
        self.norm_t = None
        self.norm_c = None

        self.norm_xs = None
        self.norm_ys = None

        self.generate_random_emission(verbose=verbose)

    def generate_random_emission(
        self,
        verbose: bool = False,
        num_knots: int = 6,
        degree: int = 3,
        num_points_sampled: int = 10,
        bounds: Tuple[float, float] = None,
    ):
        """
        This function generates a random emission distribution based on the given wavebounds. And will return
        the B-Spline, knots, and coefficients of the B-Spline, the x and y values of the B-Spline, the normalised B-Spline,
        and the normalised x and y values of the B-Spline.

        :return: t, c, xs, ys, norm_t, norm_c, norm_xs, norm_ys : tuple
        """
        if bounds is None:
            bounds = self.bounds

        # Randomly sample bounds to generate the bounds of the spline ensuring that the start < end
        min_wavelength = np.random.uniform(*bounds)
        min_wavelength = np.round(min_wavelength, 2)
        max_wavelength = min_wavelength + np.random.uniform(100, 200)

        # Ensure that the max wavelength is within the bounds
        max_wavelength = min(max_wavelength, (bounds[1] - CHANNEL_WIDTH))
        max_wavelength = np.round(max_wavelength, 2)

        self.emission_bounds = Wavebounds(min_wavelength, max_wavelength)

        # Generate the evenly spaced knots
        homogenous_points = np.linspace(min_wavelength, max_wavelength, num_knots)
        norm_homogenous_points = np.linspace(0, 1, num_knots)
        y_locations = np.random.uniform(0, 1, num_knots)
        y_locations = y_locations / np.sum(y_locations)

        # Generate the B-Spline
        self.spline = make_interp_spline(homogenous_points, y_locations, k=degree)
        scale_factor = quad(self.spline, min_wavelength, max_wavelength)[0]
        y_locations = y_locations / scale_factor
        self.spline = make_interp_spline(homogenous_points, y_locations, k=degree)

        self.t = self.spline.t
        self.c = self.spline.c

        self.xs = np.linspace(min_wavelength, max_wavelength, num_points_sampled)
        self.ys = self.spline(self.xs)

        # Generate the normalised B-Spline equivalent
        self.norm_spline = make_interp_spline(
            norm_homogenous_points, y_locations, k=degree
        )
        norm_scale_factor = quad(self.norm_spline, 0, 1)[0]
        y_locations = y_locations / norm_scale_factor
        self.norm_spline = make_interp_spline(
            norm_homogenous_points, y_locations, k=degree
        )

        self.norm_t = self.norm_spline.t
        self.norm_c = self.norm_spline.c

        self.norm_xs = np.linspace(0, 1, num_points_sampled)
        self.norm_ys = self.norm_spline(self.norm_xs)

        if verbose:
            x = np.linspace(min_wavelength, max_wavelength, 1000)
            norm_x = np.linspace(0, 1, 1000)
            y = self.spline(x)
            norm_y = self.norm_spline(norm_x)

            fig, axs = plt.subplots(2, figsize=(10, 10))
            fig.suptitle("Random Emission Distribution")
            axs[0].plot(x, y, label="Spline")
            axs[0].set_title("Unnormalised Distribution of random emission")
            axs[0].set_xlabel("Wavelength (nm)")
            axs[0].set_ylabel("Intensity (a.u.)")
            axs[0].legend()

            axs[1].plot(norm_x, norm_y, label="Normalised Spline")
            axs[1].set_title("Normalised Distribution of random emission")
            axs[1].set_xlabel("Normalised Wavelength")
            axs[1].set_ylabel("Intensity (a.u.)")
            axs[1].legend()

            plt.show()

        return (
            self.t,
            self.c,
            self.xs,
            self.ys,
            self.norm_t,
            self.norm_c,
            self.norm_xs,
            self.norm_ys,
        )

    def get_emission_metadata(self):
        """
        This function will be used to get the metadata of the emission distribution in the form of a dictionary

        :return: metadata : dict
        """
        return {
            "bounds": self.bounds.to_list(),
            "t": self.t.tolist(),
            "c": self.c.tolist(),
            "xs": self.xs.tolist(),
            "ys": self.ys.tolist(),
            "norm_t": self.norm_t.tolist(),
            "norm_c": self.norm_c.tolist(),
            "norm_xs": self.norm_xs.tolist(),
            "norm_ys": self.norm_ys.tolist(),
        }

    def save_emission_metadata(self, file_name: str, path: str):
        """
        This function will be used to save the metadata of the emission distribution in the form of a npz file

        Note: it will add _emission.npz to the file_name
        """
        os.makedirs(path, exist_ok=True)
        np.savez(f"{path}/{file_name}_emission.npz", **self.get_emission_metadata())

    def load_emission_metadata(self, file_path: str):
        """
        This function will be used to load the metadata of the emission distribution from a npz file
        """
        data = np.load(file_path)
        self.bounds = Wavebounds(*data["bounds"])
        self.t = data["t"]
        self.c = data["c"]
        self.xs = data["xs"]
        self.ys = data["ys"]
        self.norm_t = data["norm_t"]
        self.norm_c = data["norm_c"]
        self.norm_xs = data["norm_xs"]
        self.norm_ys = data["norm_ys"]

    def plot_emission(self):
        plt_xs = np.linspace(*self.emission_bounds, 1000)
        plt_norm_xs = np.linspace(0, 1, 1000)

        plt_ys = self.spline(plt_xs)
        plt_norm_ys = self.norm_spline(plt_norm_xs)

        fig, axs = plt.subplots(2, figsize=(10, 10))
        fig.suptitle("Random Emission Distribution")
        axs[0].plot(plt_xs, plt_ys, label="Spline")
        axs[0].set_title("Unnormalised Distribution of random emission")
        axs[0].set_xlabel("Wavelength (nm)")
        axs[0].set_ylabel("Intensity (a.u.)")
        axs[0].legend()

        axs[1].plot(plt_norm_xs, plt_norm_ys, label="Normalised Spline")
        axs[1].set_title("Normalised Distribution of random emission")
        axs[1].set_xlabel("Normalised Wavelength")
        axs[1].set_ylabel("Intensity (a.u.)")
        axs[1].legend()

        plt.show()


if __name__ == "__main__":
    # Generate a random emission distribution
    emission_generator = Emission_Generator()
    emission_generator.generate_random_emission(verbose=True)
    temp = emission_generator.get_emission_metadata()
    emission_generator.save_emission_metadata("test", "../data/emission_distributions")
    emission_generator.generate_random_emission(verbose=True)
    emission_generator.load_emission_metadata(
        "../data/emission_distributions/test_emission.npz"
    )
    assert temp == emission_generator.get_emission_metadata()
