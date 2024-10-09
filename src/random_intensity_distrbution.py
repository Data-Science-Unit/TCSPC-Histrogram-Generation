import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from Intensity_PDF import Wavebounds
from probe_config import CHANNEL_RANGE
from typing import Tuple


class IntensityPDFGenerator:
    """
    This class is used to generate a random intensity distribution for generating synthetic fluorophores
    """

    def __init__(self, bounds: Wavebounds = CHANNEL_RANGE) -> None:
        self.bounds = bounds
        self.x_control_points = None
        self.y_control_points = None
        self.strictness = None

        self.random_intensity_distribution(verbose=True)

    def random_intensity_distribution(self, verbose: bool = False) -> None:
        """
        This function generates a random intensity distribution based on the given wavebounds.
        :return: bounds of the spline, spline : tuple
        """
        start_wavelength = np.random.uniform(self.bounds[0], self.bounds[1])
        end_wavelength = start_wavelength + np.random.uniform(25, 200)

        # Ensure that the end wavelength is within the self.bounds
        if end_wavelength > self.bounds[1]:
            end_wavelength = self.bounds[1]

        num_control_points = 10  # np.random.randint(4, 10)
        degree = 4

        num_knots = 5
        knots = np.sort(
            np.random.uniform(
                start_wavelength + 0.5 * np.abs(start_wavelength - end_wavelength),
                end_wavelength - 0.5 * np.abs(start_wavelength - end_wavelength),
                num_knots,
            )
        )

        # Generate random control points that have a minimum distance between them to ensure the spline is
        # not jumping beyond 0 and 1
        self.x_control_points = np.linspace(
            start_wavelength, end_wavelength, num_control_points
        )

        self.y_control_points = np.random.uniform(0, 1, num_control_points)
        # Normalize the control points
        self.y_control_points = self.y_control_points / np.sum(self.y_control_points)
        self.strictness = np.random.uniform(0, 0.1)
        self.spline = interp.UnivariateSpline(
            self.x_control_points,
            self.y_control_points,
            s=self.strictness,
            k=degree,
        )

        # self.lsq = interp.LSQUnivariateSpline(
        #     self.x_control_points,
        #     self.y_control_points,
        #     t=knots,
        # )

        self.spline_bounds = Wavebounds(start_wavelength, end_wavelength)

        if verbose:
            self.plot_intensity_distribution()
        # Return the wavebounds and the spline

    def get_intensity_distribution(self) -> interp.UnivariateSpline:
        return self.spline

    def get_intensity_bounds(self) -> Wavebounds:
        return self.spline_bounds

    def plot_intensity_distribution(self) -> None:
        """
        This function plots the intensity distribution
        """
        print(f"Strictness: {self.strictness}")
        print(f"Start Wavelength: {self.spline_bounds[0]}")
        print(f"End Wavelength: {self.spline_bounds[-1]}")

        print(f"Number of knots : {self.spline.get_knots().shape}")
        print(f"Number of parameters: {self.spline.get_coeffs().shape}")

        eval_x = np.linspace(self.spline_bounds[0], self.spline_bounds[-1], 1000)
        eval_y = self.spline(eval_x)

        b_spline = self.convert_to_BSpline()
        eval_y_bspline = b_spline(eval_x)

        plt.plot(
            self.x_control_points,
            self.y_control_points,
            "ro",
            label="Control Points",
        )
        plt.plot(
            eval_x,
            eval_y,
            label="Adjusted Spline",
        )
        plt.plot(
            eval_x,
            eval_y_bspline,
            "--",
            label="B-Spline",
        )
        # plt.plot(
        #     eval_x,
        #     self.lsq(eval_x),
        #     label="LSQ Spline",
        # )
        plt.title("Intensity Distribution")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()

    def save_intensity_distribution(self, filename: str) -> None:
        """
        This function saves the intensity distribution to the given file name
        :param filename: str -> this should end with .npz
        """
        np.savez(
            filename,
            x_control_points=self.x_control_points,
            y_control_points=self.y_control_points,
            strictness=self.strictness,
        )

        print(f"Intensity distribution saved to {filename}")

    def load_intensity_distribution(self, filename: str) -> None:
        """
        This function loads the intensity distribution from the given file name
        :param filename: str -> this should end with .npz
        """
        data = np.load(filename)
        self.x_control_points = data["x_control_points"]
        self.y_control_points = data["y_control_points"]
        self.strictness = data["strictness"]

        self.spline = interp.UnivariateSpline(
            self.x_control_points, self.y_control_points, s=self.strictness
        )

        self.spline_bounds = Wavebounds(
            self.x_control_points[0], self.x_control_points[-1]
        )

    def convert_to_BSpline(self):
        """
        This function converts the univariate spline to a B-spline
        """
        k = self.spline._data[5]
        t = self.spline.get_knots()
        t = np.concatenate(([t[0]] * k, t, [t[-1]] * k))
        c = self.spline.get_coeffs()

        return interp.BSpline(t, c, k)


if __name__ == "__main__":
    ig = IntensityPDFGenerator()
    # ig.save_intensity_distribution("test.npz")
    # ig.load_intensity_distribution(
    #     "samples/single_fluoro/Fluorophore_2024_3_14_18_38_58_352946_emission.npz"
    # )
    # ig.plot_intensity_distribution()
