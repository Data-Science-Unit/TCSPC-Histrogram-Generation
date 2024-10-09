import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from PIL import Image
from Intensity_PDF import Wavebounds

# original red pde range = (0, 7)
# original blue pde range = (0, 17.5)


class SpectralSensitivity:
    """
    This class is used to represent the sensitivity of the SPADs used in the
    the in house probe at the University of Edinburgh.
    """

    def __init__(
        self,
        red_pde_range=(0, 1),
        blue_pde_range=(0, 1),
        path="../data/simulation-requirements/pde-spad-sensitivity.png",
    ):
        self.spad_figure_emission_range = (300, 1100)
        self.red_pde_range = red_pde_range
        self.blue_pde_range = blue_pde_range

        self.border_color = (92, 92, 92)

        self.red_spad_color = (217, 88, 41)  # red taken from the image
        self.blue_spad_color = (19, 114, 185)

        self.spad_sensitivity_path = path
        self.spad_sensitivity = Image.open(self.spad_sensitivity_path)
        self.spad_sensitivity = self.spad_sensitivity.convert("RGB")

        self.spad_sens_pixels = self.extract_image_from_border(
            self.spad_sensitivity, self.border_color
        )

        (
            self.red_spad_range,
            self.red_spad_sensitivity,
        ) = self.extract_sensitivity_lookup(
            self.red_spad_color,
            self.spad_sens_pixels,
            self.red_pde_range,
            self.spad_figure_emission_range,
        )

        (
            self.blue_spad_range,
            self.blue_spad_sensitivity,
        ) = self.extract_sensitivity_lookup(
            self.blue_spad_color,
            self.spad_sens_pixels,
            self.blue_pde_range,
            self.spad_figure_emission_range,
        )

        # lets plot the sensitivity curves
        self.red_range = np.linspace(
            *self.red_spad_range,
            1000,
        )
        self.blue_range = np.linspace(
            *self.blue_spad_range,
            1000,
        )

    def extract_image_from_border(
        self, image, border_color: tuple[int, int, int], threshold: int = 10
    ):
        width, height = image.size
        pixels = image.load()

        top, left = width, height
        bottom, right = 0, 0

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                if (
                    abs(r - border_color[0]) < threshold
                    and abs(g - border_color[1]) < threshold
                    and abs(b - border_color[2]) < threshold
                ):
                    top = min(top, y)
                    left = min(left, x)
                    bottom = max(bottom, y)
                    right = max(right, x)

        return image.crop((left, top, right, bottom))

    def extract_sensitivity_lookup(
        self,
        spad_colour: tuple[int, int, int],
        pixels: Image.Image,
        pde_range: tuple[float, float],
        wavebounds: tuple[float, float],
        threshold: int = 20,
    ) -> tuple[Wavebounds, interp.interp1d]:
        spad_pixels = []
        for y in range(pixels.height):
            for x in range(pixels.width):
                r, g, b = pixels.getpixel((x, y))
                if (
                    abs(r - spad_colour[0]) < threshold
                    and abs(g - spad_colour[1]) < threshold
                    and abs(b - spad_colour[2]) < threshold
                ):
                    spad_pixels.append((x, y))
        spad_pixels = np.array(spad_pixels)
        spad_x_pixels = spad_pixels[:, 0]
        spad_y_pixels = spad_pixels[:, 1]
        x_positions = wavebounds[0] + spad_x_pixels
        # x_positions = spad_x_pixels
        y_positions = np.interp(
            spad_y_pixels, (spad_y_pixels.min(), spad_y_pixels.max()), pde_range[::-1]
        )
        positions = np.array([x_positions, y_positions]).T
        unique_positions = np.unique(positions[:, 0], return_index=True)[1]
        positions = positions[unique_positions]
        sensitivity_function = interp.interp1d(
            positions[:, 0],
            positions[:, 1],
            kind="cubic",
            bounds_error=False,
            fill_value=(positions[:, 1][0], positions[:, 1][-1]),
        )

        bounds = Wavebounds(positions[:, 0].min(), positions[:, 0].max())

        return (bounds, sensitivity_function)


if __name__ == "__main__":
    ss = SpectralSensitivity(blue_pde_range=(0, 1), red_pde_range=(0, 1))
    plt.plot(ss.blue_range, ss.blue_spad_sensitivity(ss.blue_range), label="Blue SPAD")
    plt.plot(ss.red_range, ss.red_spad_sensitivity(ss.red_range), label="Red SPAD")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Sensitivity (PDE)")
    plt.legend()
    plt.show()
