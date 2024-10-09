import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.integrate import quad
import numpy as np
from PIL import Image
from typing import Tuple


class Wavebounds:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end
        self.range = end - start

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.start
        elif index == 1:
            return self.end
        elif index == -1:
            return self.end
        else:
            raise IndexError("Index must be 0 or 1")

    def __str__(self) -> str:
        return f"({self.start}, {self.end})"

    def to_list(self):
        return [self.start, self.end]

    def toJson(self):
        return {"start": self.start, "end": self.end}


class Fluorophore_Intensity_PDF:
    def __init__(
        self,
        name: str,
        image_path: str,
        image_wavebounds: tuple[float, float],
        fluor_colour: tuple[int, int, int],
        intensity_range: tuple[int, int] = (1, 0),
        intensity_threshold: int = 20,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.image_path = image_path
        self.image_wavebounds = image_wavebounds
        self.fluor_colour = fluor_colour
        self.intensity_range = intensity_range

        self.border_color = (0, 0, 0)

        self.image = Image.open(self.image_path)
        self.image = self.image.convert("RGB")

        self.verbose = verbose

        self.intensity_threshold = intensity_threshold

        self.fluor_pixels = self.extract_image_from_border()
        self.strictness = 0
        self.wavebounds, self.intensity_pdf = self.extract_fluorophore_pdf()

    def extract_image_from_border(self, threshold: int = 10) -> Image:
        width, height = self.image.size
        pixels = self.image.load()

        top, left = width, height
        bottom, right = 0, 0

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                if (
                    abs(r - self.border_color[0]) < threshold
                    and abs(g - self.border_color[1]) < threshold
                    and abs(b - self.border_color[2]) < threshold
                ):
                    top = min(top, y)
                    left = min(left, x)
                    bottom = max(bottom, y)
                    right = max(right, x)

        # The 2 and 6 are to assist with the border removal
        pixels = self.image.crop((left + 2, top, right, bottom - 6))
        return pixels

    def extract_fluorophore_pdf(self) -> Tuple[Wavebounds, interp.UnivariateSpline]:
        fluoro_pixels = []
        for y in range(self.fluor_pixels.height):
            for x in range(self.fluor_pixels.width):
                r, g, b = self.fluor_pixels.getpixel((x, y))
                if (
                    abs(r - self.fluor_colour[0]) < self.intensity_threshold
                    and abs(g - self.fluor_colour[1]) < self.intensity_threshold
                    and abs(b - self.fluor_colour[2]) < self.intensity_threshold
                ):
                    fluoro_pixels.append((x, y))

        fluoro_pixels = np.array(fluoro_pixels)
        fluoro_x_pixels = fluoro_pixels[:, 0]
        fluoro_y_pixels = fluoro_pixels[:, 1]

        x_positions = self.image_wavebounds[0] + fluoro_x_pixels + 8
        y_positions = self.intensity_range[0] - (
            fluoro_y_pixels / self.fluor_pixels.height
        )

        positions = np.array([x_positions, y_positions]).T
        unique_positions = np.unique(positions[:, 0], return_index=True)[1]
        positions = positions[unique_positions]

        # Normalising the intensity PDF to make it a probability distribution to make the y between 0 and 1
        if self.verbose:
            positions[:, 1] = (positions[:, 1] - positions[:, 1].min()) / (
                positions[:, 1].max() - positions[:, 1].min()
            )
        else:
            positions[:, 1] = positions[:, 1] / positions[:, 1].sum()

        fluoro_function_wavebounds = Wavebounds(
            positions[:, 0].min(), positions[:, 0].max()
        )
        original_stricness = 0.00005 if not self.verbose else 0.005
        self.strictness = original_stricness
        fluoro_function = interp.UnivariateSpline(
            positions[:, 0], positions[:, 1], s=original_stricness
        )

        return fluoro_function_wavebounds, fluoro_function

    def intensity_cdf(self, x: float) -> float:
        return quad(self.intensity_pdf, self.wavebounds.start, x)[0]

    def obtain_inverse_cdf(self):
        x = np.linspace(*self.wavebounds, 1000)
        y = np.array([self.intensity_cdf(x_) for x_ in x])
        cdf_values = (y - y.min()) / (y.max() - y.min())
        lookup = interp.UnivariateSpline(cdf_values, x, s=0.00005)

        # plot
        plt.figure()
        plt.plot(cdf_values, x, label="Inverse CDF")
        plt.xlabel("CDF")
        plt.ylabel("Wavelength (nm)")
        plt.title("Inverse CDF of NADH")
        plt.show()

    def save_emission(self, filename: str) -> None:
        x_control_points = np.linspace(self.wavebounds.start, self.wavebounds.end, 1000)
        y_control_points = self.intensity_pdf(x_control_points)
        np.savez(
            filename,
            x_control_points=x_control_points,
            y_control_points=y_control_points,
            strictness=self.strictness,
        )


if __name__ == "__main__":
    biomaker_path = (
        "../../generate-reference-images/biomarker-intensity-distribution.png"
    )
    nadh_colour = (0, 0, 0)  # roughly black
    flavin_colour = (227, 220, 135)  # roughly yellow
    elastin_colour = (183, 91, 107)  # roughly red
    image_wavebounds = (300, 700)

    nadh_intensity_pdf = Fluorophore_Intensity_PDF(
        "NADH", biomaker_path, image_wavebounds, nadh_colour
    )

    nadh_intensity_pdf.obtain_inverse_cdf()

    nadh_x = np.linspace(*nadh_intensity_pdf.wavebounds, 1000)
    nadh_y = nadh_intensity_pdf.intensity_pdf(nadh_x)

    flavin_intensity_pdf = Fluorophore_Intensity_PDF(
        "Flavin", biomaker_path, image_wavebounds, flavin_colour
    )
    flavin_x = np.linspace(*flavin_intensity_pdf.wavebounds, 1000)
    flavin_y = flavin_intensity_pdf.intensity_pdf(flavin_x)

    elastin_intensity_pdf = Fluorophore_Intensity_PDF(
        "Elastin", biomaker_path, image_wavebounds, elastin_colour
    )
    elastin_x = np.linspace(*elastin_intensity_pdf.wavebounds, 1000)
    elastin_y = elastin_intensity_pdf.intensity_pdf(elastin_x)

    plt.plot(nadh_x, nadh_y, label="NADH")
    plt.plot(flavin_x, flavin_y, label="Flavin", color="yellow")
    plt.plot(elastin_x, elastin_y, "r-", label="Elastin")
    plt.legend()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    # plt.ylim(0, 1)
    plt.xlim(300, 700)
    plt.title("NADH's Normalised Intensity over Emission Spectra (nm)")
    plt.show()
