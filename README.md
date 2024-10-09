# TSCPC Data Simulation Package
Wecome to this package for TSCPC histogram data generation. This package is intended to allow the simulation of single and mixed fluorophore fluoroscence data.

This package will allow you to simulate any fluorophore given:
- The lifetime (ns) is known
- The emission spectra is known

The package also offers the user to control the temporal and spatial resolutions of the histogram, meaning:
- The total length of time of the histogram can be adjusted
- The time bin size can be adjusted
- The spectral range to which the histogram capture can be adjusted.

These setting can be configured in the probe-config.py file.

The package will also simulate properties of the device:
-  PDE (Photon Dectector Efficency), recreated from Erdogan et al 2019 Figure 15.2. This option will reduce the number of photons simulated and aggretated in the histogram by a scalor taken from the PDE function.
- IRF, we simulate the IRF using a Guassian function, allowing the user to provied a wavelength dependent $/mu$ and $/sigma$

This package allows for PDE (Photon Dectector Efficency), recreated from Erdogan et al 2019 Figure 15.2, to be generated and a Poission noise to be added to the samples.


## Set-up requirements
I recommend that the user sets up a conda environment to handle all of the packages required, installation instructions can be found [here](https://docs.anaconda.com/miniconda/miniconda-install/):


Next lets create a new conda environment and install the necessary packages:

```sh
conda create -n data-simulation python=3.11
conda install numpy scipy matplotlib jupyter
conda install -c conda-forge ipympl
conda activate 
```

Ensure to activate the conda environment before running any of the code to generate data.

And to whenever you are finished with the data-generation you can stop using your conda environment with:

```sh
conda deactivate
```

## Quick Start
You can find an example of how to you the package in the jupyter notebook in the /scripts/simulation-mixture-plate-reader.ipynb.

