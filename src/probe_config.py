from Intensity_PDF import Wavebounds

# ORIGINAL PROBE CONFIG
# CHANNEL_RANGE = Wavebounds(474.51, 735.12)
# NUM_CHANNELS = 512
# CHANNEL_WIDTH = 0.509  # nm

# SYNTHETIC DATA CONFIG -> REQUIRES A LARGER CHANNEL RANGE FOR ENDOGENOUS FLUOROPHORES
# Wavelength settings
CHANNEL_RANGE = Wavebounds(300, 1000)
CHANNEL_WIDTH = 0.509  # nm
NUM_CHANNELS = round((CHANNEL_RANGE[1] - CHANNEL_RANGE[0]) / CHANNEL_WIDTH)

# Time settings
EPISODE_TIME = 60  # ns
TIME_BIN_WIDTH = 0.05  # ns
NUM_BINS = int(EPISODE_TIME / TIME_BIN_WIDTH)

# Bias settingns
BIAS_PHOTON_RANGE = (50, 75)  # number of photons see in bin -> for 50ps bins
DYNAMIC_BIAS = False
if TIME_BIN_WIDTH != 0.05 and DYNAMIC_BIAS:
    bias_scaling_factor = TIME_BIN_WIDTH / 0.05
    BIAS_PHOTON_RANGE = (
        int(BIAS_PHOTON_RANGE[0] * bias_scaling_factor),
        int(BIAS_PHOTON_RANGE[1] * bias_scaling_factor),
    )


def get_probe_metadata() -> dict:
    return {
        "CHANNEL_RANGE": CHANNEL_RANGE.to_list(),
        "NUM_CHANNELS": NUM_CHANNELS,
        "CHANNEL_WIDTH": CHANNEL_WIDTH,
        "EPISODE_TIME": EPISODE_TIME,
        "TIME_BIN_WIDTH": TIME_BIN_WIDTH,
        "NUM_BINS": NUM_BINS,
    }


if __name__ == "__main__":
    print(f"NUM_CHANNELS: {NUM_CHANNELS}")
    print(f"BIAS: {BIAS_PHOTON_RANGE}")
