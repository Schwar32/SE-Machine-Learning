import soundfile as sf
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import matplotlib


def testing():
    data, sample_rate = sf.read("./Data/Audio/acafly/XC6671.ogg")

    plt.specgram(data, Fs=sample_rate)
    plt.savefig("testing.png", transparent=True, dpi=100)


if __name__ == "__main__":
    testing()
