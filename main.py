import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio


def load_audio(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resamples to standard 16000Hz
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


if __name__ == "__main__":
    wav = load_audio("Data/Audio/acafly/XC6671.ogg")
    print(wav)
