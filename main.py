import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib
import soundfile as sf

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten


def load_audio(filename):
    data, sample_rate = sf.read(filename)
    return data


def preprocess(file_path, label):
    wav = load_audio(file_path)
    wav = wav[:320000]
    zero_padding = tf.zeros([320000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


if __name__ == "__main__":
    POS = "Data/Audio/acafly/"
    NEG = "Data/Audio/acowoo/"

    pos = tf.data.Dataset.list_files(POS+'*')
    neg = tf.data.Dataset.list_files(NEG+'*')

    positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
    data = positives.concatenate(negatives)

    filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()

    data = data.map(preprocess)
    data = data.cache()
    data = data.shuffle(buffer_size=1000)
    data = data.batch(16)
    data = data.prefetch(8)

    train = data.take(226)
    test = data.skip(226).take(96)

    samples, labels = train.as_numpy_iterator().next()
    print("-------------------------" + samples.shape)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1491, 257, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    model.summary()

    hist = model.fit(train, epochs=4, validation_data=test)

    plt.title('Loss')
    plt.plot(hist.history['loss'], 'r')
    plt.plot(hist.history['val_loss'], 'b')
    plt.show()
    plt.title('Precision')
    plt.plot(hist.history['precision'], 'r')
    plt.plot(hist.history['val_precision'], 'b')
    plt.show()
    plt.title('Recall')
    plt.plot(hist.history['recall'], 'r')
    plt.plot(hist.history['val_recall'], 'b')
    plt.show()

