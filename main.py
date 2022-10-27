import os

import keras.models
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import soundfile as sf

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten


def load_audio(file_name):
    audio_data, sample_rate = sf.read(file_name.numpy())
    return audio_data[::4]


def preprocess(file_path, label):
    [wav,] = tf.py_function(load_audio, [file_path], [tf.float32])
    wav = wav[:96000]
    zero_padding = tf.zeros([96000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=512, frame_step=256)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


def load_data():
    POS = "Data/Audio/acafly/"
    NEG = "Data/Audio/ameavo/"

    pos = tf.data.Dataset.list_files(POS+'*')
    neg = tf.data.Dataset.list_files(NEG+'*')

    positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
    return positives.concatenate(negatives)


def setup_data(data):
    data = data.map(preprocess)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)
    data = data.batch(2)
    data = data.prefetch(1)
    return data


def create_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1496, 257, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    return model


def load_model():
    return keras.models.load_model("model")


def display_training_data(hist):
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


def train_model(epoch_amt, save):
    model = create_model()

    hist = model.fit(train, epochs=epoch_amt, validation_data=test)

    display_training_data(hist)

    if save:
        model.save("model")


def check_model(test):
    successes = 0
    fails = 0
    for x in range(len(test)):
        X_test, y_test = test.as_numpy_iterator().next()
        yhat = model.predict(X_test)
        yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
        if yhat[0] == y_test.astype(int)[0]:
            successes += 1
        else:
            fails += 1
        if yhat[1] == y_test.astype(int)[1]:
            successes += 1
        else:
            fails += 1

    print(successes)
    print(fails)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    data = load_data()
    data = setup_data(data)

    # filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()

    # spectrogram, label = preprocess(filepath, label)

    # plt.figure(figsize=(30, 20))
    # plt.imshow(tf.transpose(spectrogram)[0])
    # plt.show()

    training_size = int(len(data) * .80)        # 80% of data is for training
    testing_size = len(data) - training_size    # 20% of data is for validating

    train = data.take(training_size)
    test = data.skip(training_size).take(testing_size)

    # samples, labels = train.as_numpy_iterator().next()
    # print("-----------------------------------------------------")
    # print(samples.shape)
    # print("-----------------------------------------------------")

    # train_model(10, False)

    model = load_model()
    check_model(test)

