import os

import keras.models
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib
import soundfile as sf

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten


def load_audio(file_name):
    data, sample_rate = sf.read(file_name.numpy())
    return data[::4]


def preprocess(file_path, label):
    [wav,] = tf.py_function(load_audio, [file_path], [tf.float32])
    wav = wav[:96000]
    zero_padding = tf.zeros([96000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=64)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    POS = "Data/Audio/acafly/"
    NEG = "Data/Audio/ameavo/"

    pos = tf.data.Dataset.list_files(POS+'*')
    neg = tf.data.Dataset.list_files(NEG+'*')

    positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
    data = positives.concatenate(negatives)

    #filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()

    #spectrogram, label = preprocess(filepath, label)

    #plt.figure(figsize=(30, 20))
    #plt.imshow(tf.transpose(spectrogram)[0])
    #plt.show()

    data = data.map(preprocess)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)
    data = data.batch(2)
    data = data.prefetch(1)

    train = data.take(50)
    test = data.skip(50).take(25)
    """

    #samples, labels = train.as_numpy_iterator().next()
    #print("-----------------------------------------------------")
    #print(samples.shape)
    #print("-----------------------------------------------------")

    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1496, 257, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    model.summary()

    hist = model.fit(train, epochs=10, validation_data=test)

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

    for x in range(20):
        X_test, y_test = test.as_numpy_iterator().next()
        yhat = model.predict(X_test)
        yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
        print(yhat)
        print(y_test.astype(int))

    model.save("model")
    """

    model = keras.models.load_model("model")
    successes = 0
    fails = 0
    for x in range(25):
        X_test, y_test = test.as_numpy_iterator().next()
        yhat = model.predict(X_test)
        yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
        if(yhat[0] == y_test.astype(int)[0]):
            successes += 1
        else:
            fails += 1
        if (yhat[1] == y_test.astype(int)[1]):
            successes += 1
        else:
            fails += 1

    print(successes)
    print(fails)