import os

import keras.models
from keras import regularizers
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import soundfile as sf

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten

import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dropout

def load_audio(file_name):
    print(file_name)
    audio_data, sample_rate = sf.read(file_name)
    return audio_data[::4]


def preprocess(file_path):
    wav = load_audio(file_path)
    wav = wav[:80000]
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)

    spectrogram = tfio.audio.spectrogram(
        wav, nfft=512, window=512, stride=256)

    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=8000, mels=256, fmin=0, fmax=4000)

    dbscale_mel_spectrogram = tfio.audio.dbscale(
        mel_spectrogram, top_db=80)

    freq_mask = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=10)

    time_mask = tfio.audio.time_mask(freq_mask, param=10)
    time_mask = tf.expand_dims(time_mask, axis=2)

    return time_mask


def load_data():
    POS = "Data/Audio/banswa/"
    NEG = "Data/Audio/batpig1/"

    pos = tf.data.Dataset.list_files(POS+'*')
    neg = tf.data.Dataset.list_files(NEG+'*')

    positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
    return positives.concatenate(negatives)


def setup_data(data):
    data = data.map(preprocess)
    data = data.shuffle(buffer_size=1024)
    data = data.batch(4)
    data = data.prefetch(2)
    return data


def create_model(num_labels):
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(313, 256, 1), kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))
    model.summary()
    model.compile('Adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
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


def train_model(X_train, y_train, X_test, y_test, num_labels, epoch_amt, save):
    model = create_model(num_labels)

    hist = model.fit(X_train, y_train, batch_size=8, epochs=epoch_amt, validation_data=(X_test, y_test))

    display_training_data(hist)

    test_accuracy = model.evaluate(X_test, y_test)
    print(test_accuracy[1])

    if save:
        model.save("model")
    return model


def check_model(model, test):
    successes = 0
    fails = 0
    for x in range(len(test)):
        X_test, y_test = test.as_numpy_iterator().next()
        yhat = model.predict(X_test)
        yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
        for i in range(len(yhat)):
            if yhat[i] == y_test.astype(int)[i]:
                successes += 1
            else:
                fails += 1
    print(successes)
    print(fails)


def check_file(file_name, model):
    wav = load_audio(file_name)
    wav = wav[:80000]
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)

    spectrogram = tfio.audio.spectrogram(
        wav, nfft=512, window=512, stride=256)

    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=8000, mels=256, fmin=0, fmax=4000)

    dbscale_mel_spectrogram = tfio.audio.dbscale(
        mel_spectrogram, top_db=80)

    freq_mask = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=10)

    time_mask = tfio.audio.time_mask(freq_mask, param=10)
    time_mask = tf.expand_dims(time_mask, axis=2)
    predicted_label = model.predict(time_mask.numpy().reshape(1, 313, 256, 1))
    print("Label:" + str(predicted_label))
    audio_path = "./Data/Audio/"
    metadata = pd.read_csv('./Data/train_metadata.csv')
    print(metadata['primary_label'].value_counts())

    extracted_features = []
    for index_num, row in metadata.iterrows():
        if row["primary_label"] == "amewig":
            break
        file_name = os.path.join(audio_path + row["primary_label"] + "/" + row["filename"])
        final_class_labels = row["primary_label"]
        data = preprocess(file_name)
        extracted_features.append([data, final_class_labels])

    print("test")
    extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
    print(extracted_features_df.head(10))

    X = np.array(extracted_features_df['feature'].tolist())
    y = np.array(extracted_features_df['class'].tolist())
    print(X.shape)
    print(y.shape)

    labelencoder = LabelEncoder()
    y = to_categorical(labelencoder.fit_transform(y))
    classes_x = np.argmax(predicted_label, axis=1)
    prediction_class = labelencoder.inverse_transform(classes_x)
    print("Prediction:" + str(prediction_class))


if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(len(tf.config.list_physical_devices('GPU')))
    """
    audio_path = "./Data/Audio/"
    metadata = pd.read_csv('./Data/train_metadata.csv')
    print(metadata['primary_label'].value_counts())

    extracted_features = []
    for index_num, row in metadata.iterrows():
        if row["primary_label"] == "amewig":
            break
        file_name = os.path.join(audio_path + row["primary_label"] + "/" + row["filename"])
        final_class_labels = row["primary_label"]
        data = preprocess(file_name)
        extracted_features.append([data, final_class_labels])

    print("test")
    extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
    print(extracted_features_df.head(10))

    X = np.array(extracted_features_df['feature'].tolist())
    y = np.array(extracted_features_df['class'].tolist())
    print(X.shape)
    print(y.shape)

    labelencoder = LabelEncoder()
    y = to_categorical(labelencoder.fit_transform(y))

    print(y.shape)
    num_labels = y.shape[1]
    print(num_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    train_model(X_train, y_train, X_test, y_test, num_labels, 50, True)
    """
    """
    data = load_data()
    data = setup_data(data)

    training_size = int(len(data) * .80)        # 80% of data is for training
    testing_size = len(data) - training_size    # 20% of data is for validating

    train = data.take(training_size)
    test = data.skip(training_size).take(testing_size)

    #model = train_model(8, False)
    """
    model = load_model()
    check_file("./Data/Audio/ameavo/XC99571.ogg", model)
    """
    check_model(model, test)
    """

