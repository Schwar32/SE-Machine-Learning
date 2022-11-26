import os
import datetime
import pandas as pd
import numpy as np
import soundfile as sf

import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt
import keras.models
from keras.utils import to_categorical
from keras import Model
from keras.layers import Flatten, Dense, Dropout, GaussianNoise, Input, BatchNormalizationV1
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50V2
from sklearn.preprocessing import LabelEncoder

import tensorflowjs as tfjs

# Loads in audio file and return wav data
def load_audio(file_path):
    audio_data, sample_rate = sf.read(file_path.numpy())
    return audio_data


# Takes in a file and returns a processed (224, 224, 3) spectrogram
def preprocess(file_path, training):
    [wav, ] = tf.py_function(load_audio, [file_path], [tf.float32])
    wav = wav[:960000]
    zero_padding = tf.zeros([960000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)

    spectrogram = tfio.audio.spectrogram(
        wav, nfft=2048, window=2048, stride=int(960000/224) + 1)

    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=32000, mels=224, fmin=500, fmax=13000)

    dbscale_mel_spectrogram = tfio.audio.dbscale(
        mel_spectrogram, top_db=80)

    if training:
        freq_mask = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=5)
        time_mask = tfio.audio.time_mask(freq_mask, param=5)
        time_mask = tf.expand_dims(time_mask, axis=2)
        time_mask = tf.repeat(time_mask, repeats=3, axis=2)
        noise = tf.random.normal(shape=tf.shape(time_mask), mean=0.0, stddev=.2, dtype=tf.float32)
        output = time_mask + noise
        output = tf.divide(
            tf.add(tf.subtract(
                output,
                tf.reduce_min(output)
            ), tf.keras.backend.epsilon()),
            tf.maximum(tf.subtract(
                tf.reduce_max(output),
                tf.reduce_min(output)
            ), tf.keras.backend.epsilon() * 2),
        )
        return output
    else:
        dbscale_mel_spectrogram = tf.expand_dims(dbscale_mel_spectrogram, axis=2)
        dbscale_mel_spectrogram = tf.repeat(dbscale_mel_spectrogram, repeats=3, axis=2)
        dbscale_mel_spectrogram = tf.divide(
            tf.add(tf.subtract(
                dbscale_mel_spectrogram,
                tf.reduce_min(dbscale_mel_spectrogram)
            ), tf.keras.backend.epsilon()),
            tf.maximum(tf.subtract(
                tf.reduce_max(dbscale_mel_spectrogram),
                tf.reduce_min(dbscale_mel_spectrogram)
            ), tf.keras.backend.epsilon() * 2),
        )
        return dbscale_mel_spectrogram


# Takes in a file path and returns the spectrogram and label
def process_training_images(file_path, label):
    spectrogram = preprocess(file_path, True)
    return spectrogram, label


def process_non_training_images(file_path, label):
    spectrogram = preprocess(file_path, False)
    return spectrogram, label


# Takes in an optional cutoff point and returns the dataset, number of labels, and csv dataframe
def load_data(cutoff):
    global label_encoder
    cols = ["filename", "primary_label", "secondary_labels", "common_name"]
    df = pd.read_csv("./Data/train_metadata.csv", usecols=cols)
    if cutoff is not None:
        df = df.loc[df['primary_label'] <= cutoff]
    #counts = df['primary_label'].value_counts()
    #df = df[~df['primary_label'].isin(counts[counts < 50].index)]  # Removes birds that have less than 50 calls from df
    untouched_df = df.copy()
    df['file_path'] = df.apply(lambda row: "./Data/Audio/" + row.primary_label + "/" + row.filename, axis=1)
    images = df["file_path"]

    labels = df.pop('common_name')
    labels = to_categorical(label_encoder.fit_transform(labels))

    output = tf.data.Dataset.from_tensor_slices((images, labels))
    return output, labels.shape[1], untouched_df


# Takes in data and a batch size, prepares the data for training, and returns
def setup_training_data(data, batch_size):
    autotune = tf.data.experimental.AUTOTUNE

    data = data.map(process_training_images, num_parallel_calls=autotune)
    data = data.shuffle(buffer_size=50000, seed=0)
    data = data.batch(batch_size)
    data = data.prefetch(autotune)
    return data


def setup_non_training_data(data, batch_size):
    autotune = tf.data.experimental.AUTOTUNE

    data = data.map(process_non_training_images, num_parallel_calls=autotune)
    data = data.shuffle(buffer_size=50000, seed=0)
    data = data.batch(batch_size)
    data = data.prefetch(autotune)
    return data


# Creates a new model using VGG16 cnn architecture with transfer learning
def create_model(num_labels):
    input_layer = Input(shape=(224, 224, 3))
    cnn = keras.applications.ResNet50V2(input_tensor=input_layer, weights='imagenet', include_top=False)

    for layer in cnn.layers:
        layer.trainable = False

    x = Flatten()(cnn.output)
    output = Dense(units=num_labels, activation='softmax')(x)

    model = Model([input_layer], [output])
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Recall(), keras.metrics.Precision()])
    return model


# Trains the model on given data
def train_model(training_model, training_data, validation_data, num_labels, epoch_amt, save):
    if training_model is None:
        training_model = create_model(num_labels)

    training_model.summary()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath="best_model",
                                                 mode='max',
                                                 monitor='val_categorical_accuracy',
                                                 save_best_only=True)

    training_model.fit(training_data, validation_data=validation_data, epochs=epoch_amt,
                       callbacks=[tensorboard_callback, checkpoint], verbose=True)

    if save:
        training_model.save("model")
        tfjs.converters.save_keras_model(training_model, "js_model")

    return training_model


# Takes in an audio file and model and returns predicted bird
def predict_file(file_path):
    global model
    image = preprocess(file_path, False)
    image = image.numpy().reshape(1, 224, 224, 3)
    predicted_label = model.predict([image], verbose=False)
    return predicted_label


# Takes in audio file and prints a detailed description of the prediction
def detailed_prediction(file_path, show_amount):
    global label_encoder
    prediction = predict_file(file_path)[0]
    max_indexes = np.argpartition(prediction, -show_amount)[-show_amount:]
    max_indexes = max_indexes[np.argsort(prediction[max_indexes])]
    max_indexes = np.flip(max_indexes)
    for index in range(len(max_indexes)):
        confidence = prediction[max_indexes[index]]
        class_prediction = label_encoder.inverse_transform([max_indexes[index]])
        print(class_prediction + " with " + str(round(confidence * 100, 2)) + "% confidence")


# Takes in csv dataframe and prints out one by one the success of a prediction on each file
def predict_all(df):
    global label_encoder
    count = 0
    correct = 0

    for index_num, row in df.iterrows():
        count += 1
        prediction = predict_file("./Data/Audio/" + row["primary_label"] + "/" + row["filename"])
        classes_x = np.argmax(prediction, axis=1)
        prediction_class = label_encoder.inverse_transform(classes_x)
        if prediction_class[0] == row["common_name"]:
            correct += 1
        print(str(correct / count) + ": " + row["common_name"] + " - " + prediction_class)
    print("FINAL RESULT: " + str(correct / count))


# Takes in the testing data, training data, and dataframe of training csv and prints out some checks
def check_model(testing_data, df, show_amount):
    global model

    print("EVALUATING ON TESTING DATA")
    model.evaluate(testing_data)

    test_file = "./Data/Audio/acafly/XC31063.ogg"
    print("PREDICTING " + test_file)
    detailed_prediction(test_file, show_amount)

    test_file = "./Data/Audio/amecro/XC80525.ogg"
    print("PREDICTING " + test_file)
    detailed_prediction(test_file, show_amount)

    test_file = "./Data/Audio/banswa/XC138873.ogg"
    print("PREDICTING " + test_file)
    detailed_prediction(test_file, show_amount)

    test_file = "./Data/Audio/caltow/XC126344.ogg"
    print("PREDICTING " + test_file)
    detailed_prediction(test_file, show_amount)

    test_file = "./Data/Audio/foxspa/XC120607.ogg"
    print("PREDICTING " + test_file)
    detailed_prediction(test_file, show_amount)

    print("EVALUATING ON ALL DATA")
    predict_all(df[::100])


if __name__ == "__main__":
    TRAINING_SIZE = .60
    VALIDATION_SIZE = .20
    BATCH_SIZE = 16
    EPOCH_AMOUNT = 1
    SHOW_AMOUNT = 5
    label_encoder = LabelEncoder()

    dataset, label_count, saved_df = load_data(cutoff="amewig")
    dataset = dataset.shuffle(buffer_size=75000, seed=0)

    training_size = int(len(dataset) * TRAINING_SIZE)
    validating_size = int(len(dataset) * VALIDATION_SIZE)
    testing_size = int(len(dataset) - training_size - validating_size)

    train = dataset.take(training_size)
    testing_dataset = dataset.skip(training_size)
    validation = testing_dataset.skip(validating_size)
    testing = testing_dataset.take(validating_size)

    train = setup_training_data(train, BATCH_SIZE)
    validation = setup_non_training_data(validation, BATCH_SIZE)
    testing = setup_non_training_data(testing, BATCH_SIZE)

    """
    for x in train:
        print(np.min(x[0][0].numpy()))
        print(np.max(x[0][0].numpy()))
        plt.imshow(x[0][0].numpy())
        plt.show()
        break
    for x in validation:
        print(np.min(x[0][0].numpy()))
        print(np.max(x[0][0].numpy()))
        plt.imshow(x[0][0].numpy())
        plt.show()
        break
    for x in testing:
        print(np.min(x[0][0].numpy()))
        print(np.max(x[0][0].numpy()))
        plt.imshow(x[0][0].numpy())
        plt.show()
        break
    """

    #model = keras.models.load_model("best_model")
    model = train_model(training_model=None, training_data=train, validation_data=validation, num_labels=label_count,
                        epoch_amt=EPOCH_AMOUNT, save=True)
    # Accuracy on data
    check_model(testing_data=testing, df=saved_df, show_amount=SHOW_AMOUNT)
#
    # tensorboard --logdir=logs/fit --host localhost --port 8088