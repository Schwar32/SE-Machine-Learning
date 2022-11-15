import os
import datetime
import pandas as pd
import numpy as np
import librosa
import soundfile as sf

import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt
import keras.models
from keras.utils import to_categorical
from keras import Model
from keras.layers import Flatten, Dense, Dropout, GaussianNoise, Input
from keras import regularizers
from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import LabelEncoder


# Loads in audio file and return wav data
def load_audio(file_path):
    audio_data, sample_rate = sf.read(file_path.numpy())
    return audio_data


# Takes in a file and returns a processed (224, 224, 3) spectrogram
def preprocess(file_path):
    [wav, ] = tf.py_function(load_audio, [file_path], [tf.float32])
    wav = wav[:960000]
    zero_padding = tf.zeros([960000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)

    spectrogram = tfio.audio.spectrogram(
        wav, nfft=4096, window=4096, stride=int(960000/224) + 1)

    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=32000, mels=224, fmin=0, fmax=16000)

    dbscale_mel_spectrogram = tfio.audio.dbscale(
        mel_spectrogram, top_db=80)

    dbscale_mel_spectrogram = tf.expand_dims(dbscale_mel_spectrogram, axis=2)
    dbscale_mel_spectrogram = tf.repeat(dbscale_mel_spectrogram, repeats=3, axis=2)
    dbscale_mel_spectrogram = dbscale_mel_spectrogram / 80
    return dbscale_mel_spectrogram


# Takes in a file path and returns the spectrogram and label
def process_images(file_path, label):
    spectrogram = preprocess(file_path)
    return spectrogram, label


# Takes in an optional cutoff point and returns the dataset, number of labels, and csv dataframe
def load_data(cutoff):
    global label_encoder
    cols = ["filename", "primary_label"]
    df = pd.read_csv("./Data/train_metadata.csv", usecols=cols)
    if cutoff is not None:
        df = df.loc[df['primary_label'] <= cutoff]
    counts = df['primary_label'].value_counts()
    df = df[~df['primary_label'].isin(counts[counts < 75].index)]  # Removes birds that have less than 75 calls from df
    untouched_df = df.copy()
    df['file_path'] = df.apply(lambda row: "./Data/Audio/" + row.primary_label + "/" + row.filename, axis=1)
    images = df["file_path"]

    labels = df.pop('primary_label')
    labels = to_categorical(label_encoder.fit_transform(labels))

    output = tf.data.Dataset.from_tensor_slices((images, labels))
    return output, labels.shape[1], untouched_df


# Takes in data and a batch size, prepares the data for training, and returns
def setup_data(data, batch_size):
    autotune = tf.data.experimental.AUTOTUNE

    data = data.map(process_images, num_parallel_calls=autotune)
    data = data.shuffle(buffer_size=2048, seed=0)
    data = data.batch(batch_size)
    data = data.prefetch(autotune)
    return data


# Creates a new model using VGG16 cnn architecture with transfer learning
def create_model(num_labels):
    input_layer = Input(shape=(224, 224, 3))
    noise = GaussianNoise(0.5)(input_layer)
    cnn = VGG16(input_shape=[224, 224, 3], input_tensor=noise, weights='imagenet', include_top=False)

    for layer in cnn.layers[:-2]:
        layer.trainable = False

    x = Flatten()(cnn.output)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0025))(x)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0025))(x)
    x = Dropout(0.4)(x)
    output = Dense(units=num_labels, activation='softmax')(x)

    model = Model([input_layer], [output])
    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Recall(), keras.metrics.Precision()])
    return model


# Trains the model on given data
def train_model(training_model, training_data, num_labels, epoch_amt, save):
    if training_model is None:
        training_model = create_model(num_labels)

    training_model.summary()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath="best_model",
                                                 mode='max',
                                                 monitor='categorical_accuracy',
                                                 save_best_only=True)

    training_model.fit(training_data, epochs=epoch_amt,
                       callbacks=[tensorboard_callback, checkpoint], verbose=True)

    if save:
        training_model.save("model")

    return training_model


# Takes in an audio file and model and returns predicted bird
def predict_file(file_path):
    global model
    image = preprocess(file_path)
    plt.imshow(image)
    plt.show()
    image = image.numpy().reshape(1, 224, 224, 3)
    predicted_label = model.predict([image], verbose=False)
    return predicted_label


# Takes in audio file and prints a detailed description of the prediction
def detailed_prediction(file_path):
    global label_encoder
    prediction = predict_file(file_path)[0]
    SHOW_AMOUNT = 5
    max_indexes = np.argpartition(prediction, -SHOW_AMOUNT)[-SHOW_AMOUNT:]
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
        if prediction_class[0] == row["primary_label"]:
            correct += 1
        print(str(correct / count) + ": " + row["primary_label"] + " - " + prediction_class)
    print("FINAL RESULT: " + str(correct / count))


# Takes in the testing data, training data, and dataframe of training csv and prints out some checks
def check_model(validation_data, df):
    global model

    print("EVALUATING ON VALIDATION DATA")
    model.evaluate(validation_data)

    test_file = "./Data/Audio/acafly/XC31063.ogg"
    print("PREDICTING " + test_file)
    detailed_prediction(test_file)

    test_file = "./Data/Audio/amecro/XC80525.ogg"
    print("PREDICTING " + test_file)
    detailed_prediction(test_file)

    test_file = "./Data/Audio/banswa/XC138873.ogg"
    print("PREDICTING " + test_file)
    detailed_prediction(test_file)

    print("EVALUATING ON ALL DATA")
    predict_all(df[::100])


if __name__ == "__main__":
    TRAINING_SIZE = .80
    BATCH_SIZE = 8
    EPOCH_AMOUNT = 200

    label_encoder = LabelEncoder()

    dataset, label_count, saved_df = load_data(cutoff=None)
    dataset = setup_data(dataset, BATCH_SIZE)

    training_size = int(len(dataset) * TRAINING_SIZE)
    validating_size = len(dataset) - training_size

    train = dataset.take(training_size)
    validation = dataset.skip(training_size).take(validating_size)

    model = keras.models.load_model("best_model")
    #model = train_model(training_model=None, training_data=train, num_labels=label_count,
                        #epoch_amt=EPOCH_AMOUNT, save=True)
    # Accuracy on data
    check_model(validation_data=validation, df=saved_df)
