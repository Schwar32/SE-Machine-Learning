#!/usr/bin/env python

# In[1]:

import datetime
import pandas as pd
import numpy as np
import soundfile as sf

import tensorflow as tf
import tensorflow_io as tfio

import keras.models
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from keras.layers import Flatten, Dense, Dropout, GaussianNoise
from keras import Model

from keras.applications.vgg16 import VGG16

# In[2]:


def load_audio(file_name):
    audio_data, sample_rate = sf.read(file_name.numpy())
    return audio_data


# In[3]:


def preprocess(file_path):
    [wav, ] = tf.py_function(load_audio, [file_path], [tf.float32])
    wav = wav[:960000]
    zero_padding = tf.zeros([960000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)

    spectrogram = tfio.audio.spectrogram(
        wav, nfft=1024, window=1024, stride=int(960000/224) + 1)

    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=32000, mels=224, fmin=0, fmax=16000)

    dbscale_mel_spectrogram = tfio.audio.dbscale(
        mel_spectrogram, top_db=80)

    freq_mask = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=10)

    time_mask = tfio.audio.time_mask(freq_mask, param=10)
    time_mask = tf.expand_dims(time_mask, axis=2)
    time_mask = tf.repeat(time_mask, repeats=3, axis=2)
    return time_mask


# In[4]:


def create_model(num_labels):
    cnn = VGG16(input_shape=[224, 224, 3], weights='imagenet', include_top=False)
    
    for layer in cnn.layers:
        layer.trainable = False

    x = Flatten()(cnn.output) 
    output = GaussianNoise(0.1)(x)
    output = Dropout(0.5)(output)
    output = Dense(units=num_labels, activation='softmax')(output)
    
    model = Model([cnn.input], [output])
    model.summary()
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, decay=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Recall(), keras.metrics.Precision()])
    return model


# In[5]:


def train_model(model, train, test, num_labels, epoch_amt, save):
    if model is None:
        model = create_model(num_labels)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath="best_model",
                                                 mode='max',
                                                 monitor='val_categorical_accuracy',
                                                 save_best_only=True)
    
    model.fit(train, validation_data=test, epochs=epoch_amt, callbacks=[tensorboard_callback, checkpoint], verbose=True)
    
    if save:
        model.save("model")

    return model


# In[6]:


def predict_file(file_name, model):
    image = preprocess(file_name)
    image = image.numpy().reshape(1, 224, 224, 3)
    predicted_label = model.predict([image], verbose=False)
    return predicted_label


# In[7]:


def process_images(input_path, label):
    spectrogram = preprocess(input_path)
    return spectrogram, label


# In[8]:


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[9]:


cols = ["filename", "primary_label"]
df = pd.read_csv("./Data/train_metadata.csv", usecols=cols)
#df = df.loc[df['primary_label'] <= "clcrob"]
saved_df = df.copy()


# In[10]:


df['file_path'] = df.apply(lambda row: "./Data/Audio/" + row.primary_label + "/" + row.filename, axis=1)
images = df["file_path"]


# In[11]:


labels = df.pop('primary_label')
labels_encoder = LabelEncoder()
labels = to_categorical(labels_encoder.fit_transform(labels))


# In[12]:


dataset = tf.data.Dataset.from_tensor_slices((images, labels))


# In[13]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

dataset = dataset.map(process_images, num_parallel_calls=AUTOTUNE)
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(AUTOTUNE)


# In[14]:


training_size = int(len(dataset) * .80)        # 80% of data is for training
testing_size = len(dataset) - training_size    # 20% of data is for validating

train = dataset.take(training_size)
test = dataset.skip(training_size).take(testing_size)


# In[ ]:


#model = keras.models.load_model("model")
model = train_model(model=None, train=train, test=test, num_labels=labels.shape[1], epoch_amt=20, save=True)


# In[ ]:


prediction = predict_file("./Data/Audio/acafly/XC31063.ogg", model, labels_encoder)
classes_x = np.argmax(prediction, axis=1)
prediction_class = labels_encoder.inverse_transform(classes_x)
str(prediction_class[0]) + " with " + str(prediction[0][classes_x][0]*100) + "% confidence"


# In[ ]:


model.evaluate(train)


# In[ ]:


model.evaluate(test)


# In[ ]:


#Accuracy on data
count = 0
correct = 0
for index_num, row in saved_df.iterrows():
    prediction = predict_file("./Data/Audio/" + row["primary_label"] + "/" + row["filename"], model)
    classes_x = np.argmax(prediction, axis=1)
    prediction_class = labels_encoder.inverse_transform(classes_x)
    if prediction_class[0] == row["primary_label"]:
        correct += 1
        print(str(correct/count) + ": " + row["primary_label"] + " - " + prediction_class)
    count += 1
