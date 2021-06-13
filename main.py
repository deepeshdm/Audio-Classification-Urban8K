

# Author @ Deepesh Mhatre
# NOTE : Change th dataset_dir to where the dataset is present in your pc.

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.callbacks import Callback
from time import time

dataset_dir = r"C:/Users/dipesh/Desktop/Urban8K_Dataset"
df = pd.read_csv(dataset_dir + r"/UrbanSound8K.csv")

# ---------------------------------------------------------------

n_mfcc = 150


# Takes audio file name/path & returns its MFCC.
def convert_audio2MFCC(audio_file):
    samples, sample_rate = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(samples, sr=sample_rate, n_mfcc=n_mfcc)
    # scaling the mfcc values
    scaled_mfcc = np.mean(mfcc.T, axis=0)
    # mfcc & scaled mfcc have different shapes
    return scaled_mfcc


print("Data Preprocessing Started...")
print("Converting raw audio data to mfcc...")

processed_data = []
for audio_filename, fold, label in df[["slice_file_name", "fold", "class"]].values:
    file_path = dataset_dir + "/" + "fold" + str(fold) + "/" + audio_filename
    mfcc = convert_audio2MFCC(file_path)
    processed_data.append([mfcc, label])

print("Bringing relevant data togeteher , creating dataframe...")
df = pd.DataFrame(processed_data, columns=["mfcc", "label"])
x = np.array(df["mfcc"].tolist())
y = np.array(df["label"].tolist())

# contains all string class labels
labels = (list(pd.get_dummies(y)))

print("On-Hot encoding class labels...")
# one-hot-encoding labels
y = pd.get_dummies(y).values

print("Splitting dataset into train and test...")
# splitting train & test data.
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=0)

print("Data Preprocessing Completed Sucessfully")

# ---------------------------------------------------------------

print("Creating model...")

# Defining the model structure
model = Sequential()
model.add(Dense(100, activation="relu", input_shape=(128,)))
model.add(Dropout(0.2))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(25, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# Custom Keras callback to stop training when certain accuracy is achieved.
class MyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True


print("Started model training...")
start = time()
model.fit(x_train, y_train, batch_size=100, epochs=100, callbacks=[MyThresholdCallback(0.9)],
          validation_data=(x_test, y_test))
end = time()
print("Model training completed in {} mins sucessfully".format(round(end - start, 2) / 60))

# saving the model
model.save("Audio_Classification_Urban8K")


# ---------------------------------------------------------------


# Takes audio filename/path and returns predicted label
def predict_audio_class(filename):
    labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
              'jackhammer', 'siren', 'street_music']

    # loading trained model
    model = load_model("Audio_Classification_Urban8K")
    mfcc = convert_audio2MFCC(filename)
    mfcc_reshaped = mfcc.reshape(1, -1)
    predicted_label = model.predict(mfcc_reshaped)
    label = labels[np.argmax(predicted_label)]
    return label


# ---------------------------------------------------------------

audio_file = r"C:\Users\dipesh\Desktop\horn.wav"
predicted = predict_audio_class(audio_file)
print(predicted)
