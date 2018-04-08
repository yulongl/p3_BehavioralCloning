import os
import csv
from PIL import Image
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Cropping2D, Conv2D
import matplotlib.pyplot as plt


samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('\\')[-1]
                center_image = Image.open(name)
                center_image = np.asarray(center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


learning_rate = 0.001
epoch = 3
batch_size = 64

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((60, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation="relu"))  # 65 320 => 31 158
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation="relu"))  # 31 158 => 14 77
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation="relu"))  # 14 77 => 5 37
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation="relu"))  # 5 37 => 3 35
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation="relu"))  # 3 35 => 1 33
model.add(Flatten())
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# opt = keras.optimizers.Adam(lr=learning_rate)
opt = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mse', optimizer=opt)
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),
                                     nb_epoch=epoch,
                                     verbose=1)

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')

#model = load_model('my_model.h5')