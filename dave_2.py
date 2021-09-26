from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
import utils


def build_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), padding='same', strides=(2, 2), input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(36, (5, 5), padding='same', strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(48, (5, 5), padding='same', strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.summary()

    model.compile(optimizer=Adam(1e-4), loss="mse")
    
    return model

train_generator = utils.generate_batch()
valid_generator = utils.generate_batch()

model = build_model()

model.fit_generator(
    train_generator,
    steps_per_epoch=20032//64,
    epochs=5,
    validation_data=valid_generator,
    validation_steps=6400//64
)