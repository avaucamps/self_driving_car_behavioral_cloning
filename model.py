from BatchGenerator import BatchGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import os


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

np.random.seed(0)


def load_data(base_path):
    driving_log_path = os.path.join(base_path, 'driving_log.csv')
    df = pd.read_csv(driving_log_path, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    X, y = remove_driving_straight_bias(df[['center', 'left', 'right']].values, df['steering'].values)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_valid, y_train, y_valid


def remove_driving_straight_bias(X, y, removal_rate=0.6):
    X_unbiased = []
    y_unbiased = []
    for i in range(len(y)):
        if y[i] == 0:
            if np.random.rand() < removal_rate:
                X_unbiased.append(X[i])
                y_unbiased.append(y[i])
        else:
            X_unbiased.append(X[i])
            y_unbiased.append(y[i]) 

    return np.asarray(X_unbiased), np.asarray(y_unbiased)


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Dropout((0.5)))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    return model


def train_model(model, base_path, batch_generator, n_epochs, batch_size, train_step, data):
    X_train, X_valid, y_train, y_valid = data
    checkpoint = ModelCheckpoint(os.path.join(base_path, 'model.h5'),
                                 monitor='val_loss',
                                 verbose=0,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))
    model.fit_generator(batch_generator.generate_batch(X_train, y_train, batch_size, is_training=True),
                        train_step,
                        n_epochs,
                        validation_data=batch_generator.generate_batch(X_valid, y_valid, batch_size, is_training=True),
                        validation_steps=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)


def main():
    base_path = os.path.dirname(os.path.realpath(__file__))
    n_epochs = 3
    batch_size = 40
    train_step = 20000
    batch_generator = BatchGenerator(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

    data = load_data(base_path)
    model = build_model()
    train_model(model, base_path, batch_generator, n_epochs, batch_size, train_step, data)


if __name__ == '__main__':
    main()