import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import Perceptron
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.activations import relu, selu
from tensorflow.keras.layers import Activation, LeakyReLU, PReLU
from tensorflow.keras.backend import clear_session

# clear tf backend
clear_session()

# get MNIST data and prepare for model
X, y = fetch_openml('mnist_784', return_X_y=True)
X = np.asarray(X)
y = np.asarray(y)

# normalise and convert types
X_full = X / 255.0
y_full = y.astype('int64')

# create train, val and test set
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full,
                                                    test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.1)


# tensorboard log dir getter
def get_tb_dir():
    curr_dir = os.path.join(os.curdir, "tensorboard_logs")
    tb_dir = time.strftime("model_%Y_%m_%d-%H-%M-%S")
    return os.path.join(curr_dir, tb_dir)


# create callbacks for model training
tensorboard_cb = TensorBoard(get_tb_dir())
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
callbacks = [tensorboard_cb, early_stopping_cb]

# Epochs and batch size
epochs = 20
batch_size = 32

# Exponentialy decaying learning rate
s = epochs * len(X_train) // batch_size
exp_decay_lr = ExponentialDecay(0.01, s, 0.1)

# Adam model
adam_model = Sequential()
adam_model.add(Dense(100, activation="relu"))
adam_model.add(BatchNormalization(input_shape=[784]))
for i in range(20):
    adam_model.add(Dense(50, activation="relu"))
    adam_model.add(BatchNormalization())
adam_model.add(Dense(10, activation="softmax"))
adam_model.compile(loss="sparse_categorical_crossentropy",
                   optimizer=Adam(exp_decay_lr), metrics=["accuracy"])

# fit model with minibatches
history = adam_model.fit(X_train, y_train,
						 epochs=epochs, batch_size=batch_size,
                         validation_data=(X_val, y_val),
                         callbacks=callbacks)

adam_model.save('models/explrdecaymodel')

adam_model.evaluate(X_test, y_test)
