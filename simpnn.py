"""
Imports
"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Sequential API - Classification MLP
"""
# fashion_mnist = keras.datasets.fashion_mnist
# (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train_full.shape)
# print(x_train_full.dtype)

# x_valid, x_train = x_train_full[:5000]/255.0, x_train_full[5000:]/255.0
# y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# x_test = x_test/255.0

# class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28,28]))
# model.add(keras.layers.Dense(300, activation="relu"))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))

# model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))
# print(history.params)
# print(history.epoch)
# print(history.history)
# metricdf = pd.DataFrame(history.history)
# metricdf.plot()
# plt.grid(True)
# plt.gca().set_ylim(0,1)
# plt.show()

# model.evaluate(x_test, y_test)
# x_new = x_test[:3]
# y_proba = model.predict(x_new)
# y_proba = y_proba.round(2)

# print(y_proba)
# print(model.summary())
# print(model.layers)
# hidden1 = model.layers[1]
# print(hidden1.name)
# print(model.get_layer('dense') is hidden1)

# weights, biases = hidden1.get_weights()
# print(weights)
# print(weights.shape)

# print(biases)
# print(biases.shape)

"""
Sequential API - Regression MLP
"""

housing = fetch_california_housing()

x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

# model = keras.models.Sequential([keras.layers.Dense(30, activation="relu", input_shape=x_train.shape[1:]), keras.layers.Dense(1)])
# model.compile(loss="mean_squared_error", optimizer="sgd")
# history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))
# mse_test = model.evaluate(x_test, y_test)

x_new = x_test[:3]
# y_pred = model.predict(x_new)

# print(mse_test)
# print(y_pred)

"""
Functional API - Wide and Deep NN
"""

# input_ = keras.layers.Input(shape=x_train.shape[1:])
# hidden1 = keras.layers.Dense(30, activation="relu")(input_)
# hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# concat = keras.layers.Concatenate()([input_, hidden2])
# output = keras.layers.Dense(1)(concat)
# model = keras.Model(inputs=[input_], outputs=[output])

# model.compile(loss="mean_squared_error", optimizer="sgd")
# history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid,y_valid))
# metricdf = pd.DataFrame(history.history)
# metricdf.plot()
# plt.grid(True)
# plt.gca().set_ylim(0,1)
# plt.show()

# test = model.evaluate(x_test, y_test)
# print(test)
# y_pred = model.predict(x_new)
# print(y_pred)

"""
Functional API - Multi-Output
"""
# input_A = keras.layers.Input(shape=[5], name="wide_input")
# input_B = keras.layers.Input(shape=[6], name="deep_input")
# hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
# hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# concat = keras.layers.Concatenate()([input_A, hidden2])
# output = keras.layers.Dense(1, name="main_output")(concat)
# aux_output = keras.layers.Dense(1, name="auxoutput")(hidden2)
# model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])

# model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))

# x_train_A, x_train_B = x_train[:, :5], x_train[:, 2:]
# x_valid_A, x_valid_B = x_valid[:, :5], x_valid[:, 2:]
# x_test_A, x_test_B = x_test[:, :5], x_test[:, 2:]
# x_new_A, x_new_B = x_test[:3, :5], x_train[:3, 2:]

# history = model.fit([x_train_A, x_train_B], [y_train, y_train], epochs=20, validation_data=([x_valid_A, x_valid_B], [y_valid, y_valid]))
# total_loss, main_loss, aux_loss = model.evaluate([x_test_A, x_test_B], [y_test, y_test])
# y_pred_main, y_pred_aux = model.predict([x_new_A, x_new_B])

# print(total_loss, main_loss, aux_loss)
# print(y_pred)
"""
Subclassing API - Wide and Deep NN
"""
class WideAndDeepModel(keras.Model):
    
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()