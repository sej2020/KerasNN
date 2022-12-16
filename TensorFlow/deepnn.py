import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal

mnist_train, mnist_valid, mnist_test= tfds.load(name="mnist", shuffle_files=True, split=["train[20%:]", "train[0%:20%]", "test"], batch_size=32, as_supervised=True)

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[28,28], activation="relu", model = keras.models.Sequential()):
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons,activation=activation))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_clsfr = keras.wrappers.scikit_learn.KerasRegressor(build_model)

param_distribs = {"n_hidden": [0, 1, 2, 3], "n_neurons": np.arange(1,100), "learning_rate": reciprocal(3e-4,3e-2), "activation": ["relu", "sigmoid", "tanh", "selu", "elu"]}

rnd_searc_cv = RandomizedSearchCV(keras_clsfr, param_distribs, n_iter=20, cv=5)
rnd_searc_cv.fit(mnist_train, epochs=100, validation_data=mnist_valid, callbacks=[keras.callbacks.EarlyStopping(patience=10)])


# x = tf.constant([[1.,2.,3.],[4.,5.,6.]])
# # print(x@tf.transpose(x))

# a = np.array([2.,4.,5.])
# # print(tf.constant(a))

# t2 = tf.constant(40., dtype=tf.float64)

# v = tf.Variable([[1.,2.,3.],[4.,5.,6.]])
# v.assign(2*v)
# print(v)
# v[0,1].assign(42)
# print(v)
# v[:,2].assign([0.,1.])
# print(v)
# v.scatter_nd_update(indices=[[0,0], [1,2]], updates=[100.,200.])

# x = tf.range(10)
# dataset = tf.data.Dataset.from_tensor_slices(x)
# dataset = dataset.repeat(3).batch(7)
# dataset = dataset.map(lambda x: x*2)
# dataset = dataset.apply(tf.data.experimental.unbatch())
# dataset = dataset.filter(lambda x: x < 10)
# for item in dataset:
#     print(item)

# x_mean, x_std = [tf.constant([10., 4., 5., 4.3, 4345., 34., 12., .0222]),tf.constant([2., 23., 34., 456., 56., 57., 23., 2.,])]
# n_inputs = 8

# def preprocess(line):
#     defs = [0.]*n_inputs + [tf.constant([], dtype=tf.float32)]
#     fields = tf.io.decode_csv(line, record_defaults=defs)
#     x = tf.stack(fields[:-1])
#     y = tf.stack(fields[-1:])
#     return (x-x_mean) / x_std, y

means = np.mean(x_train, axis=0, keepdims=True)
stds = np.std(x_train, axis= 0, keepdims=True)
eps = tf.keras.backend.epsilon()
model = tf.keras.models.Sequential([tf.keras.layers.Lambda(lambda inputs: (inputs - means)/(stds+eps))])

class Standardization(tf.keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)
    def call(self, inputs):
        return (inputs-self.means_)/(self.stds_ + tf.keras.backend.epsilon())

def create_huber(threshold=1.0):
    def huber_fun(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error)/2
        linear_loss = threshold*tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fun

class HuberLoss(tf.keras.losses.Loss):

    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error)/2
        linear_loss = self.threshold*tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

def my_softplus(z):
    return tf.math.log(tf.exp(z)+1.0)

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01*weights))

class MyL1Regularizer(tf.keras.regularizers.Regularizer):
    
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self,weights):
        return tf.reduce_sum(tf.abs(self.factor*weights))
    
    def get_config(self):
        return {"factor": self.factor}

def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)

# precision = tf.keras.metrics.Precision()
# precision([0,1,1,1,0,1,0,1],[1,1,0,1,0,1,0,1])
# precision([0,1,0,0,1,0,1,1],[1,0,1,1,0,0,0,0])
# print(precision.result())

# print(precision.variables)

class HuberMetric(tf.keras.metrics.Metric):

    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fun = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fun(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total/self.count
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
    
exponential_layer = tf.keras.layers.Lambda(lambda x: tf.exp(x))


class MyDense(tf.keras.layers.Layer):

    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, batch_input_shape):
        self.kernel = self.add_weight(name="kernel", shape=[batch_input_shape[-1], self.units], initializer="glorot_normal")
        self.bias = self.add_weight(name="bias", shape=[self.units], intializer="zeros")
        super().build(batch_input_shape)
    
    def call(self, x):
        return self.activation(x@self.kernel + self.bias)
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, "activation": tf.keras.activations.serialize(self.activation)}

class MyGaussianNoise(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, x, training=None):
        if training:
            noise = tf.random.normal(tf.shape(x), stddev=self.stddev)
            return x + noise
        else:
            return x
        
    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="elu", kernel_initializer="he_normal") for _ in range(n_layers)]
        
    def call(self, inputs):
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        return inputs + z

class ResidualRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out= tf.keras.Dense(output_dim)

    def call(self, inputs):
        z = self.hidden1(inputs)
        for _ in range(1 + 3):
            z = self.block1(z)
        z = self.block2(z)
        return self.out(z)

class ReconstructingRegressor(tf.keras.Model):

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal") for _ in range(5)]
        self.out = tf.keras.layers.Dense(output_dim)

    def build(self, batch_input_shape):
        n_inputs  = batch_input_shape[-1]
        self.reconstruct = tf.keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)

    def call(self, inputs):
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        reconstruction = self.reconstruct(z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05*recon_loss)
        return self.out(z)





