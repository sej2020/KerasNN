import tensorflow as tf
import numpy as np
x = tf.constant([[1.,2.,3.],[4.,5.,6.]])
# print(x@tf.transpose(x))

a = np.array([2.,4.,5.])
# print(tf.constant(a))

t2 = tf.constant(40., dtype=tf.float64)

v = tf.Variable([[1.,2.,3.],[4.,5.,6.]])
v.assign(2*v)
print(v)
v[0,1].assign(42)
print(v)
v[:,2].assign([0.,1.])
print(v)
v.scatter_nd_update(indices=[[0,0], [1,2]], updates=[100.,200.])

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

precision = tf.keras.metrics.Precision()
precision([0,1,1,1,0,1,0,1],[1,1,0,1,0,1,0,1])
precision([0,1,0,0,1,0,1,1],[1,0,1,1,0,0,0,0])
print(precision.result())

print(precision.variables)

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