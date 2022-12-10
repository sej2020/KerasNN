import tensorflow as tf
import numpy as np
x = tf.constant([[1.,2.,3.],[4.,5.,6.]])
print(x@tf.transpose(x))

a = np.array([2.,4.,5.])
print(tf.constant(a))

t2 = tf.constant(40., dtype=tf.float64)
