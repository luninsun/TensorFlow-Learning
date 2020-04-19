import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np

(x, y), (x_test, y_test) = datasets.mnist.load_data()

x_train = np.concatenate((x, x_test), axis = 0)
y_train = np.concatenate((y, y_test), axis = 0)
print(x_train.shape)
print(y_train.shape)