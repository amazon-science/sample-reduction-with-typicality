import numpy as np
import pandas as pd
import random
import tensorflow as tf

from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from src.sample_reduction_with_typicality import SampleReductionWithTypicality



(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape + (1,)))
A=np.reshape(X_train, (60000,784))
data=np.concatenate((A, np.atleast_2d(y_train).T), axis=1)


# Reduce
srwt = SampleReductionWithTypicality(batch_size=60000, verbose=True)
X_final_large = srwt.reduce(data)
log.info(f"Start number of rows: {n}. End number of rows: {X_final_large.shape}")
log.info("Testing SampleReductionWithTypicality class")

