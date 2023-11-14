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
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)



(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = (X_train - 0.0) / (255.0 - 0.0)
X_test = (X_test - 0.0) / (255.0 - 0.0)


#define model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(10, activation="softmax")
])

optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(
    optimizer=optimizer, 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

## train model on full data

model.fit(X_train, y_train, epochs=10, batch_size=32)
predictions = np.argmax(model.predict(X_test), axis=-1)
print('The model accuracy with full data =', accuracy_score(y_test, predictions))



# Reduce
X_train = X_train.reshape((X_train.shape + (1,)))
A=np.reshape(X_train, (60000,784))
data=np.concatenate((A, np.atleast_2d(y_train).T), axis=1)

srwt = SampleReductionWithTypicality(batch_size=60000, verbose=True)
X_final_large = srwt.reduce(data)
log.info(f"Start number of rows: {60000}. End number of rows: {X_final_large.shape}")
log.info("Testing SampleReductionWithTypicality class")


## train model with the reduced input data

y_train_red=X_final_large[:,-1]
X_train_red=X_final_large[:,:-1].reshape(X_final_large.shape[0],28,28,1)

model.fit(X_train_red, y_train_red, epochs=10, batch_size=32)

predictions_red = np.argmax(model.predict(X_test), axis=-1)
print('The model accuracy with reduced data =', accuracy_score(y_test, predictions_red))





