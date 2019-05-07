from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten
from keras.optimizers import Adamax

from sklearn.model_selection import train_test_split

"""
Bidirectional LSTM neural network
Structure consists of two hidden layers and a BLSTM layer
Parameters, as from the VulDeePecker paper:
    Nodes: 300
    Dropout: 0.5
    Optimizer: Adamax
    Batch size: 64
    Epochs: 4
"""
class BLSTM:
    def __init__(self, data, length=100, name=""):
        train, test = train_test_split(data, test_size=0.2)
        self.training_set = train
        self.test_set = test
        self.length = length
        self.name = name
        model = Sequential()
        model.add(Bidirectional(LSTM(300, return_sequences=True, dropout=0.5), input_shape=(self.length,1)))
        model.add(Flatten())
        model.add(Dense(300, activation='linear'))
        model.add(Dropout(0.5))
        model.add(Dense(300, activation='linear'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='relu'))
        # Lower learning rate to prevent divergence
        adamax = Adamax(lr=0.001)
        model.compile(adamax, 'binary_crossentropy', metrics=['accuracy'])
        self.model = model

    """
    Trains model based on training data
    """
    def train(self):
        # Reshape array of arrays into 2D array, then 3D array
        vectors = np.stack(self.training_set.iloc[:,0].values)
        vectors = np.reshape(vectors, (vectors.shape[0], vectors.shape[1], 1))
        labels = self.training_set.iloc[:,1].values
        batch_size = 64
        self.model.fit(vectors, labels, batch_size=batch_size, epochs=4)
        self.model.save_weights(self.name + "_model.h5")

    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def test(self):
        if not self.model.get_weights():
            self.model.load_weights(self.name + "_model.h5")
        vectors = np.stack(self.test_set.iloc[:,0].values)
        vectors = np.reshape(vectors, (vectors.shape[0], vectors.shape[1], 1))
        labels = self.test_set.iloc[:,1].values
        values = self.model.evaluate(vectors, labels, batch_size=64)
        print("Accuracy is...", values[1])
