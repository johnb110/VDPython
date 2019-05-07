from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten

from sklearn.model_selection import train_test_split

class BLSTM:
    def __init__(self, data, length=100):
        train, test = train_test_split(data, test_size=0.2)
        self.training_set = train
        self.test_set = test
        self.length = length

    def train(self):
        vectors = np.stack(self.training_set.iloc[:,0].values)
        vectors = np.reshape(vectors, (vectors.shape[0], vectors.shape[1], 1))
        #print(vectors)
        labels = self.training_set.iloc[:,1].values
        batch_size = 64
        model = Sequential()
        model.add(Bidirectional(LSTM(300, return_sequences=True), input_shape=(self.length,1)))
        model.add(Dropout(0.5))
        #model.add(Flatten())
        model.add(Bidirectional(LSTM(300, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation='softmax'))
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        model.fit(vectors, labels, batch_size=batch_size, epochs=4)
        self.model = model

    def test(self):
        self.model.evaluate()
        print("Accuracy is...")

"""
max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
validation_data=[x_test, y_test])
"""