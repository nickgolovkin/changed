from __future__ import absolute_import
from six import with_metaclass

from keras.models import Sequential

from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import LSTM

from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation

from bulbea.learn.models import Supervised

class ANN(Supervised):
    pass

class RNNCell(object):
    RNN  = SimpleRNN
    GRU  = GRU
    LSTM = LSTM

class RNN(ANN):
    def __init__(self, sizes,
                 cell       = RNNCell.LSTM,
                 dropout    = 0.2,
                 activation = 'linear',
                 loss       = 'mse',
                 optimizer  = 'rmsprop'):
        self.model = Sequential()
        self.model.add(cell(
            units = 100,
            return_sequences = True
        ))

        for i in range(2, len(sizes) - 1):
            self.model.add(cell(sizes[i], return_sequences = False))
            self.model.add(Dropout(dropout))

        self.model.add(Dense(output_dim = sizes[-1]))
        self.model.add(Activation(activation))

        self.model.compile(loss = loss, optimizer = optimizer)

    def fit(self, X, y, *args, **kwargs):
        return self.model.fit(X, y, *args, **kwargs)

    def predict(self, X):
        return self.model.predict(X)
