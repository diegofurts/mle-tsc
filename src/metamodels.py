from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from utils import check_unilabel
import numpy as np
from joblib import dump

class MetaKNeighbors(object):

    def __init__(self):

        self.model = KNeighborsClassifier()

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)

    def save(self, path):

        dump(self.model, path+'.joblib')

class MetaRandomForest(object):

    def __init__(self):

        self.model = RandomForestClassifier()

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)

    def save(self, path):

        dump(self.model, path+'.joblib')

class MetaLogisticRegression(object):

    def __init__(self, n_clfs):

        self.models = [LogisticRegression() for i in range(n_clfs)]
        self.n = n_clfs

    def fit(self, X, y):

        for model, j in zip(self.models, [i for i in range(self.n)]):
            model.fit(X, check_unilabel(y[:, j]))

    def predict(self, X):

        preds = []
        for model in self.models:
            preds.append(model.predict(X))

        return np.array(preds).T

    def save(self, path):

        for i, model in enumerate(self.models):
            dump(model, path+'_'+str(i)+'.joblib')

class MetaLSTM(object):

    def __init__(self, len_ts, n_clfs):

        self.n = n_clfs
        self.len_ts = len_ts
        self.model = self.get_model()

    def get_model(self):

        inps = Input(shape=(self.len_ts, 1,))
        lstm = LSTM(4, recurrent_activation='tanh', return_sequences=False)(inps)
        out = Dense(self.n, activation='sigmoid')(lstm)

        model = Model(inputs=inps, outputs=out)
        model.compile(optimizer='RMSprop', loss='binary_crossentropy')

        return model

    def fit(self, X, y):

        self.model.fit(X, y, epochs=10, batch_size=32, verbose=False)

    def predict(self, X):

        pred = self.model.predict(X)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    def save(self, path):

        self.model.save(path+'.h5')

# class MetaLSTM_CustomLoss(object):
#
#     def __init__(self, len_ts, n_clfs):
#
#         self.n = n_clfs
#         self.len_ts = len_ts
#         self.model = self.get_model()
#
#     def customloss(y_true, y_pred):
#
#
#     def get_model(self):
#
#         inps = Input(shape=(self.len_ts, 1,))
#         lstm = LSTM(4, recurrent_activation='tanh', return_sequences=False)(inps)
#         out = Dense(self.n, activation='sigmoid')(lstm)
#
#         model = Model(inputs=inps, outputs=out)
#         model.compile(optimizer='RMSprop', loss=customloss)
#
#         return model
#
#     def fit(self, X, y):
#
#         self.model.fit(X, y, epochs=10, batch_size=32, verbose=False)
#
#     def predict(self, X):
#
#         pred = self.model.predict(X)
#         pred[pred >= 0.5] = 1
#         pred[pred < 0.5] = 0
#         return pred
