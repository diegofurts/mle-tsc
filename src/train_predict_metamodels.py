import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sklearn.ensemble import RandomForestClassifier
from utils import *
from metamodels import *
from tqdm import tqdm
import sys
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

classifiers = [
               'BOSSEnsemble',
               'BOSSIndividual',
               'TemporalDictionaryEnsemble',
               'IndividualTDE',
               'KNeighborsTimeSeriesClassifier',
               'ShapeDTW',
               'RandomIntervalSpectralForest',
               'TimeSeriesForest'
               ]

dataset = sys.argv[1]

X_train, y_train = load_from_tsfile_to_dataframe('../datasets/Univariate_ts/'+dataset+'/'+dataset+'_TRAIN.ts')
X_train = transform_df(X_train)
y_train = pd.Series(y_train, dtype='float64', name='y_true')

X_test, y_test = load_from_tsfile_to_dataframe('../datasets/Univariate_ts/'+dataset+'/'+dataset+'_TEST.ts')
X_test = transform_df(X_test)
y_test = pd.Series(y_test, dtype='float64', name='y_true_test')

metamodels = {
              'MetaKNeighbors':MetaKNeighbors(),
              'MetaRandomForest':MetaRandomForest(),
              'MetaLogisticRegression':MetaLogisticRegression(len(classifiers)),
              'MetaLSTM':MetaLSTM(X_train.shape[1], len(classifiers))
              }

encoder, decoder = create_dict_encoder_decoder(classifiers)
mlb = MultiLabelBinarizer()
mlb.fit([[i] for i in range(len(classifiers))])

all_scores_true_label = pd.DataFrame()

for clf in classifiers:

    scores = pd.read_csv('../datasets/Univariate_ts/'+dataset+'/'+clf+'_PREDICTION_TRAIN.csv')
    scores_true_label = pd.concat([scores, y_train], axis=1).apply(get_score_true_label, axis=1)
    scores_true_label.rename(clf, inplace=True)
    all_scores_true_label = pd.concat([all_scores_true_label, scores_true_label], axis=1)

y_metamodel_train = all_scores_true_label.apply(choose_clfs, axis=1)

def encode(choosen_clfs):
    return [encoder[clf] for clf in choosen_clfs]

y_metamodel_train = mlb.transform(y_metamodel_train.apply(encode))

metas_predictions = {}
for name, metamodel in metamodels.items():

    metamodel.fit(X_train, y_metamodel_train)
    metas_predictions[name] = pd.DataFrame(metamodel.predict(X_test), index=X_test.index, columns=classifiers)
    metas_predictions[name].to_csv('../datasets/Univariate_ts/'+dataset+'/'+name+'_PREDICTION_TEST.csv', index=False)
