import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.classification.dictionary_based import BOSSEnsemble, BOSSIndividual, TemporalDictionaryEnsemble, IndividualTDE
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier # ElasticEnsemble with problems
from sktime.classification.distance_based._shape_dtw import ShapeDTW
from sktime.classification.frequency_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import TimeSeriesForest
# ShapeletTransformClassifier too slow!
# from sktime.classification.shapelet_based import ShapeletTransformClassifier
from utils import *
from tqdm import tqdm
import sys

classifiers = {
               'BOSSEnsemble':BOSSEnsemble(),
               'BOSSIndividual':BOSSIndividual(),
               'TemporalDictionaryEnsemble':TemporalDictionaryEnsemble(),
               'IndividualTDE':IndividualTDE(),
               'KNeighborsTimeSeriesClassifier':KNeighborsTimeSeriesClassifier(n_neighbors=1),
               'ShapeDTW':ShapeDTW(n_neighbors=1),
               'RandomIntervalSpectralForest':RandomIntervalSpectralForest(),
               'TimeSeriesForest':TimeSeriesForest()
               }

dataset = sys.argv[1]

X_train, y_train = load_from_tsfile_to_dataframe('../datasets/Univariate_ts/'+dataset+'/'+dataset+'_TRAIN.ts')
y_train = pd.Series(y_train)

X_test, y_test = load_from_tsfile_to_dataframe('../datasets/Univariate_ts/'+dataset+'/'+dataset+'_TEST.ts')

clfs_predictions = pd.DataFrame()

for name, classifier in tqdm(classifiers.items()):

    batches = create_idxs_batches(X_train.index.to_list())
    labels = y_train.unique()
    labels.sort()
    predictions_train = pd.Series(dtype='object')

    for batch in batches:

        batches_copy = batches.copy()
        batches_copy.remove(batch)

        X_train_batch, y_train_batch = remove_idxs(X_train, y_train, batch)

        classifier.fit(X_train_batch.reset_index(drop=True), y_train_batch.reset_index(drop=True))
        batch_prediction = pd.DataFrame(classifier.predict_proba(X_train.loc[batch]), index=batch, columns=labels)

        predictions_train = pd.concat([predictions_train, batch_prediction])

    predictions_train.drop(0, axis=1, inplace=True)
    predictions_train.to_csv('../datasets/Univariate_ts/'+dataset+'/'+name+'_PREDICTION_TRAIN.csv', index=False)


    classifier.fit(X_train, y_train)
    predictions_test = pd.DataFrame(classifier.predict(X_test), index=X_test.index, columns=['y_pred_'+name])
    predictions_test.to_csv('../datasets/Univariate_ts/'+dataset+'/'+name+'_PREDICTION_TEST.csv', index=False)
