import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from utils import *
from tqdm import tqdm
import sys

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

metamodels = [
              'MetaKNeighbors',
              'MetaRandomForest',
              'MetaLogisticRegression',
              'MetaLSTM'
              ]

dataset = sys.argv[1]

_, y_test = load_from_tsfile_to_dataframe('../datasets/Univariate_ts/'+dataset+'/'+dataset+'_TEST.ts')
y_test = pd.Series(y_test, dtype='float64', name='y_true_test')

classifiers_test_predictions = pd.DataFrame()

best_individual_classifier = {}

for classifier in classifiers:

    test_predictions = pd.read_csv('../datasets/Univariate_ts/'+dataset+'/'+classifier+'_PREDICTION_TEST.csv')

    individual_metric = get_metrics(y_test.astype(int), test_predictions)

    if 'acc' not in best_individual_classifier.keys():
        best_individual_classifier['classifier'] = classifier
        best_individual_classifier['metrics'] = individual_metric

    else:
        if best_individual_classifier['acc'] < individual_metric['acc']:
            best_individual_classifier['classifier'] = classifier
            best_individual_classifier['metrics'] = individual_metric

    classifiers_test_predictions = pd.concat([classifiers_test_predictions, test_predictions], axis=1)

best_individual_classifier['metrics']['draws'] = 0
best_individual_classifier['metrics']['full-blank-predictions'] = 0
best_individual_classifier['metrics']['classifiers-used'] = 1
all_metrics = pd.DataFrame(best_individual_classifier['metrics'].rename(best_individual_classifier['classifier']))

naive_predictions, n_draws = naive_ensemble(classifiers_test_predictions)
metrics = get_metrics(y_test.astype(int), naive_predictions)
metrics['draws'] = n_draws
metrics['full-blank-predictions'] = 0
metrics['classifiers-used'] = len(classifiers)
all_metrics = pd.concat([all_metrics, pd.DataFrame(metrics.rename('NaiveEnsemble'))], axis=1)

for metamodel in metamodels:

    mask = pd.read_csv('../datasets/Univariate_ts/'+dataset+'/'+metamodel+'_PREDICTION_TEST.csv')
    meta_predictions, draws = ensemble_predictions(mask, classifiers_test_predictions)


    metrics = get_metrics(y_test.astype(int), meta_predictions)
    metrics['draws'] = draws
    metrics['full-blank-predictions'] = get_blank_predictions(mask)
    metrics['classifiers-used'] = get_n_classifiers_used(mask)
    all_metrics = pd.concat([all_metrics, metrics.rename(metamodel)], axis=1)

all_metrics.T.to_excel('../reports/'+dataset+'.xlsx')
# all_metrics.T.to_csv('../reports/'+dataset+'.csv')
