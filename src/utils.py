import pandas as pd
import numpy as np
from random import shuffle
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def remove_idxs(X_df, y_series, idx_list):
    return X_df.drop(idx_list, axis=0), y_series.drop(idx_list)

def create_idxs_batches(idx_list_complete):

    shuffle(idx_list_complete)
    tam_batches = len(idx_list_complete) // 10

    batches = []
    for i in range(0, len(idx_list_complete), tam_batches):
        batches.append(idx_list_complete[i:i+tam_batches])

    return batches

def transform_df(nested_X_df):
    return nested_X_df.apply(lambda x: pd.Series(x[0]), axis=1)

def get_label(score_predictions):
    return pd.Series([score_predictions[np.argmax(score_predictions)], np.argmax(score_predictions) + 1])

def get_score_true_label(score_predictions):
    return score_predictions[str(int(score_predictions['y_true']))]

def choose_clfs(scores):
    max_score = max(scores)
    return list(scores[scores == max_score].index)

def create_dict_encoder_decoder(list_clfs):
    return {clf:i for i, clf in enumerate(list_clfs)}, {i:clf for i, clf in enumerate(list_clfs)}

def detect_draw(frequencies):

    max_value = max(frequencies.values())

    counter = 0
    for value in frequencies.values():
        if value == max_value:
            counter += 1

    return counter > 1

def majority_voting(masked_labels, raw_labels):

    if not masked_labels.any():

        frequency_labels = Counter(raw_labels)

    else:

        frequency_labels = Counter(masked_labels[masked_labels != 0])

    return max(frequency_labels, key=frequency_labels.get), detect_draw(frequency_labels)

def ensemble_predictions(masks, clfs_predictions):

    mask_applied_on_predictions = masks.values * clfs_predictions.values

    preds = []
    draws = []
    for i in range(mask_applied_on_predictions.shape[0]):

        pred, draw_flag = majority_voting(mask_applied_on_predictions[i], clfs_predictions.values[i])

        preds.append(pred)
        draws.append(draw_flag)

    return pd.Series(preds, index=clfs_predictions.index), sum(draws) / len(draws)

def naive_major_voting(raw_labels):

    frequency_labels = Counter(raw_labels)

    return max(frequency_labels, key=frequency_labels.get), detect_draw(frequency_labels)

def naive_ensemble(clfs_predictions):

    preds = []
    draws = []
    for i in range(clfs_predictions.shape[0]):

        pred, draw_flag = naive_major_voting(clfs_predictions.values[i])
        preds.append(pred)
        draws.append(draw_flag)

    return pd.Series(preds, index=clfs_predictions.index), sum(draws) / len(draws)

def get_metrics(y_true, y_pred):

    return pd.Series({'acc':accuracy_score(y_true, y_pred), 'balanced_acc':balanced_accuracy_score(y_true, y_pred),
                      'f1':f1_score(y_true, y_pred), 'weighted_f1':f1_score(y_true, y_pred, average='weighted')})

def get_blank_predictions(predictions):

    return (predictions.values == 0).all(axis=1).sum() / predictions.shape[0]

def check_unilabel(y):

    if  (y == y[0]).all():
        y[0] = 1 if y[0] == 0 else 0

    return y

def get_n_classifiers_used(predictions):

    return predictions.values.sum(axis=1).mean()
