import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import json
import os
import logging

import pandas as pd
import numpy as np
from scipy.special import softmax

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Preprocessing
from .preprocessing.xgboost_preprocessing import *



def auprc_xgboost(y_hat, data):
    '''
    Calculate AUPRC for XGBoost model. This function is used as a custom evaluation metric for XGBoost.

    Parameters
    ----------
    y_hat : np.array
        The predicted probabilities.
    data : xgboost.DMatrix
        The data matrix.
    
    Returns
    -------
    str
        The name of the metric.
    float
        The AUPRC score.
    '''
    y_true = data.get_label()
    y_hat = np.array([1 if x > 0.5 else 0 for x in y_hat])
    precision, recall, thresholds = precision_recall_curve(y_true, y_hat)
    auc_precision_recall = auc(recall, precision)
    return 'auprc', -auc_precision_recall


def train_model(full_labvitals, features, split, model, return_model=False, return_scores=False, weight=None):
    '''
    Train a XGBoost model for a given split.

    Parameters
    ----------
    full_labvitals : pd.DataFrame
        The labvitals data.
    features : list
        The features to use.
    split : int
        The split to train on.
    model : xgboost.XGBClassifier
        The model to train.
    
    Returns
    -------
    float
        The validation AUPRC score of the model during training.
    '''
    logging.info(f'SPLIT {split}')

    # CONFIG
    # Load parameters for the current split
    logging.info(f'Loading config')
    with open(f'./configs/paper_configs/config_split{split}.json') as json_file:
        config = json.load(json_file)
    dataset_config = config['dataset']

    for key,val in dataset_config.items():
        exec(key + '=val')

    # Load train/validation/test info for the current split
    logging.info(f'Loading tvt_info for split {split}')
    tvt_info_split = pd.read_pickle(f'output/tvt_info_split_{split}.pkl')

    # DATA
    logging.info(f'Loading data')
    X_train, X_val, X_test = split_datasets(full_labvitals, tvt_info_split)
    
    logging.info(f'Loading train data')
    if not os.path.isfile(f'input/xgboost/train/train_series_stats_split_{split}.csv'):
        X_train = get_experiments_stats(X_train, tvt_info_split['train'], ['chart_time'] + features)
        columns_to_drop = [column for column in X_train.columns if ('chart' in column) and (column != 'chart_time_max')]
        X_train = X_train.drop(columns=columns_to_drop)
        X_train.to_csv(f'input/xgboost/train/train_series_stats_split_{split}.csv', index=False)
    else:
        X_train = pd.read_csv(f'input/xgboost/train/train_series_stats_split_{split}.csv')
    
    y_train = X_train['label'].values
    X_train = X_train.drop(columns=['icustay_id', 'subject_id', 'label'])


    # MODEL
    logging.info(f'Fitting model')
    if weight is None:

        logging.info(f'Loading validation data')
        if not os.path.isfile(f'input/xgboost/validation/validation_series_stats_split_{split}.csv'):
            X_val = get_experiments_stats(X_val, tvt_info_split['validation'], ['chart_time'] + features)
            columns_to_drop = [column for column in X_val.columns if ('chart' in column) and (column != 'chart_time_max')]
            X_val = X_val.drop(columns=columns_to_drop)
            X_val.to_csv(f'input/xgboost/validation/validation_series_stats_split_{split}.csv', index=False)
        else:
            X_val = pd.read_csv(f'input/xgboost/validation/validation_series_stats_split_{split}.csv')
        
        y_val = X_val['label'].values
        X_val = X_val.drop(columns=['icustay_id', 'subject_id', 'label'])
        
        logging.info(f'Fitting model')
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=auprc_xgboost, early_stopping_rounds=10, verbose=False)

        split_validation_score = model.best_score
        evals_score = model.evals_result()
        
        if return_model:
            if return_scores:
                return abs(split_validation_score), model, 
            return abs(split_validation_score), evals_score, model 
        else:
            return abs(split_validation_score)
    else:
        auprc_scores_val = []
        evals_scores = {horizon:None for horizon in range(7, -1, -1)}
        
        for horizon in range(7, -1, -1):

            logging.info(f'Loading validation data')
            if not os.path.isfile(f'input/xgboost/validation/validation_series_stats_split_{split}_horizon_{horizon}.csv'):
                logging.info(f'Generating validation_series_stats for split {split} and horizon {horizon}')
                X_val_horizon = get_data_n_horizon(X_val, tvt_info_split['validation'], horizon)
                X_val_horizon_stats = get_experiments_stats(X_val_horizon, tvt_info_split['validation'], ['chart_time'] + features)
                columns_to_drop = [column for column in X_val_horizon_stats.columns if ('chart' in column) and (column != 'chart_time_max')]
                X_val_horizon_stats = X_val_horizon_stats.drop(columns=columns_to_drop)
                X_val_horizon_stats.to_csv(f'input/xgboost/validation/validation_series_stats_split_{split}_horizon_{horizon}.csv', index=False)
            else:
                logging.info(f'Loading data')
                X_val_horizon_stats = pd.read_csv(f'input/xgboost/validation/validation_series_stats_split_{split}_horizon_{horizon}.csv')

            y_val = X_val_horizon_stats['label'].values
            X_val_horizon_stats = X_val_horizon_stats.drop(columns=['icustay_id', 'subject_id', 'label'])

            model.fit(X_train, y_train, eval_set=[(X_val_horizon_stats, y_val)], eval_metric=auprc_xgboost, early_stopping_rounds=10, verbose=False)

            auprc_scores_val.append(model.best_score)
            evals_scores[horizon] = model.evals_result()

        if weight=='softmax':
            split_validation_score = np.average(auprc_scores_val, weights=softmax([horizon for horizon in range(7, -1, -1)]))
        elif weight=='reversed_softmax':
            split_validation_score = np.average(auprc_scores_val, weights=list(reversed(softmax([horizon for horizon in range(7, -1, -1)]))))
        elif weight=='average':
            split_validation_score = np.average(auprc_scores_val)
    
        if return_model:
            if return_scores:
                return abs(split_validation_score), None, 
            return abs(split_validation_score), evals_scores, None
        else:
            return abs(split_validation_score)



def test_model(full_labvitals, features, split, model):
    '''
    Test model on test set and return the AUPRC and AUROC scores for each hour of the prediction horizon.

    Parameters
    ----------
    full_labvitals : pd.DataFrame
        Full labvitals dataframe.
    features : list
        List of features to use.
    split : int
        Split number.
    model : xgboost.XGBClassifier
        Model to train and test.
    
    Returns
    -------
    auprc_scores : dict
        AUPRC scores for each hour of the prediction horizon.
    auroc_scores : dict
        AUROC scores for each hour of the prediction horizon.
    evals_result : dict
        Evaluation during training.
    '''
    logging.info(f'Testing model')
    logging.info(f'SPLIT {split}')

    # Load train/validation/test info for the current split
    logging.info(f'Loading tvt_info for split {split}')
    tvt_info_split = pd.read_pickle(f'output/tvt_info_split_{split}.pkl')

    X_train, X_val, X_test = split_datasets(full_labvitals, tvt_info_split)

    logging.info(f'Loading train data')
    if not os.path.isfile(f'input/xgboost/train/train_series_stats_split_{split}.csv'):
        X_train = get_experiments_stats(X_train, tvt_info_split['train'], ['chart_time'] + features)
        columns_to_drop = [column for column in X_train.columns if ('chart' in column) and (column != 'chart_time_max')]
        X_train = X_train.drop(columns=columns_to_drop)
        X_train.to_csv(f'input/xgboost/train/train_series_stats_split_{split}.csv', index=False)
    else:
        X_train = pd.read_csv(f'input/xgboost/train/train_series_stats_split_{split}.csv')

    
    y_train = X_train['label'].values
    X_train = X_train.drop(columns=['icustay_id', 'subject_id', 'label'])

    logging.info(f'Loading validation data')
    if not os.path.isfile(f'input/xgboost/validation/validation_series_stats_split_{split}.csv'):
        X_val = get_experiments_stats(X_val, tvt_info_split['validation'], ['chart_time'] + features)
        columns_to_drop = [column for column in X_val.columns if ('chart' in column) and (column != 'chart_time_max')]
        X_val = X_val.drop(columns=columns_to_drop)
        X_val.to_csv(f'input/xgboost/validation/validation_series_stats_split_{split}.csv', index=False)
    else:
        X_val = pd.read_csv(f'input/xgboost/validation/validation_series_stats_split_{split}.csv')
    
    
    y_val = X_val['label'].values
    X_val = X_val.drop(columns=['icustay_id', 'subject_id', 'label'])


    # Train model    
    logging.info(f'Fitting model')
    model.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)], eval_metric=auprc_xgboost, early_stopping_rounds=10, verbose=False)

    evals_result = model.evals_result()

    # Test model
    auprc_scores = {horizon:0 for horizon in range(7, -1, -1)}
    aucroc_scores = {horizon:0 for horizon in range(7, -1, -1)}

    for horizon in range(7, -1, -1):

        if not os.path.isfile(f'input/xgboost/test/test_series_stats_split_{split}_horizon_{horizon}.csv'):
            logging.info(f'Generating test_series_stats for split {split} and horizon {horizon}')
            X_test_horizon = get_data_n_horizon(X_test, tvt_info_split['test'], horizon)
            X_test_horizon_stats = get_experiments_stats(X_test_horizon, tvt_info_split['test'], ['chart_time'] + features)
            columns_to_drop = [column for column in X_test_horizon_stats.columns if ('chart' in column) and (column != 'chart_time_max')]
            X_test_horizon_stats = X_test_horizon_stats.drop(columns=columns_to_drop)
            X_test_horizon_stats.to_csv(f'input/xgboost/test/test_series_stats_split_{split}_horizon_{horizon}.csv', index=False)
        else:
            logging.info(f'Loading data')
            X_test_horizon_stats = pd.read_csv(f'input/xgboost/test/test_series_stats_split_{split}_horizon_{horizon}.csv')
            

        y_test = X_test_horizon_stats['label'].values
        X_test_horizon_stats = X_test_horizon_stats.drop(columns=['icustay_id', 'subject_id', 'label'])
        
        y_hat = model.predict(X_test_horizon_stats)

        precision, recall, thresholds = precision_recall_curve(y_test, y_hat)
        auc_precision_recall = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, y_hat)
        logging.info(f'HORIZON: {horizon}, AUC-PRC: {auc_precision_recall}, AUC-ROC: {roc_auc}')

        auprc_scores[horizon] = auc_precision_recall
        aucroc_scores[horizon] = roc_auc

    return auprc_scores, aucroc_scores, evals_result


