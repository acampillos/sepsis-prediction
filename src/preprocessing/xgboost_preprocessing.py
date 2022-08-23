import pandas as pd
import numpy as np


def split_datasets(full_labvitals, datasets_icustay_ids):
    '''
    Splits the full_labvitals dataset into train, validation and test datasets using the icustay_ids (icustay_ids_train, icustay_ids_validation and icustay_ids_test).

    Parameters
    ----------
    full_labvitals : pandas.DataFrame
        The full labvitals dataset
    datasets_icustay_ids : dict
        The dictionary with the icustay_ids for each dataset (icustay_ids_train, icustay_ids_validation and icustay_ids_test)
    
    Returns
    ----------
    X_train : pandas.DataFrame
        The train dataset
    X_val : pandas.DataFrame
        The validation dataset
    X_test : pandas.DataFrame
        The test dataset
    '''
    X_train = full_labvitals[full_labvitals['icustay_id'].isin(datasets_icustay_ids['train'])]
    X_val = full_labvitals[full_labvitals['icustay_id'].isin(datasets_icustay_ids['validation'])]
    X_test = full_labvitals[full_labvitals['icustay_id'].isin(datasets_icustay_ids['test'])]
    return X_train, X_val, X_test


def get_control_sepsis_cases_ids(X, icustay_ids):
    '''
    Returns the ids of the control and the sepsis cases in the dataset.
    Cases IDs are the ones in the icustay_ids list.

    Parameters
    ----------
    X : pandas.DataFrame
        The full dataset
    icustay_ids : list
        The list of icustay_ids in the dataset
    
    Returns
    -------
    dict
        The dictionary with the control and the sepsis cases ids
    '''
    controls = []
    cases = []
    for id in icustay_ids:
        if any(X[X['icustay_id']==id]['label'].values):
            cases.append(id)
        else:
            controls.append(id)
    return {'controls': controls, 'cases': cases}


def get_time_series_data(X, icustay_ids, features):
    '''
    Returns a np.array for the given features and icustay_ids.

    Parameters
    ----------
    X : pandas.DataFrame
        The full dataset
    icustay_ids : list
        List of icustay_ids
    features : list
        List of features to get the data for
    
    Returns
    -------
    np.array
        The np.array with the time series data for the given features and icustay_ids
    '''
    time_series_data = []
    for id in icustay_ids:
        time_series_data.append(X[X['icustay_id']==id][features].values)
    return np.array(time_series_data)


def get_y(X, icustay_ids):
    '''
    Returns the y vector for the given icustay_ids. One label for each icustay_id.

    Parameters
    ----------
    X : pandas.DataFrame
        The full dataset
    icustay_ids : list
        List of icustay_ids
    
    Returns
    -------
    list
        The list of labels for the given icustay_ids
    '''
    y = []
    for id in icustay_ids:
        y.append(X[X['icustay_id']==id]['label'].values[0])
    return y


def get_serie_describe(serie):
    '''
    Returns a dictionary with its pandas describe() method values

    Parameters
    ----------
    serie : pandas.DataFrame
        The time serie to describe

    Returns
    ----------
    values : dict
        The dictionary with the describe() method values
    '''
    serie_describe = serie.describe().transpose().drop(columns=['count'])

    values = dict()

    for index, row in serie_describe.iterrows():
        for col in row.index:
            values[f'{index}_{col}'] = row[col]
    return values


def get_experiments_stats(X, icustay_ids, features):
    '''
    Creates a dataframe with the statistics of the given features for the given icustay_ids in X.
    The statistics are the describe() method values for each feature.

    Parameters
    ----------
    X : pandas.DataFrame
        Dataset
    icustay_ids : list
        List of icustay_ids
    features : list
        List of features to get the statistics for
    
    Returns
    -------
    pandas.DataFrame
        The dataframe with the statistics of the given features for the given icustay_ids in X
    '''
    time_series_stats = []

    for id in icustay_ids:
        try:
            serie_describe = get_serie_describe(X[X['icustay_id']==id][features])
            serie_describe['label'] = X[X['icustay_id']==id]['label'].values[0]
            serie_describe['icustay_id'] = id
            serie_describe['subject_id'] = X[X['icustay_id']==id]['subject_id'].values[0]
        except:
            continue
        
        time_series_stats.append(serie_describe)

    return pd.DataFrame(time_series_stats)


def get_last_indexes(X, icustay_ids):
    '''
    Returns the last index of the series associated to each icustay_id.
    The last index is the index of the last row of the series.
    The index is the index of the row in the series.

    Parameters
    ----------
    X : pandas.DataFrame
        The full dataset
    icustay_ids : list
        List of icustay_ids

    Returns
    -------
    dict
        The dictionary with the last indexes of the series associated to each icustay_id
    '''
    last_indexes = {}
    for id in icustay_ids:
        last_indexes[id] = X[X['icustay_id']==id].index[-1]
    return last_indexes


def get_data_n_horizon(X, icustay_ids, n):
    '''
    Slice data until last time step minus n (n is the horizon) for each icustay_id and feature in X.
    Returns a DataFrame with the sliced data.

    Parameters
    ----------
    X : pandas.DataFrame
        The full dataset
    icustay_ids : list
        List of icustay_ids
    n : int
        The horizon
    
    Returns
    -------
    pandas.DataFrame
        The sliced data
    '''
    dataframes = []
    for id in icustay_ids:
        last_index = X[X['icustay_id']==id].index[-1]
        dataframes.append(X[X['icustay_id']==id].loc[:last_index-n])
    return pd.concat(dataframes)


# def get_y_n_horizon(X, icustay_ids, n):
#     y = []
#     for id in icustay_ids:
#         last_index = X[X['icustay_id']==id].index[-1]
#         y.append(X[X['icustay_id']==id].loc[:last_index-n]['label'])
#     return np.array(pd.concat(y))