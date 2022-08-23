import numpy as np

import os
import json

from src.preprocessing.main_preprocessing import get_dataset

def pad_data(X, pad_value, pad_length=None):
    '''
    Pad data with pad_value to make all data have the same length.
    The length of the data is determined by the longest data if pad_length is None.

    Parameters
    ----------
    X : list of numpy arrays
        List of data to be padded.
    pad_value : int or float
        Value to pad with.
    
    Returns
    -------
    X_padded : 2D numpy array
        Padded data.
    '''
    if pad_length:
        max_length = pad_length
    else:
        max_length = max([x.shape[0] for x in X])
    
    X_padded = []
    for x in X:
        x_padded = np.full((max_length, x.shape[1]), pad_value)
        x_padded[:x.shape[0], :] = x
        X_padded.append(x_padded)
    return np.array(X_padded)


def get_labels(data, labels):
    '''
    Get single label for each data point.

    Parameters
    ----------
    data : list of numpy arrays
        List of ICU stays data.
    labels : list of numpy arrays
        List of ICU stays labels.
    
    Returns
    -------
    labels : numpy array of ints
        Labels for each data point.
    '''
    index = 0
    y = []
    for df in data:
        df_labels = labels.iloc[index:index+df.shape[0]]
        y.append(df_labels.values[0])
        index += df.shape[0]
    return np.array(y)


def get_y_classes(y):
    '''
    Get classes from probabilities.

    Parameters
    ----------
    y : numpy array of floats
        Probabilities.

    Returns
    -------
    y_classes : numpy array of ints
        Classes.
    '''
    y_classes = []
    for y_i in y:
        if y_i > 0.5:
            y_classes.append(1)
        else:
            y_classes.append(0)
    return np.array(y_classes)


def get_series_max_length():
    '''
    Get maximum length of series in all data.

    Returns
    -------
    max_length : int
        Maximum length of series.
    '''
    na_thres = 500
    datapath = 'output/'
    overwrite = 0
    horizon = 0
    data_sources = ['labs', 'vitals', 'covs']
    min_length = 7
    max_length = 200
    num_obs_thres = 10000

    series_max_length = 0
    for split in [0,1,2]:

        full_dataset = get_dataset(na_thres, datapath, overwrite, horizon, data_sources, min_length, max_length, split, num_obs_thres)

        variables, labels, train_data, validation_data, test_data, train_static_data, validation_static_data, test_static_data = full_dataset
        train_labels, validation_labels, test_labels = labels
        
        max_len_train = max([len(x) for x in train_data])
        max_len_validation = max([len(x) for x in validation_data])
        max_len_test = max([len(x) for x in test_data])

        series_max_length = max(series_max_length, max(max_len_train, max(max_len_validation, max_len_test)))
    
    return series_max_length


def get_data(split, prior_max_len=None):
    '''
    Get data for a split. The data is padded with 999 to make all data have the same length. The length of the data is determined by the longest data if prior_max_len is None.

    Parameters
    ----------
    split : int
        Split to get data for.

    Returns
    -------
    X : list of numpy arrays
        List of data.
    '''
    # CONFIG
    with open(f'./configs/paper_configs/config_split{split}.json') as json_file:
        config = json.load(json_file)
    dataset_config = config['dataset']

    na_thres = dataset_config['na_thres']
    # datapath = dataset_config['datapath']
    datapath = 'input/rnn/'
    overwrite = dataset_config['overwrite']
    horizon = dataset_config['horizon']
    data_sources = dataset_config['data_sources']
    min_length = dataset_config['min_length']
    max_length = dataset_config['max_length']
    num_obs_thres = dataset_config['num_obs_thres']

    processed_datapath_train = f'{datapath}/train/X_train_split_{split}.npy'
    processed_datapath_validation = f'{datapath}/validation/X_validation_split_{split}.npy'
    processed_datapath_test = f'{datapath}/test/X_test_split_{split}.npy'
    processed_datapath_train_static = f'{datapath}/train/X_train_static_split_{split}.npy'
    processed_datapath_validation_static = f'{datapath}/validation/X_validation_static_split_{split}.npy'
    processed_datapath_test_static = f'{datapath}/test/X_test_static_split_{split}.npy'
    processed_datapath_train_labels = f'{datapath}/train/y_train_split_{split}.npy'
    processed_datapath_validation_labels = f'{datapath}/validation/y_validation_split_{split}.npy'
    processed_datapath_test_labels = f'{datapath}/test/y_test_split_{split}.npy'
    processed_datapaths = [processed_datapath_train, processed_datapath_validation, processed_datapath_test, processed_datapath_train_static, processed_datapath_validation_static, processed_datapath_test_static, processed_datapath_train_labels, processed_datapath_validation_labels, processed_datapath_test_labels]

    processed_data_available = all(map(os.path.isfile, processed_datapaths))

    if not processed_data_available:
        full_dataset = get_dataset(na_thres, datapath, overwrite, horizon, data_sources, min_length, max_length, split, num_obs_thres)

        variables, labels, train_data, validation_data, test_data, X_train_static, X_val_static, X_test_static = full_dataset
        train_labels, validation_labels, test_labels = labels
        
        # Series data processing
        max_len_train = max([len(x) for x in train_data])
        max_len_validation = max([len(x) for x in validation_data])
        max_len_test = max([len(x) for x in test_data])

        X_train = []
        for df in train_data:
            X_train.append(df.to_numpy())
        X_train = np.array(X_train)

        X_val = []
        for df in validation_data:
            X_val.append(df.to_numpy())
        X_val = np.array(X_val)

        X_test = []
        for df in test_data:
            X_test.append(df.to_numpy())
        X_test = np.array(X_test)

        if prior_max_len:
            max_len_series = prior_max_len
        else:
            max_len_series = max(max_len_train, max(max_len_validation, max_len_test))

        X_train = pad_data(X_train, 999, max_len_series)
        X_val = pad_data(X_val, 999, max_len_series)
        X_test = pad_data(X_test, 999, max_len_series)

        # Labels data processing
        y_train = get_labels(train_data, train_labels)
        y_val = get_labels(validation_data, validation_labels)
        y_test = get_labels(test_data, test_labels)

        np.save(processed_datapath_train, X_train)
        np.save(processed_datapath_validation, X_val)
        np.save(processed_datapath_test, X_test)
        np.save(processed_datapath_train_static, X_train_static)
        np.save(processed_datapath_validation_static, X_val_static)
        np.save(processed_datapath_test_static, X_test_static)
        np.save(processed_datapath_train_labels, y_train)
        np.save(processed_datapath_validation_labels, y_val)
        np.save(processed_datapath_test_labels, y_test)
    else:
        # Load processed data
        X_train = np.load(processed_datapath_train)
        X_val = np.load(processed_datapath_validation)
        X_test = np.load(processed_datapath_test)
        X_train_static = np.load(processed_datapath_train_static)
        X_val_static = np.load(processed_datapath_validation_static)
        X_test_static = np.load(processed_datapath_test_static)
        y_train = np.load(processed_datapath_train_labels)
        y_val = np.load(processed_datapath_validation_labels)
        y_test = np.load(processed_datapath_test_labels)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_static, X_val_static, X_test_static


def get_data_n_horizon(X, n):
    '''
    Slice data until last time step minus n (n is the horizon).

    Parameters
    ----------
    X : numpy array
        Input data.
    n : int
        Horizon.
    
    Returns
    -------
    numpy array
        Data sliced until last time step minus n.
    '''
    X_horizon = []
    for x in X:
        try:
            first_index = np.where(x == 999)[0][0]
            x[first_index-n:first_index] = 999
        except:
            pass
        X_horizon.append(x)
    return np.array(X_horizon)


def load_test_data(X_test, horizon, X_test_horizon_path):
    '''
    Load test data sliced until last time step minus horizon. If the data is not available, it is sliced and saved.

    Parameters
    ----------
    X_test : numpy array
        Test data.
    horizon : int
        Horizon.
    X_test_horizon_path : str
        Path to test data sliced until last time step minus horizon.
    
    Returns
    -------
    numpy array
        Test data sliced until last time step minus horizon.
    '''
    if not os.path.isfile(X_test_horizon_path):
        X_test_horizon = []
        for x in X_test:
            X_test_horizon.append(x[:x.shape[0]-horizon])
        X_test_horizon = np.array(X_test_horizon)

        X_test_horizon = pad_data(X_test_horizon, 999, 173)
        
        np.save(X_test_horizon_path, X_test_horizon)
    else:
        X_test_horizon = np.load(X_test_horizon_path)
    return X_test_horizon
