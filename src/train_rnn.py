import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

import os
import random
import datetime

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping

# Preprocessing
from .preprocessing.rnn_preprocessing import *



def reset_random_seeds():
    '''
    Reset random seeds for reproducibility
    '''
    os.environ['PYTHONHASHSEED']=str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)


def train_rnn(model, model_name, split, batch_size=None):
    '''
    Train a RNN model for a given split.

    Parameters
    ----------
    model : keras model
        The model to train.
    model_name : str
        The name of the model.
    split : int
        The split to train on.
    
    Returns
    -------
    float
        The validation AUPRC score of the model during training.
    '''
    reset_random_seeds()

    stop_early = EarlyStopping(monitor='val_auprc', patience=20, restore_best_weights=True, mode='max')

    log_dir = f'logs/hyperparam_opt/{model_name}/split_{split}/date_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [stop_early, tensorboard_callback]

    # Data
    X_train, y_train, X_val, y_val, X_test, y_test, X_train_static, X_val_static, X_test_static = get_data(split, prior_max_len=173)

    # Model
    history = model.fit(X_train, y_train, epochs=200, batch_size=batch_size, validation_data=(X_val, y_val), verbose=2, shuffle=True, callbacks=callbacks)

    return history.history['val_auprc'][-1]


def test_rnn(model, model_name, split, batch_size=64, early_stopping_monitor='val_auprc'):
    '''
    Test a RNN model for a given split. The model is trained on the train set and evaluated on the test set. 
    The model is trained until the early_stopping_monitor metric stops improving.
    Returns AUPRC and AUCROC scores and the model history for each hour in the horizon.

    Parameters
    ----------
    model : keras model
        The model to test.
    model_name : str
        The name of the model.
    split : int
        The split to test on.
    
    Returns
    -------
    float
        The test AUPRC scores of the model on each hour in the horizon.
    float
        The test AUCROC scores of the model on each hour in the horizon.
    list
        The history of the model during training.
    '''
    reset_random_seeds()

    if early_stopping_monitor == 'val_auprc':
        stop_early = EarlyStopping(monitor=early_stopping_monitor, patience=50, restore_best_weights=True, mode='max')
    elif early_stopping_monitor == 'val_loss':
        stop_early = EarlyStopping(monitor=early_stopping_monitor, patience=50, restore_best_weights=True, mode='min')

    log_dir = f'logs/train/{early_stopping_monitor}/{model_name}/split_{split}/date_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [stop_early, tensorboard_callback]

    X_train, y_train, X_val, y_val, X_test, y_test, X_train_static, X_val_static, X_test_static = get_data(split, 173)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=200, shuffle=True, callbacks=callbacks)

    auprc_scores = {horizon:0 for horizon in range(7, -1, -1)}
    aucroc_scores = {horizon:0 for horizon in range(7, -1, -1)}

    for horizon in range(7, -1, -1):
        X_test_horizon_path = f'input/rnn/test/X_test_split_{split}_horizon_{horizon}.npy'
        X_test_horizon = load_test_data(X_test, horizon, X_test_horizon_path)

        # Predict
        test_loss, test_auprc, test_aucroc = model.evaluate(X_test_horizon, y_test)
        auprc_scores[horizon] = test_auprc
        aucroc_scores[horizon] = test_aucroc
        
    return auprc_scores, aucroc_scores, history

