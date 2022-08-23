import keras

import tensorflow as tf

from keras.models import Model
from keras.layers import LSTM, Dense, Masking, Dropout, GRU
from keras.optimizers import Adam, SGD
from keras.activations import *



def model_compile(model, optimizer):
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
            metrics=[
                tf.keras.metrics.AUC(curve='PR', num_thresholds=200, name='auprc'),
                tf.keras.metrics.AUC(curve='ROC', num_thresholds=200, name='aucroc')
            ],
            # run_eagerly=True
            ) 
    return model



## Base-lines
def build_lstm_baseline(optimizer, learning_rate, num_rnn_nodes):
    '''
    Build a LSTM model with a single LSTM layer and a single dense layer

    Parameters
    ----------
    optimizer : str
        Optimizer to use for training
    learning_rate : float
        Learning rate for the optimizer
    num_rnn_nodes : int
        Number of nodes in the LSTM layer
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model
    '''
    input_layer = keras.Input(shape=(173, 44))
    masking_layer = Masking(mask_value=999)(input_layer)

    lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=False)(masking_layer)

    output_layer = Dense(1, activation='sigmoid')(lstm_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate, clipnorm=1.0)
    
    return model_compile(model, optimizer)

def build_gru_baseline(optimizer, learning_rate, num_rnn_nodes):
    '''
    Build a GRU model with a single GRU layer and a single dense layer

    Parameters
    ----------
    optimizer : str
        Optimizer to use for training
    learning_rate : float
        Learning rate for the optimizer
    num_rnn_nodes : int
        Number of nodes in the GRU layer
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model
    '''
    input_layer = keras.Input(shape=(173, 44))
    masking_layer = Masking(mask_value=999)(input_layer)

    gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=False)(masking_layer)

    output_layer = Dense(1, activation='sigmoid')(gru_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate, clipnorm=1.0)
    
    return model_compile(model, optimizer)

## Stacked dense layers
def build_lstm_denses_stacked(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, activation):
    '''
    Build a LSTM model with a single LSTM layer and stacked dense layers

    Parameters
    ----------
    optimizer : str
        Optimizer to use for training
    learning_rate : float
        Learning rate for the optimizer
    num_layers : int
        Number of dense layers to stack
    num_rnn_nodes : int
        Number of nodes in the LSTM layer
    num_dense_nodes : int
        Number of nodes in the dense layers
    activation : str
        Activation function to use in the dense layers

    Returns
    -------
    model : keras.Model
        Compiled Keras model
    '''
    input_layer = keras.Input(shape=(173, 44))
    masking_layer = Masking(mask_value=999)(input_layer)

    lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=False)(masking_layer)

    dense_layer = Dense(num_dense_nodes, activation=activation)(lstm_layer)

    for i in range(num_layers):
        dense_layer = Dense(num_dense_nodes, activation=activation)(dense_layer)

    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    
    return model_compile(model, optimizer)

def build_gru_denses_stacked(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, activation):
    '''
    Build a GRU model with a single GRU layer and stacked dense layers

    Parameters
    ----------
    optimizer : str
        Optimizer to use for training
    learning_rate : float
        Learning rate for the optimizer
    num_layers : int
        Number of dense layers to stack
    num_rnn_nodes : int
        Number of nodes in the GRU layer
    num_dense_nodes : int
        Number of nodes in the dense layers
    activation : str
        Activation function to use in the dense layers
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model
    '''
    input_layer = keras.Input(shape=(173, 44))
    masking_layer = Masking(mask_value=999)(input_layer)

    gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=False)(masking_layer)

    dense_layer = Dense(num_dense_nodes, activation=activation)(gru_layer)

    for i in range(num_layers):
        dense_layer = Dense(num_dense_nodes, activation=activation)(dense_layer)

    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    
    return model_compile(model, optimizer)

## Stacked dense layers with dropout
def build_lstm_denses_stacked_dropout(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation):
    '''
    Build a LSTM model with a single LSTM layer and stacked dense layers with dropout

    Parameters
    ----------
    optimizer : str
        Optimizer to use for training
    learning_rate : float
        Learning rate for the optimizer
    num_layers : int
        Number of dense layers to stack
    num_rnn_nodes : int
        Number of nodes in the LSTM layer
    num_dense_nodes : int
        Number of nodes in the dense layers
    dropout_prob : float
        Dropout probability
    activation : str
        Activation function to use in the dense layers
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model
    '''
    input_layer = keras.Input(shape=(173, 44))
    masking_layer = Masking(mask_value=999)(input_layer)

    lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=False)(masking_layer)

    dense_layer = Dense(num_dense_nodes, activation=activation)(lstm_layer)

    for i in range(num_layers):
        dense_layer = Dense(num_dense_nodes, activation=activation)(dense_layer)
        dense_layer = Dropout(dropout_prob)(dense_layer)

    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)

    return model_compile(model, optimizer)

def build_gru_denses_stacked_dropout(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation):
    '''
    Build a GRU model with a single GRU layer and stacked dense layers with dropout

    Parameters
    ----------
    optimizer : str
        Optimizer to use for training
    learning_rate : float
        Learning rate for the optimizer
    num_layers : int
        Number of dense layers to stack
    num_rnn_nodes : int
        Number of nodes in the GRU layer
    num_dense_nodes : int
        Number of nodes in the dense layers
    dropout_prob : float
        Dropout probability
    activation : str
        Activation function to use in the dense layers
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model
    '''
    input_layer = keras.Input(shape=(173, 44))
    masking_layer = Masking(mask_value=999)(input_layer)

    gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=False)(masking_layer)

    dense_layer = Dense(num_dense_nodes, activation=activation)(gru_layer)

    for i in range(num_layers):
        dense_layer = Dense(num_dense_nodes, activation=activation)(dense_layer)
        dense_layer = Dropout(dropout_prob)(dense_layer)

    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)

    return model_compile(model, optimizer)


## Stacked LSTMs
def build_lstm_stacked(optimizer, learning_rate, num_rnn_layers, num_rnn_nodes):
    '''
    Build a LSTM model with stacked LSTM layers

    Parameters
    ----------
    optimizer : str
        Optimizer to use for training
    learning_rate : float
        Learning rate for the optimizer
    num_rnn_layers : int
        Number of LSTM layers to stack
    num_rnn_nodes : int
        Number of nodes in the LSTM layers
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model
    '''
    input_layer = keras.Input(shape=(173, 44))
    masking_layer = Masking(mask_value=999)(input_layer)

    if num_rnn_layers > 0:
        lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=True)(masking_layer)
    else:
        lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=False)(masking_layer)

    for i in range(num_rnn_layers):
        if i<num_rnn_layers-1:
            lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=True)(lstm_layer)
        else:
            lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=False)(lstm_layer)

    output_layer = Dense(1, activation='sigmoid')(lstm_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)

    return model_compile(model, optimizer)

def build_gru_stacked(optimizer, learning_rate, num_rnn_layers, num_rnn_nodes):
    '''
    Build a GRU model with stacked GRU layers

    Parameters
    ----------
    optimizer : str
        Optimizer to use for training
    learning_rate : float
        Learning rate for the optimizer
    num_rnn_layers : int
        Number of GRU layers to stack
    num_rnn_nodes : int
        Number of nodes in the GRU layers
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model
    '''
    input_layer = keras.Input(shape=(173, 44))
    masking_layer = Masking(mask_value=999)(input_layer)

    if num_rnn_layers > 0:
        gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=True)(masking_layer)
    else:
        gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=False)(masking_layer)

    for i in range(num_rnn_layers):
        if i<num_rnn_layers-1:
            gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=True)(gru_layer)
        else:
            gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=False)(gru_layer)

    output_layer = Dense(1, activation='sigmoid')(gru_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)

    return model_compile(model, optimizer)

## Stacked LSTMs + Stacked Denses with dropout
def build_lstm_stacked_denses_stacked_dropout(optimizer, learning_rate, num_rnn_layers, num_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation):
    '''
    Build a LSTM model with stacked LSTM layers and stacked dense layers with dropout

    Parameters
    ----------
    optimizer : str
        Optimizer to use for training
    learning_rate : float
        Learning rate for the optimizer
    num_rnn_layers : int
        Number of LSTM layers to stack
    num_layers : int
        Number of dense layers to stack
    num_rnn_nodes : int
        Number of nodes in the LSTM layers
    num_dense_nodes : int
        Number of nodes in the dense layers
    dropout_prob : float
        Dropout probability
    activation : str
        Activation function to use in the dense layers
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model
    '''
    input_layer = keras.Input(shape=(173, 44))
    masking_layer = Masking(mask_value=999)(input_layer)

    if num_rnn_layers > 0:
        lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=True)(masking_layer)
    else:
        lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=False)(masking_layer)

    for i in range(num_rnn_layers):
        if i<num_rnn_layers-1:
            lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=True)(lstm_layer)
        else:
            lstm_layer = LSTM(num_rnn_nodes, activation='tanh', return_sequences=False)(lstm_layer)

    dense_layer = Dense(num_dense_nodes, activation=activation)(lstm_layer)

    for i in range(num_layers):
        dense_layer = Dense(num_dense_nodes, activation=activation)(dense_layer)
        dense_layer = Dropout(dropout_prob)(dense_layer)

    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)

    return model_compile(model, optimizer)

def build_gru_stacked_denses_stacked_dropout(optimizer, learning_rate, num_rnn_layers, num_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation):
    '''
    Build a GRU model with stacked GRU layers and stacked dense layers with dropout

    Parameters
    ----------
    optimizer : str
        Optimizer to use for training
    learning_rate : float
        Learning rate for the optimizer
    num_rnn_layers : int
        Number of GRU layers to stack
    num_layers : int
        Number of dense layers to stack
    num_rnn_nodes : int
        Number of nodes in the GRU layers
    num_dense_nodes : int
        Number of nodes in the dense layers
    dropout_prob : float
        Dropout probability
    activation : str
        Activation function to use in the dense layers
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model
    '''
    input_layer = keras.Input(shape=(173, 44))
    masking_layer = Masking(mask_value=999)(input_layer)

    if num_rnn_layers > 0:
        gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=True)(masking_layer)
    else:
        gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=False)(masking_layer)

    for i in range(num_rnn_layers):
        if i<num_rnn_layers-1:
            gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=True)(gru_layer)
        else:
            gru_layer = GRU(num_rnn_nodes, activation='tanh', return_sequences=False)(gru_layer)

    dense_layer = Dense(num_dense_nodes, activation=activation)(gru_layer)

    for i in range(num_layers):
        dense_layer = Dense(num_dense_nodes, activation=activation)(dense_layer)
        dense_layer = Dropout(dropout_prob)(dense_layer)

    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)

    return model_compile(model, optimizer)
