import skopt
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args

import time
import os
import inspect

import pandas as pd

# Models
from src.models.rnn import *

# Training
from src.train_rnn import train_rnn, test_rnn



def hyperparameter_optimization_experiments():
    '''
    Run hyperparameter optimization experiments for all models and save the results.
    '''

    dim_optimizer = Categorical(categories=['Adam', 'SGD'], name='optimizer')
    dim_learning_rate = Real(low=0.00001, high=0.001, prior='log-uniform', name='learning_rate')
    dim_rnn_num_layers = Integer(low=0, high=5, name='num_rnn_layers')
    dim_num_layers = Integer(low=1, high=5, name='num_layers')
    dim_rnn_num_nodes = Integer(low=44, high=128, name='num_rnn_nodes')
    dim_num_dense_nodes = Integer(low=44, high=128, name='num_dense_nodes')
    dim_dropout_prob = Real(low=0.0, high=0.99, prior='uniform', name='dropout_prob')
    dim_activation = Categorical(['tanh', 'relu'], name='activation')
    dim_batch_size = Integer(low=64, high=128, name='batch_size')
    # dim_epochs = Integer(low=10, high=200, name='epochs')

    baseline_dimensions = [dim_optimizer, dim_learning_rate, dim_rnn_num_nodes, dim_batch_size]
    stacked_denses_dimensions = [dim_optimizer, dim_learning_rate, dim_num_layers, dim_rnn_num_nodes, dim_num_dense_nodes, dim_activation, dim_batch_size]
    stacked_denses_dropout_dimensions = [dim_optimizer, dim_learning_rate, dim_num_layers, dim_rnn_num_nodes, dim_num_dense_nodes, dim_dropout_prob, dim_activation, dim_batch_size]
    stacked_rnn_dimensions = [dim_optimizer, dim_learning_rate, dim_rnn_num_layers, dim_rnn_num_nodes, dim_batch_size]
    stacked_rnn_denses_dropout_dimensions = [dim_optimizer, dim_learning_rate, dim_rnn_num_layers, dim_num_layers, dim_rnn_num_nodes, dim_num_dense_nodes, dim_dropout_prob, dim_activation, dim_batch_size]

    # Base-line
    @use_named_args(dimensions=baseline_dimensions)
    def fitness_lstm_baseline(optimizer, learning_rate, num_rnn_nodes, batch_size):
        model = build_lstm_baseline(optimizer, learning_rate, num_rnn_nodes)
        return -train_rnn(model, model_name='baseline_lstm', split=split, batch_size=batch_size)

    @use_named_args(dimensions=baseline_dimensions)
    def fitness_gru_baseline(optimizer, learning_rate, num_rnn_nodes, batch_size):
        model = build_gru_baseline(optimizer, learning_rate, num_rnn_nodes)
        return -train_rnn(model, model_name='baseline_gru', split=split, batch_size=batch_size)

    # Stacked Denses
    @use_named_args(dimensions=stacked_denses_dimensions)
    def fitness_lstm_stacked_denses(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, activation, batch_size):
        model = build_lstm_denses_stacked(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, activation)
        return -train_rnn(model, model_name='lstm_denses_stacked', split=split, batch_size=batch_size)

    @use_named_args(dimensions=stacked_denses_dimensions)
    def fitness_gru_stacked_denses(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, activation, batch_size):
        model = build_gru_denses_stacked(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, activation)
        return -train_rnn(model, model_name='gru_denses_stacked', split=split, batch_size=batch_size)
    
    # Stacked Denses and Dropout
    @use_named_args(dimensions=stacked_denses_dropout_dimensions)
    def fitness_lstm_stacked_denses_dropout(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation, batch_size):
        model = build_lstm_denses_stacked_dropout(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation)
        return -train_rnn(model, model_name='lstm_denses_stacked_dropout', split=split, batch_size=batch_size)
    
    @use_named_args(dimensions=stacked_denses_dropout_dimensions)
    def fitness_gru_stacked_denses_dropout(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation, batch_size):
        model = build_gru_denses_stacked_dropout(optimizer, learning_rate, num_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation)
        return -train_rnn(model, model_name='gru_denses_stacked_dropout', split=split, batch_size=batch_size)

    # Stacked RNNs
    @use_named_args(dimensions=stacked_rnn_dimensions)
    def fitness_lstm_stacked(optimizer, learning_rate, num_rnn_layers, num_rnn_nodes, batch_size):
        model = build_lstm_stacked(optimizer, learning_rate, num_rnn_layers, num_rnn_nodes)
        return -train_rnn(model, model_name='lstm_stacked', split=split, batch_size=batch_size)
    
    @use_named_args(dimensions=stacked_rnn_dimensions)
    def fitness_gru_stacked(optimizer, learning_rate, num_rnn_layers, num_rnn_nodes, batch_size):
        model = build_gru_stacked(optimizer, learning_rate, num_rnn_layers, num_rnn_nodes)
        return -train_rnn(model, model_name='gru_stacked', split=split, batch_size=batch_size)
    
    # Stacked RNNs + stacked Denses + Dropout
    @use_named_args(dimensions=stacked_rnn_denses_dropout_dimensions)
    def fitness_lstm_stacked_denses_stacked_dropout(optimizer, learning_rate, num_layers, num_rnn_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation, batch_size):
        model = build_lstm_stacked_denses_stacked_dropout(optimizer, learning_rate, num_rnn_layers, num_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation)
        return -train_rnn(model, model_name='lstm_stacked_denses_stacked_dropout', split=split, batch_size=batch_size)

    @use_named_args(dimensions=stacked_rnn_denses_dropout_dimensions)
    def fitness_gru_stacked_denses_stacked_dropout(optimizer, learning_rate, num_layers, num_rnn_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation, batch_size):
        model = build_gru_stacked_denses_stacked_dropout(optimizer, learning_rate, num_layers, num_rnn_layers, num_rnn_nodes, num_dense_nodes, dropout_prob, activation)
        return -train_rnn(model, model_name='stacked_denses_stacked_dropout', split=split, batch_size=batch_size)


    acq_func = 'EI'
    n_calls = 30

    # times = []
    hyperparam_opt_times_path = 'output/rnn/hyperparam_opt_times.csv'
    if not os.path.isfile(hyperparam_opt_times_path):
        times = pd.DataFrame([], columns=['model', 'split', 'time'])
        times.to_csv(hyperparam_opt_times_path, index=False)

    def append_time(model, split, time):
        times = pd.read_csv(hyperparam_opt_times_path)
        times = times.append({'model': model, 'split': split, 'time': time}, ignore_index=True)
        times.to_csv(hyperparam_opt_times_path, index=False)

    model_names = ['baseline_lstm', 'baseline_gru', 'lstm_stacked_denses', 'gru_stacked_denses', 'lstm_stacked_denses_dropout', 'gru_stacked_denses_dropout', 'lstm_stacked', 'gru_stacked', 'lstm_stacked_denses_stacked_dropout', 'gru_stacked_denses_stacked_dropout']
    models_fitness_functions = [fitness_lstm_baseline, fitness_gru_baseline, fitness_lstm_stacked_denses, fitness_gru_stacked_denses, fitness_lstm_stacked_denses_dropout, fitness_gru_stacked_denses_dropout, fitness_lstm_stacked, fitness_gru_stacked, fitness_lstm_stacked_denses_stacked_dropout, fitness_gru_stacked_denses_stacked_dropout]
    models_dimensions = [baseline_dimensions, baseline_dimensions, stacked_denses_dimensions, stacked_denses_dimensions, stacked_denses_dropout_dimensions, stacked_denses_dropout_dimensions, stacked_rnn_dimensions, stacked_rnn_dimensions, stacked_rnn_denses_dropout_dimensions, stacked_rnn_denses_dropout_dimensions]


    splits = [0,1,2]
    for split in splits:

        for name, fitness_function, dimensions in zip(model_names, models_fitness_functions, models_dimensions):
            
            search_result_path = f'models/hyperparam_opt/{name}/{name}_split_{split}_search_result.pkl'
            
            if not os.path.isfile(search_result_path):
                t0 = time.time()
                search_result = gp_minimize(func=fitness_function, dimensions=dimensions, acq_func=acq_func, n_calls=n_calls)
                tf = time.time()

                append_time(name, split, tf-t0)
                skopt.dump(search_result, f'models/hyperparam_opt/{name}/{name}_split_{split}_search_result.pkl', store_objective=False)


def test_models(models_names, build_model_functions, save_scores_to_file=True, early_stopping_monitor='val_auprc'):
    '''
    Test models on the test set and save the scores to a csv file.

    Parameters
    ----------
    models_names : list
        List of models names.
    build_model_functions : list
        List of functions that build the models.

    Returns
    -------
    None
    '''
    scores = []

    splits = [0,1,2]
    for split in splits:
        for name, build_model_function in zip(models_names, build_model_functions):

            model_best_results = skopt.load(f'models/hyperparam_opt/{name}/{name}_split_{split}_search_result.pkl')
            model = build_model_function(*model_best_results.x[:-1])

            t0 = time.time()
            model_auprc_scores, model_aucroc_scores, model_histories = test_rnn(model, model_name=name, split=split, batch_size=model_best_results.x[-1], early_stopping_monitor=early_stopping_monitor)
            tf = time.time()

            for horizon, auprc in model_auprc_scores.items():
                row = [name, horizon, split, 'auprc', auprc, tf - t0]
                scores.append(row)
            
            for horizon, aucroc in model_aucroc_scores.items():
                row = [name, horizon, split, 'aucroc', aucroc, tf - t0]
                scores.append(row)
        
    scores_df = pd.DataFrame(scores, columns=['model_type', 'horizon', 'split', 'score_type', 'score', 'time'])
    if save_scores_to_file:
        scores_df.to_csv(f'output/rnn/scores_{early_stopping_monitor}.csv', index=False)
    
    return scores_df


def bayesian_optimization_plots(models_names):
    '''
    Plot associated to Bayesian optimization process.

    Parameters
    ----------
    models_names : list
        List of models names.
    
    Returns
    -------
    None
    '''

    for model_name in models_names:
        model_best_results = skopt.load(f'models/hyperparam_opt/{model_name}_search_result.pkl')

        plot_convergence(model_best_results)
        _ = plot_objective(result=model_best_results)


def best_hyperparameters_to_df(models_names, build_model_functions):
    '''
    Get the best hyperparameters for each model and save them to a csv file.

    Parameters
    ----------
    models_names : list
        List of models names.
    build_model_functions : list
        List of functions that build the models.
    
    Returns
    -------
    None
    '''

    all_arguments = ['optimizer', 'learning_rate', 'num_layers', 'num_rnn_layers', 'num_rnn_nodes', 'num_dense_nodes', 'dropout_prob', 'activation', 'batch_size']
    best_hyperparameters = pd.DataFrame(columns=all_arguments, index=range(len(models_names)))

    i = 0
    splits = [0,1,2]
    for split in splits:
        for name, build_model_function in zip(models_names, build_model_functions):

            model_args = inspect.getfullargspec(build_model_function).args
            model_best_results = skopt.load(f'models/hyperparam_opt/{name}/{name}_split_{split}_search_result.pkl')
            
            best_hyperparameters.loc[i, ['model_name', 'split'] + model_args + ['batch_size']] = [name, split] + model_best_results.x
            i += 1
    
    best_hyperparameters.to_csv(f'output/rnn/best_hyperparameters.csv', index=False)
    return best_hyperparameters




if __name__ == '__main__':

    models_names = ['baseline_lstm', 'baseline_gru', 'lstm_stacked_denses', 'gru_stacked_denses', 'lstm_stacked_denses_dropout', 'gru_stacked_denses_dropout', 'lstm_stacked', 'gru_stacked', 'lstm_stacked_denses_stacked_dropout', 'gru_stacked_denses_stacked_dropout']
    build_model_functions = [build_lstm_baseline, build_gru_baseline, build_lstm_denses_stacked, build_gru_denses_stacked, build_lstm_denses_stacked_dropout, build_gru_denses_stacked_dropout, build_lstm_stacked, build_gru_stacked, build_lstm_stacked_denses_stacked_dropout, build_gru_stacked_denses_stacked_dropout]

    # hyperparameter_optimization_experiments()
    # test_models(models_names, build_model_functions, save_scores_to_file=True, early_stopping_monitor='val_loss')
    # best_hyperparameters_to_df(models_names, build_model_functions)
