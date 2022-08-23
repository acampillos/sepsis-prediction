import pandas as pd

import logging
import csv
import time

import xgboost as xgb

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Plots
from ..visualization.util import get_average_importances
from ..visualization.plots import plot_optimizer_search_evolution, plot_models_importances 

# Training
from train_xgboost import train_model, test_model


def bayesian_optimization_xgboost(full_labvitals, features, split, model_params, model_train_function, init_points=10, n_iter=20, weight=None):
    '''
    Performs Bayesian Optimization for XGBoost model hyperparameters.

    Parameters
    ----------
    full_labvitals : pandas.DataFrame
        Dataframe containing the labvitals data.
    features : list
        List of features to use for training.
    split : int
        Split to use for training.
    model_params : dict
        Dictionary containing the model parameters.
    model_train_function : function
        Function to train the model.
    
    Returns
    -------
    optimizer : bayes_opt.BayesianOptimization
        Optimizer object containing the results of the optimization.
    '''

    def train_xgboost(n_estimators, max_depth, learning_rate, gamma, min_child_weight, max_delta_step, colsample_bytree, colsample_bylevel, reg_alpha, reg_lambda, scale_pos_weight, base_score):
        xgboost_cls = xgb.XGBClassifier(n_estimators=int(n_estimators),\
            max_depth=int(max_depth),\
            learning_rate=learning_rate,\
            gamma=gamma,\
            min_child_weight=min_child_weight,\
            max_delta_step=max_delta_step,\
            colsample_bytree=colsample_bytree,\
            colsample_bylevel=colsample_bylevel,\
            reg_alpha=reg_alpha,\
            reg_lambda=reg_lambda,\
            scale_pos_weight=scale_pos_weight,\
            base_score=base_score,\
            objective='binary:logistic')
        return model_train_function(full_labvitals, features, split, xgboost_cls, weight=weight)

    optimizer = BayesianOptimization(
        f=train_xgboost,
        pbounds=model_params
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    return optimizer

def load_hyperparams_xgboost(xgboost_params_path):
    '''
    Loads the hyperparameters for the XGBoost model.

    Parameters
    ----------
    xgboost_params_path : str
        Path to the file containing the hyperparameters.
    
    Returns
    -------
    model_params : dict
        Dictionary containing the model parameters.
    '''
    reader = csv.DictReader(open(xgboost_params_path))
    xgboost_params = next(reader)
    xgboost_params['n_estimators'] = int(float(xgboost_params['n_estimators']))
    xgboost_params['max_depth'] = int(float(xgboost_params['max_depth']))
    del xgboost_params['target']
    del xgboost_params['init_points']
    del xgboost_params['n_iter']
    del xgboost_params['time']
    del xgboost_params['split']

    return xgboost_params


def dump_bayesian_optimization_results(optimizer, init_points, n_iter, time, split, results_path, search_evoluton_save_path):
    '''
    Dumps the results of the Bayesian Optimization.

    Parameters
    ----------
    optimizer : bayes_opt.BayesianOptimization
        Optimizer object containing the results of the optimization.
    init_points : int
        Number of initial points to use for the optimization.
    n_iter : int
        Number of iterations to use for the optimization.
    time : float
        Time taken for the optimization.
    split : int
        Split to use for training.
    results_path : str
        Path to the file where the results will be saved.
    search_evoluton_save_path : str
        Path to the file where the search evolution plot will be saved.
    
    Returns
    -------
    None
    '''
    output_dict = optimizer.max["params"].copy()
    output_dict['target'] = optimizer.max["target"]
    output_dict['init_points'] = init_points
    output_dict['n_iter'] = n_iter
    output_dict['time'] = time
    output_dict['split'] = split

    with open(results_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = output_dict.keys())
        writer.writeheader()
        writer.writerow(output_dict)
    
    print(f'Best result: {optimizer.max["params"]}; f(x) = {optimizer.max["target"]}; time: {time}; init_points: {init_points}; n_iter: {n_iter}')
    plot_optimizer_search_evolution(optimizer, save_path=search_evoluton_save_path)


def hyperparameters_optimization_experiments():
    '''
    Performs the hyperparameters optimization experiments.
    '''

    logging.basicConfig(level=logging.INFO)

    vitals_features = ['sysbp', 'diabp', 'meanbp', 'resprate', 'heartrate',
       'spo2_pulsoxy', 'tempc', 'cardiacoutput', 'tvset', 'tvobserved',
       'tvspontaneous', 'peakinsppressure', 'totalpeeplevel', 'o2flow',
       'fio2']
    labs_features = ['albumin', 'bands', 'bicarbonate', 'bilirubin',
        'creatinine', 'chloride', 'glucose', 'hematocrit', 'hemoglobin',
        'lactate', 'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium',
        'bun', 'wbc', 'creatinekinase', 'ck_mb', 'fibrinogen', 'ldh',
        'magnesium', 'calcium_free', 'po2_bloodgas', 'ph_bloodgas',
        'pco2_bloodgas', 'so2_bloodgas', 'troponin_t']
    all_features = vitals_features + labs_features


    xgboost_params = {
        'n_estimators': (100, 2000),
        'max_depth': (3, 100),
        'learning_rate': (0.001, 0.1),
        'gamma': (0, 50),
        'min_child_weight': (1, 50),
        'max_delta_step': (0, 50),
        'colsample_bytree': (0.5, 1.0),
        'colsample_bylevel': (0.5, 1.0),
        'reg_alpha': (0, 50),
        'reg_lambda': (0, 50),
        'scale_pos_weight': (1, 50),
        'base_score': (0.5, 0.99)
    }

    splits = [0,1,2]

    logging.info(f'Loading all data')

    full_labvitals = pd.read_csv('output/full_labvitals_horizon_0_dropped_short.csv')

    logger = logging.getLogger()
    logger.disabled = True

    init_points = 100
    n_iter = 100


    for split in splits:
        # AUPRC
        t0 = time.time()
        optimizer = bayesian_optimization_xgboost(full_labvitals, all_features, split, xgboost_params, train_model, init_points=init_points, n_iter=n_iter)
        tf = time.time()

        dump_bayesian_optimization_results(optimizer, init_points, n_iter, tf - t0, split, results_path=f'models/hyperparam_opt/xgboost/results_split_{split}.csv', search_evoluton_save_path=f'images/search_evolution_split_{split}_n_iter_{n_iter}_init_points_{init_points}.png')

        # AUPRC average
        t0 = time.time()
        optimizer = bayesian_optimization_xgboost(full_labvitals, all_features, split, xgboost_params, train_model, init_points=init_points, n_iter=n_iter, weight='average')
        tf = time.time()

        dump_bayesian_optimization_results(optimizer, init_points, n_iter, tf - t0, split, results_path=f'models/hyperparam_opt/xgboost/results_split_{split}_average.csv', search_evoluton_save_path=f'images/search_evolution_split_{split}_n_iter_{n_iter}_init_points_{init_points}_average.png')

        # Softmax weighted AUPRC
        t0 = time.time()
        optimizer = bayesian_optimization_xgboost(full_labvitals, all_features, split, xgboost_params, train_model, init_points=init_points, n_iter=n_iter, weight='softmax')
        tf = time.time()

        dump_bayesian_optimization_results(optimizer, init_points, n_iter, tf - t0, split, results_path=f'models/hyperparam_opt/xgboost/results_split_{split}_softmax.csv', search_evoluton_save_path=f'images/search_evolution_split_{split}_n_iter_{n_iter}_init_points_{init_points}_softmax.png')

        # Reversed softmax weighted AUPRC
        t0 = time.time()
        optimizer = bayesian_optimization_xgboost(full_labvitals, all_features, split, xgboost_params, train_model, init_points=init_points, n_iter=n_iter, weight='reversed_softmax')
        tf = time.time()

        dump_bayesian_optimization_results(optimizer, init_points, n_iter, tf - t0, split, results_path=f'models/hyperparam_opt/xgboost/results_split_{split}_reversed_softmax.csv', search_evoluton_save_path=f'images/search_evolution_split_{split}_n_iter_{n_iter}_init_points_{init_points}_reversed_softmax.png')

        


def test_models_experiments():
    '''
    Performs the test models experiments.
    '''
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.disabled = True

    vitals_features = ['sysbp', 'diabp', 'meanbp', 'resprate', 'heartrate',
       'spo2_pulsoxy', 'tempc', 'cardiacoutput', 'tvset', 'tvobserved',
       'tvspontaneous', 'peakinsppressure', 'totalpeeplevel', 'o2flow',
       'fio2']
    labs_features = ['albumin', 'bands', 'bicarbonate', 'bilirubin',
        'creatinine', 'chloride', 'glucose', 'hematocrit', 'hemoglobin',
        'lactate', 'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium',
        'bun', 'wbc', 'creatinekinase', 'ck_mb', 'fibrinogen', 'ldh',
        'magnesium', 'calcium_free', 'po2_bloodgas', 'ph_bloodgas',
        'pco2_bloodgas', 'so2_bloodgas', 'troponin_t']
    all_features = vitals_features + labs_features

    logging.info(f'Loading all data')
    full_labvitals = pd.read_csv('output/full_labvitals_horizon_0_dropped_short.csv')
    
    splits = [0,1,2]

    scores = []
    evals_result = []

    baseline_models = []
    bo_models = []
    avg_bo_models = []
    softmax_bo_models = []
    reversed_softmax_bo_models = []

    for split in splits:
        # AUPRC baseline XGBoost
        baseline_model = xgb.XGBClassifier()
        baseline_auprc_per_horizon, baseline_aucroc_per_horizon, baseline_evals_result = test_model(full_labvitals, all_features, split, baseline_model)
        baseline_models.append(baseline_model)
        evals_result.append(baseline_evals_result)

        # AUPRC XGBoost
        xgboost_params_path = f'models/hyperparam_opt/xgboost/results_split_{split}.csv'
        xgboost_params = load_hyperparams_xgboost(xgboost_params_path)

        bo_model = xgb.XGBClassifier(**xgboost_params)
        bo_auprc_per_horizon, bo_aucroc_per_horizon, bo_evals_result = test_model(full_labvitals, all_features, split, bo_model)
        bo_models.append(bo_model)
        evals_result.append(bo_evals_result)

        # AUPRC average
        xgboost_params_path = f'models/hyperparam_opt/xgboost/results_split_{split}_average.csv'
        xgboost_params = load_hyperparams_xgboost(xgboost_params_path)

        avg_bo_model = xgb.XGBClassifier(**xgboost_params)
        avg_bo_auprc_per_horizon, avg_bo_aucroc_per_horizon, avg_bo_evals_result = test_model(full_labvitals, all_features, split, avg_bo_model)
        avg_bo_models.append(avg_bo_model)
        evals_result.append(avg_bo_evals_result)

        # Softmax weighted AUPRC
        xgboost_params_path = f'models/hyperparam_opt/xgboost/results_split_{split}_softmax.csv'
        xgboost_params = load_hyperparams_xgboost(xgboost_params_path)

        softmax_bo_model = xgb.XGBClassifier(**xgboost_params)
        softmax_bo_auprc_per_horizon, softmax_bo_aucroc_per_horizon, softmax_bo_evals_result = test_model(full_labvitals, all_features, split, softmax_bo_model)
        softmax_bo_models.append(softmax_bo_model)
        evals_result.append(softmax_bo_evals_result)

        # Reversed softmax weighted AUPRC
        xgboost_params_path = f'models/hyperparam_opt/xgboost/results_split_{split}_reversed_softmax.csv'
        xgboost_params = load_hyperparams_xgboost(xgboost_params_path)

        reversed_softmax_bo_model = xgb.XGBClassifier(**xgboost_params)
        reversed_softmax_bo_auprc_per_horizon, reversed_softmax_bo_aucroc_per_horizon, reversed_softmax_bo_evals_result = test_model(full_labvitals, all_features, split, reversed_softmax_bo_model)
        reversed_softmax_bo_models.append(reversed_softmax_bo_model)
        evals_result.append(reversed_softmax_bo_evals_result)

        # Save scores
        for horizon, auprc in baseline_auprc_per_horizon.items():
            row = ['Baseline', horizon, split, 'auprc', auprc]
            scores.append(row)
        
        for horizon, aucroc in baseline_aucroc_per_horizon.items():
            row = ['Baseline', horizon, split, 'aucroc', aucroc]
            scores.append(row)

        for horizon, auprc in bo_auprc_per_horizon.items():
            row = ['Tuned', horizon, split, 'auprc', auprc]
            scores.append(row)
        
        for horizon, aucroc in bo_aucroc_per_horizon.items():
            row = ['Tuned', horizon, split, 'aucroc', aucroc]
            scores.append(row)

        for horizon, auprc in avg_bo_auprc_per_horizon.items():
            row = ['Average', horizon, split, 'auprc', auprc]
            scores.append(row)
        
        for horizon, aucroc in avg_bo_aucroc_per_horizon.items():
            row = ['Average', horizon, split, 'aucroc', aucroc]
            scores.append(row)
        
        for horizon, auprc in softmax_bo_auprc_per_horizon.items():
            row = ['Softmax', horizon, split, 'auprc', auprc]
            scores.append(row)
        
        for horizon, aucroc in softmax_bo_aucroc_per_horizon.items():
            row = ['Softmax', horizon, split, 'aucroc', aucroc]
            scores.append(row)
        
        for horizon, auprc in reversed_softmax_bo_auprc_per_horizon.items():
            row = ['Reversed softmax', horizon, split, 'auprc', auprc]
            scores.append(row)
        
        for horizon, aucroc in reversed_softmax_bo_aucroc_per_horizon.items():
            row = ['Reversed softmax', horizon, split, 'aucroc', aucroc]
            scores.append(row)
        
    scores_df = pd.DataFrame(scores, columns=['model_type', 'horizon', 'split', 'score_type', 'score'])
    scores_df.to_csv(f'output/xgboost/scores.csv', index=False)

    models = [baseline_models, bo_models, avg_bo_models, softmax_bo_models, reversed_softmax_bo_models]

    return scores_df, models, evals_result







if __name__ == '__main__':

    vitals_features = ['sysbp', 'diabp', 'meanbp', 'resprate', 'heartrate',
       'spo2_pulsoxy', 'tempc', 'cardiacoutput', 'tvset', 'tvobserved',
       'tvspontaneous', 'peakinsppressure', 'totalpeeplevel', 'o2flow',
       'fio2']
    labs_features = ['albumin', 'bands', 'bicarbonate', 'bilirubin',
        'creatinine', 'chloride', 'glucose', 'hematocrit', 'hemoglobin',
        'lactate', 'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium',
        'bun', 'wbc', 'creatinekinase', 'ck_mb', 'fibrinogen', 'ldh',
        'magnesium', 'calcium_free', 'po2_bloodgas', 'ph_bloodgas',
        'pco2_bloodgas', 'so2_bloodgas', 'troponin_t']
    all_features = vitals_features + labs_features

    # hyperparameters_optimization_experiments()
    # _, models, evals_result = test_models_experiments()
    
    # baseline_evals_result = evals_result[::2]
    # bo_evals_result = evals_result[1::2]

    # splits = [0,1,2]
    # for split in splits:
    #     with open(f'output/xgboost/evals_split_{split}_model_baseline.json', 'w') as f:
    #         json.dump(baseline_evals_result[split], f)
    #     with open(f'output/xgboost/evals_split_{split}_model_tuned.json', 'w') as f:
    #         json.dump(bo_evals_result[split], f)

    # plot_models_importances(models)

    # average_importances = get_average_importances(models, all_features)

    # rows = []
    # for importance_type, acum_scores in average_importances.items():
    #     for feature, score in acum_scores.items():
    #         row = [importance_type, feature, score]
    #         rows.append(row)
    # average_importances_df = pd.DataFrame(rows, columns=['importance_type', 'feature', 'score'])
    # average_importances_df.to_csv('output/xgboost/feature_importances.csv', index=False)
