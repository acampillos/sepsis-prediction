import pandas as pd
import numpy as np
import math
from scipy import stats

from itertools import product
from collections import Counter

import collections
import functools
import operator



# Static variables related

def get_duplicates_count_ids(df):
    '''
    Returns a dictionary with the count of duplicated ids

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the ids to be counted
    
    Returns
    -------
    dict
        Dictionary with the count of duplicated ids
    '''
    columns_possibilites = [['subject_id'],
              ['hadm_id'],
              ['icustay_id'],
              ['subject_id','hadm_id'],
              ['hadm_id','icustay_id'],
              ['subject_id','icustay_id'],
              ['subject_id','hadm_id','icustay_id']]
    
    duplicated_per_id = {}
    
    for columns in columns_possibilites:
        duplicates_count = df.duplicated(subset=columns).sum()
        duplicated_per_id[str(columns)] = duplicates_count

    return duplicated_per_id


def conditional_entropy(x,y):
    '''
    Calculates the conditional entropy between two variables x and y

    Parameters
    ----------
    x : array-like
        Array with the values of the first variable
    y : array-like
        Array with the values of the second variable
    
    Returns
    -------
    float
        The conditional entropy between x and y
    '''
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy


def theil_u(x,y):
    '''
    Calculates the Theil U between two variables x and y
    
    Parameters
    ----------
    x : array-like
        Array with the values of the first variable
    y : array-like
        Array with the values of the second variable
    
    Returns
    -------
    float
        The Theil U between x and y
    '''
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = stats.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def association_matrix(df, columns):
    '''
    Calculates the association matrix between the columns of a dataframe
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the columns to be used
    columns : list
        List with the columns to be used
    
    Returns
    -------
    pandas.DataFrame
        Dataframe with the association matrix
    '''
    association_matrix = pd.DataFrame(index=columns, columns=columns)
    
    for index, column in product(columns, columns):
        association_matrix.loc[index, column] = theil_u(df[index], df[column])
    association_matrix = association_matrix.astype(float)
    return association_matrix


def correlation_ratio(category, measurements):
    '''
    Calculates the correlation ratio between values of category and numerical values of measurements

    Parameters
    ----------
    category : array-like
        Array with the values of the category
    measurements : array-like
        Array with the values of the measurements

    Returns
    -------
    float
        The correlation ratio between category and measurements
    '''
    fcat, _ = pd.factorize(category)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))

    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta


def correlation_ratio_matrix(df, categorical_columns, numeric_columns):
    '''
    Calculates the correlation ratio matrix between the categorical columns and the numeric columns of a dataframe
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the columns to be used
    categorical_columns : list
        List with the categorical columns to be used
    numeric_columns : list
        List with the numeric columns to be used
    
    Returns
    -------
    pandas.DataFrame
        Dataframe with the correlation ratio matrix
    '''
    correlation_ratio_matrix = pd.DataFrame(index=categorical_columns, columns=numeric_columns)
    
    for categorical_column, numeric_column in product(categorical_columns, numeric_columns):
        modified_df = df.copy()
        if df[categorical_column].any():
            modified_df = modified_df.dropna(subset=[categorical_column])
        if df[numeric_column].any():
            modified_df = modified_df.dropna(subset=[numeric_column])
            
        correlation_ratio_matrix.loc[categorical_column, numeric_column] = correlation_ratio(modified_df[categorical_column].values, modified_df[numeric_column].values)

    correlation_ratio_matrix = correlation_ratio_matrix.astype(float)
    return correlation_ratio_matrix




# Time series related

def get_nans_percent_per_hour(df, columns):
    '''
    Calculates the percentage of nans per hour of a dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the columns to be used
    columns : list
        List with the columns to be used

    Returns
    -------
    pandas.DataFrame
        Dataframe with the percentage of nans per hour
    '''
    features_nans_per_hour = {col : [] for col in columns}

    for hour in df['chart_time'].unique():
        for feature in columns:
            nans = df[df['chart_time'] == hour][feature].isna().sum()
            features_nans_per_hour[feature].append(100*nans/(nans + df[df['chart_time'] == hour][feature].notna().sum()))
            
    return pd.DataFrame.from_dict(features_nans_per_hour)


def get_n_most_correlated_features(corr_df, n):
    '''
    Returns the n most correlated features in a dictionary

    Parameters
    ----------
    corr_df : pandas.DataFrame
        Dataframe with the correlation between the features
    n : int
        Number of features to be returned
    
    Returns
    -------
    dict
        Dictionary with the n most correlated features
    '''
    c = corr_df.abs()
    np.fill_diagonal(c.values, 0)
    
    s = c.unstack()
    so = s.sort_values(kind="quicksort", ascending=False)
    
    correlated_features = {}
    for (feature1, feature2), corr in so.iteritems():
        if not ((feature2, feature1) in correlated_features):
            correlated_features[(feature1, feature2)] = corr
    
    n_most_correlated_features = {}
    i = 0
    for k,v in sorted(correlated_features.items(), key=lambda item: item[1], reverse=True):
        if i>n or i==len(correlated_features.items()):
            break
        
        n_most_correlated_features[(k[0],k[1])] = corr_df[k[0]][k[1]]
        i+=1
    
    return n_most_correlated_features


def get_most_correlated_features(df, corr_df, min_entries_both_not_null, num_pairs):
    '''
    Returns the most correlated features in a dictionary that have at least min_entries_both_not_null 
    entries in both variables. It considers num_pairs most correlated features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the features to be used
    corr_df : pandas.DataFrame
        Dataframe with the correlation between the features
    min_entries_both_not_null : int
        Minimum number of entries in both variables to be considered
    
    Returns
    -------
    dict
        Dictionary with the most correlated features
    '''
    n_most_correlated_corr = dict()
    for (f1, f2), corr in get_n_most_correlated_features(corr_df.fillna(0), num_pairs).items():
        both_not_nans = len(df[df[f1].notna() & df[f2].notna()])
        if both_not_nans > min_entries_both_not_null:
            n_most_correlated_corr[(f1,f2)] = corr
    return n_most_correlated_corr


def get_nans_percent_per_variables_pairs(variables_pairs, df, save_path=None):
    '''
    Calculates the percentage of nans for each variable in each pair of variables and the 
    percentage of not nans for each variable pair of variables.

    Parameters
    ----------
    variables_pairs : list
        List with the pairs of variables to be used
    df : pandas.DataFrame
        Dataframe with the features to be used
    save_path : str
        Path to save the dataframe with the percentage of nans per variable pair
    
    Returns
    -------
    pandas.DataFrame
        Dataframe with the percentage of nans per variable pair
    '''
    var1_nans_percent = []
    var2_nans_percent = []
    both_not_nans_percent = []

    for f1, f2 in variables_pairs:
        total_nans_f1, total_not_nans_f1 = df[f1].isna().sum(), df[f1].notna().sum()
        total_nans_f2, total_not_nans_f2 = df[f2].isna().sum(), df[f2].notna().sum()
        total = total_nans_f1+total_not_nans_f1

        var1_nans_percent.append(total_nans_f1/total*100)
        var2_nans_percent.append(total_nans_f2/total*100)
        both_not_nans_percent.append(len(df[df[f1].notna() & df[f2].notna()])/total*100)

    nans_percent_df = pd.DataFrame.from_dict({pair:0 for pair in variables_pairs}, orient='index').reset_index()
    nans_percent_df['var1'] = nans_percent_df['index'].str[0]
    nans_percent_df['var2'] = nans_percent_df['index'].str[1]
    nans_percent_df = nans_percent_df.drop(columns=['index'])
    nans_percent_df.columns = ['corr', 'var1', 'var2']
    nans_percent_df['corr'] = nans_percent_df['corr'].round(decimals = 4)
    nans_percent_df['var1_nans_percent'] = var1_nans_percent
    nans_percent_df['var2_nans_percent'] = var2_nans_percent
    nans_percent_df['both_not_nans_percent'] = both_not_nans_percent
    nans_percent_df['var1_nans_percent'] = nans_percent_df['var1_nans_percent'].round(decimals = 4)
    nans_percent_df['var2_nans_percent'] = nans_percent_df['var2_nans_percent'].round(decimals = 4)
    nans_percent_df['both_not_nans_percent'] = nans_percent_df['both_not_nans_percent'].round(decimals = 4)
    
    if save_path:
        nans_percent_df.to_csv(save_path, index=False)

    return nans_percent_df


def get_sum_grouped_by_chart_time(df):
    '''
    Calculates the sum of the values grouped by chart time.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the features to be used
    
    Returns
    -------
    pandas.DataFrame
        Dataframe with the sum of the values grouped by chart time
    '''
    grouped_sum_chart_time = df.copy()
    grouped_sum_chart_time['chart_time'] = grouped_sum_chart_time['chart_time'].astype('int64')
    grouped_sum_chart_time = grouped_sum_chart_time.groupby('chart_time').sum()
    grouped_sum_chart_time = grouped_sum_chart_time.reset_index()
    return grouped_sum_chart_time


def get_first_sepsis_appearance_entry(df):
    '''
    Calculates the first entry in the dataframe where each patient has sepsis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the features to be used
    
    Returns
    -------
    pandas.DataFrame
        Dataframe with the first entry in the dataframe where each patient has sepsis
    '''
    first_sepsis_appearance = df[df['label']==1].groupby('icustay_id').head(1)
    first_sepsis_appearance['chart_time'] = first_sepsis_appearance['chart_time'].astype('int64')
    return first_sepsis_appearance


def normalize(data):
    '''
    Normalizes the data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with the features to be used
    
    Returns
    -------
    pandas.DataFrame
        Dataframe with the normalized data
    '''
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_controls_cases_ratio_per_hour(df):
    '''
    Calculates the ratio of controls to cases per hour.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the features to be used
    
    Returns
    -------
    dict
        Dictionary with the ratio of controls to cases per hour
    '''
    controls_cases_ratio = df.copy()
    controls_cases_ratio['chart_time'] = controls_cases_ratio['chart_time'].astype('int64')

    hours_with_entries = controls_cases_ratio['chart_time'].unique()
    hours = list(range(0, max(hours_with_entries)+1))
    controls_cases_ratios = {hour: 0 for hour in hours}

    for (hour, label), count in controls_cases_ratio.groupby('chart_time')['label'].value_counts().iteritems():
        if label==0:
            controls_cases_ratios[hour] = count
        else:
            controls_cases_ratios[hour] /= count
    controls_cases_ratios.update(zip(list(controls_cases_ratios.keys()), normalize(list(controls_cases_ratios.values()))))
    return controls_cases_ratios


def get_bars_midde_top_points(fig):
    '''
    Calculates the middle top point of a bar plot given the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to be used
    
    Returns
    -------
    list
        List with the middle top point of a bar plot
    '''
    bars_middle_top_points_x = []
    bars_middle_top_points_y = []
    for b in fig:
        w,h = b.get_width(), b.get_height()
        # lower left vertex
        x0, y0 = b.xy
        # lower right vertex
        x1, y1 = x0+w,y0
        # top left vertex
        x2, y2 = x0,y0+h
        # top right vertex
        x3, y3 = x0+w,y0+h

        x, y = ((x2+x3)/2, (y2+y3)/2)
        bars_middle_top_points_x.append(x)
        bars_middle_top_points_y.append(y)
    
    return bars_middle_top_points_x, bars_middle_top_points_y


def remove_outliers(df, column):
    '''
    Removes the outliers from the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the features to be used
    column : str
        Column to be used
    
    Returns
    -------
    pandas.DataFrame
        Dataframe without the outliers
    '''
    return df[(np.abs(stats.zscore(df[column], nan_policy='omit')) < 3)]


def replace_outliers(df, column):
    '''
    Replaces the outliers from the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the features to be used
    column : str
        Column to be used
    
    Returns
    -------
    pandas.DataFrame
        Dataframe without the outliers
    '''
    df_copy = df.copy()
    df_copy[(np.abs(stats.zscore(df_copy[column], nan_policy='omit')) >= 3)] = np.nan
    return df_copy




# XGBoost related

def get_acum_score(xgboost_cls_scores, features):
    '''
    Calculates the acumulated score of the xgboost classifier for each feature in features. 
    It is the sum of the scores of the features.

    Parameters
    ----------
    xgboost_cls_scores : dict
        Dictionary with the scores of the xgboost classifier for each feature
    features : list
        List with the features to be used
    
    Returns
    -------
    dict
        Dictionary with the acumulated score of the xgboost classifier for each feature in features
    '''
    acum_score = dict()
    for feature in ['chart_time_max'] + features:
        acum_score[feature] = 0
        for score_feature, score_value in xgboost_cls_scores.items():
            if feature in score_feature:
                acum_score[feature] += score_value
    return acum_score


def get_average_importances(models, features):
    '''
    Calculates the average importance of the xgboost classifier for each feature in features.
    It is the average of the importances of the features.

    Parameters
    ----------
    models : list
        List with the XGBoost models to be used
    features : list
        List with the features to be used
    
    Returns
    -------
    dict
        Dictionary with the average importance of the xgboost classifier for each feature in features
    '''
    acum_scores_importance_type = {importance_type:None for importance_type in ['weight', 'gain', 'cover']}

    for importance_type in ['weight', 'gain', 'cover']:
        average_acum_scores = []

        for models_splits in models:
            acum_scores = []
            for model_split in models_splits:
                xgboost_cls_scores = model_split.get_booster().get_score(importance_type=importance_type)
                acum_score = get_acum_score(xgboost_cls_scores, features)
                acum_scores.append(acum_score)

            res = dict(functools.reduce(operator.add,
                                    map(collections.Counter, acum_scores)))

            res = {k: v/3 for k, v in res.items()}
            average_acum_scores.append(res)

        acum_scores_importance_type[importance_type] = average_acum_scores

    average_acum_scores_importance_type = {importance_type:None for importance_type in ['weight', 'gain', 'cover']}
    for importance_type, acum_scores in acum_scores_importance_type.items():
        res = dict(functools.reduce(operator.add,
                                    map(collections.Counter, acum_scores)))
        res = {k: v/3 for k, v in res.items()}
        average_acum_scores_importance_type[importance_type] = res
    return average_acum_scores_importance_type