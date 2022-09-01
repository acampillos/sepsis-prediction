import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
from scipy import stats
import random


from .util import get_average_importances


# Static variables related

def duplicated_ids_barplot(count_dict, title='', save_path=None, show=True):
    '''
    Plots a barplot with the count of duplicated ids

    Parameters
    ----------
    count_dict : dict
        Dictionary with the count of duplicated ids
    title : str
        Title of the plot
    save_path : str
        Path to save the plot
    show : bool
        If True, shows the plot
    
    Returns
    -------
    None
    '''
    plt.bar(range(len(count_dict)), sorted(list(count_dict.values()), reverse=True), align='center')
    plt.xticks(range(len(count_dict)), list(count_dict.keys()), rotation=90)
    plt.ylabel('Count')
    plt.title(title)
    
    if save_path:
        plt.savefig('images/unique_ids_controls_sepsis.png', bbox_inches='tight')
    if show:
        plt.show()


def plot_sepsis_controls_prop_pie_chart(controls_static_variables_df, cases_static_variables_df, show=True, save_path=None):
    '''
    Plots a pie chart with the proportion of sepsis cases and controls

    Parameters
    ----------
    controls_static_variables_df : pandas.DataFrame
        Dataframe with the controls static variables
    cases_static_variables_df : pandas.DataFrame
        Dataframe with the cases static variables
    
    Returns
    -------
    None
    '''
    values = [len(controls_static_variables_df), len(cases_static_variables_df)]

    labels = ['','']
    explode = [0, 0.1]
    autopct = '%.2f%%'

    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(values, 
        labels = labels, 
        autopct= autopct,
        explode = explode, 
        shadow = True,
        rotatelabels = False)

    plt.legend(['Control', 'Sepsis'], loc='best')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def shared_countplot(column, *args):
    '''
    Countplot for given dataframes for a given column.
    Dataframes and their titles must be given consequently (not in pairs).

    Parameters
    ----------
    column : str
        Column to plot
    *args : (pandas.DataFrame, str)
        Dataframes to plot and their title

    Returns
    -------
    None
    '''
    f, axs = plt.subplots(1, 3)

    j = 0
    for i in range(0, len(args), 2):
        sns.countplot(x=column, data=args[i], ax=axs[j]).set(title=args[i+1])
        j+=1

    for i in range(1,len(axs)):
        axs[i].set_ylim( axs[0].get_ylim() ) # align axes
    
    plt.show()


def plot_pie_chart(df, title, legend=None, show=True, savePath=None):
    '''
    Plots a pie chart with the proportion for each value in the data.
    The proportion is calculated as the percentage of the total number of values.

    Parameters
    ----------
    df : pandas.Series
        Series of the values to plot
    title : str
        Title of the plot
    
    Returns
    -------
    None
    '''
    df_value_counts = df.value_counts()

    labels = []
    for column, count in df_value_counts.items():
        prop = 100*(count/sum(df_value_counts.values))
        if prop >= 1:
            labels.append(column)
        else:
            labels.append('')
            
    explode = []
    for column, count in df_value_counts.items():
        prop = 100*(count/sum(df_value_counts.values))
        if prop >= 10:
            explode.append(0.1)
        elif prop >= 1:
            explode.append(0.1)
        else:
            explode.append(0.1)
            
    autopct = lambda p: '{}%'.format(round(p)) if p>=1 else ''

    fig, ax = plt.subplots(figsize=(10,10))
    ax.pie(df_value_counts, 
           labels = labels, 
           autopct=autopct, 
           explode = explode, 
           shadow = True,
           rotatelabels = False)

    ax.set_title(title)
    if legend:
        plt.legend(legend, loc='best')
    else:
        plt.legend(loc='best')
    
    if savePath:
        plt.savefig(savePath, bbox_inches='tight')
    if show:
        plt.show()


def plot_missing_values_per_column(df, show=True, save_path=None):
    '''
    Plots the proportion of missing values for each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to plot
    
    Returns
    -------
    None
    '''
    colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(df))
    labels = ['Missing values', 'Non missing values']
    explode = [0.1, 0.1]

    fig = plt.figure(1)
    fig.set_figheight(20)
    fig.set_figwidth(20)

    Tot = len(df.columns)
    Cols = 3
    
    Rows = Tot // Cols 
    Rows += Tot % Cols
    Position = range(1,Tot + 1)
    
    for k in range(Tot):
        ax = fig.add_subplot(Rows,Cols,Position[k])
        
        column = df.columns[k]
        percent_missing = df[column].isnull().sum() * 100 / len(df)
        
        ax.pie([percent_missing, 1-percent_missing], 
           labels = labels, 
           colors = colors, 
           autopct='%.2f%%', 
           explode = explode, 
           shadow = True,
           rotatelabels = False)
        
        ax.set_title(column)
    
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='best')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def plot_missing_values_pie_chart(df, column, title, show=True, save_path=None):
    '''
    Plots a pie chart with the proportion of missing values for a given column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with the values to plot
    column : str
        Column to plot
    
    Returns
    -------
    None
    '''
    colors = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(df))
    labels = ['Missing values', 'Non missing values']
    explode = [0.1, 0.1]
    
    percent_missing = df[column].isnull().sum() * 100 / len(df)
        
    fig, ax = plt.subplots(figsize=(4,5))
    ax.pie([percent_missing, 1-percent_missing], 
           labels = labels, 
           colors = colors, 
           autopct='%.2f%%', 
           explode = explode, 
           shadow = True,
           rotatelabels = False)
    
    ax.set_title(title)
    plt.legend(loc='best')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def plot_multiple_boxplot(column, *args, orient='v', subplots_orientation='v', save_path=None, show=True):
    '''
    Plots a grid of boxplots for a given column.
    Dataframes and their titles must be given consequently (not in pairs).
    The orientation of the boxplots can be vertical or horizontal.
    The orientation of the subplots can be vertical or horizontal.
    The number of subplots is determined by the number of dataframes.

    Parameters
    ----------
    column : str
        Column to plot
    *args : (pandas.DataFrame, str)
        Dataframes to plot and their title
    
    Returns
    -------
    None
    '''
    if subplots_orientation=='v':
        f, axs = plt.subplots(len(args)//2, 1, figsize=(25,15))
    elif subplots_orientation=='h':
        f, axs = plt.subplots(1, len(args)//2, figsize=(25,15))
    
    j = 0
    for i in range(0, len(args), 2):
        ax = axs[j] if len(args)>2 else axs
        if orient=='v':
            sns.boxplot(y=args[i][column], ax=ax).set(title=args[i+1])
        elif orient=='h':
            sns.boxplot(x=column, data=args[i], ax=ax, orient=orient).set(title=args[i+1])
        j+=1

    for i in range(1, len(args)//2):
        axs[i].set_ylim( axs[0].get_ylim() ) # align axes
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def shared_displot(column, *args):
    '''
    Plots a grid of histograms for a given column.
    Dataframes and their titles must be given consequently (not in pairs).
    The number of subplots is determined by the number of dataframes.

    Parameters
    ----------
    column : str
        Column to plot
    *args : (pandas.DataFrame, str)
        Dataframes to plot and their title
    
    Returns
    -------
    None
    '''
    f, axs = plt.subplots(1, 3, figsize=(30,15))

    j = 0
    for i in range(0, len(args), 2):
        sns.histplot(args[i], x=column, ax=axs[j], kde=True).set(title=args[i+1])
        j+=1

    for i in range(1,len(axs)):
        axs[i].set_ylim( axs[0].get_ylim() )
    
    plt.show()


def remove_outliers(df, column):
    '''
    Removes outliers from a given column.
    The column must be numeric.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to clean
    column : str
        Column to clean
    
    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe
    '''
    return df[(np.abs(stats.zscore(df[column])) < 3)]


def correlation_matrix_heatmap(figsize, corr_matrix, vmin=-1, vmax=1, show=True, save_path=None):
    '''
    Plots a heatmap for a given correlation matrix.
    
    Parameters
    ----------
    figsize : tuple
        Figure size
    corr_matrix : pandas.DataFrame
        Correlation matrix
    
    Returns
    -------
    None
    '''
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, ax=ax, vmin=vmin, vmax=vmax)
    plt.yticks(rotation=0) 

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()




# Time series related

def plot_univariate_features(df, features, title):
    '''
    Lineplot for each feature independently.
    Nans are marked with an 'x'.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to plot
    features : list
        Features to plot
    title : str
        Title of the plot
    
    Returns
    -------
    None
    '''
    for col in features:
        print(col)
        fig = plt.figure(figsize=(9, 3))
        plt.plot(df['chart_time'], df[col].fillna(np.nan), 'x-', label='')
        plt.title(title)
        plt.show()
        fig.clf()


def kdeplot_univariate_features(df, features, title):
    '''
    Kde plot for each feature independently.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to plot
    features : list
        Features to plot
    title : str
        Title of the plot
    
    Returns
    -------
    None
    '''
    for col in features:
        print(col)
        fig = plt.figure(figsize=(9, 3))
        sns.kdeplot(df[col], shade = True)
        plt.title(title)
        plt.show()
        fig.clf()


def kdeplot_univariate_features_shared(df1, df2, df3, features, title):
    '''
    Kde plot with shared x axis.
    The features are plotted in the same figure.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Dataframe to plot
    df2 : pandas.DataFrame
        Dataframe to plot
    df3 : pandas.DataFrame
        Dataframe to plot
    features : list
        Features to plot
    title : str
        Title of the plot
    
    Returns
    -------
    None
    '''
    for feature in features:
        print(feature)
        fig = plt.figure(figsize=(9, 3))
        sns.kdeplot(df1[feature], shade = False)
        sns.kdeplot(df2[feature], shade = False)
        sns.kdeplot(df3[feature], shade = False)
        plt.title(title)
        plt.show()
        fig.clf()


def boxplot_univariate_features(df, features, title):
    '''
    Boxplot for each feature independently.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to plot
    features : list
        Features to plot
    title : str
        Title of the plot
    
    Returns
    -------
    None
    '''
    for col in features:
        print(col)
        fig = plt.figure(figsize=(9, 3))
        sns.boxplot(x=df[col])
        plt.title(title)
        plt.show()
        fig.clf()


def plot_variables_evolution(df, icustay_id, variables, title, show=True, save_path=None):
    '''
    Plots the evolution of the given variables for a given icustay_id.
    The variables must be numeric.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to plot
    icustay_id : int
        icustay_id to plot
    variables : list
        Variables to plot
    title : str
        Title of the plot
    
    Returns
    -------
    None
    '''
    df = df[df['icustay_id']==icustay_id]
    fig, axes = plt.subplots(len(variables), 1, figsize=(10,28))

    for i in range(len(variables)):
        axes[i].plot(df['chart_time'], df[variables[i]].fillna(np.nan), 'x-', label='')
        axes[i].set(xlabel='time', ylabel=variables[i])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.965)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def plot_features_kde_grid(df, features, show=True, save_path=None):
    '''
    Plots the kde for each feature in a grid.
    The features must be numeric.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to plot
    features : list
        Features to plot
    
    Returns
    -------
    None
    '''
    fig, axes = plt.subplots(len(features)//4, 4, figsize=(40,75))

    for i in range(4):
        for j in range(len(features)//4):
            if i==3:
                k=j+33
            elif i==2:
                k=j+22
            elif i==1:
                k=j+11
            else:
                k=j

            sns.kdeplot(df[features[k]], shade = True, ax=axes[j][i])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

def boxplot_features_by_hour(df, features, title):
    '''
    Boxplot for each feature independently.
    The features must be numeric.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to plot
    features : list
        Features to plot
    title : str
        Title of the plot
    
    Returns
    -------
    None
    '''
    for col in features:
        print(col)
        fig = plt.figure(figsize=(14, 3))
        sns.boxplot(x='chart_time', y=col, data=df)
        plt.title(title)
        plt.show()
        fig.clf()


def plot_features_boxplot_grid(df, features, show=True, save_path=None):
    '''
    Plots a boxplot for each feature in a grid.
    The features must be numeric.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to plot
    features : list
        Features to plot
    
    Returns
    -------
    None
    '''
    fig, axes = plt.subplots(len(features)//4, 4, figsize=(40,75))

    for i in range(4):
        for j in range(len(features)//4):
            if i==3:
                k=j+33
            elif i==2:
                k=j+22
            elif i==1:
                k=j+11
            else:
                k=j

            sns.boxplot(x='chart_time', y=features[k], data=df, ax=axes[j][i])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()




# XGBoost related

def plot_optimizer_search_evolution(optimizer, save_path=None, show=False):
    '''
    Plots the target score evolution during the Bayesian optimization process of skopt.

    Parameters
    ----------
    optimizer : skopt.Optimizer
        Optimizer to plot
    
    Returns
    -------
    None
    '''
    plt.figure(figsize = (15, 5))
    plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")
    plt.grid(True)
    plt.xlabel("Iteration", fontsize = 14)
    plt.ylabel("AUPRC", fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_feature_importances(scores, title='', save_path=None, show=False):
    '''
    Plots the feature importances of a model. The features are sorted by importance.

    Parameters
    ----------
    scores : dict
        Dictionary of feature importances
    
    Returns
    -------
    None
    '''
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

    n_features = len(sorted_scores)
    fig = plt.figure(figsize=(10, 10))
    plt.barh(range(n_features), list(sorted_scores.values()), align="center")
    plt.yticks(np.arange(n_features), list(sorted_scores.keys()))
    plt.title(title)
    plt.xlabel("importance")
    plt.ylabel("features")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

def plot_models_importances(models):
    '''
    Plots the feature importances of a list of models. The features are sorted by importance.
    Feature importances are averaged over all splits.
    The features are sorted by importance.

    Parameters
    ----------
    models : list
        List of XGBoost models to plot
    
    Returns
    -------
    None
    '''
    average_acum_scores_importance_type = get_average_importances(models)
    
    for importance_type, acum_scores in average_acum_scores_importance_type.items():
        plot_feature_importances(acum_scores, title=f"Average '{importance_type}' importance for all splits", save_path=f'images/avg_{importance_type}_importances_xgboost.png', show=True)