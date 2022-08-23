"""
Some useful functions.

--> MGP utils by josephfutoma 2017
"""

import numpy as np

#utiliy function to fit data into memory (one outlier patient has 11k observation values, remove outliers by cut-off) 
def mask_large_samples(data, thres, obs_min, static=None):
    result_data = []
    n = len(data) #number of data views of compact format (values, times, indices, ..)
    mask = data[8] <= thres
    min_mask = data[8] >= obs_min #mask patients with less than n_mc_smps many num_obs_values
    removed_to_few_obs = np.sum(~min_mask)
    print('-> {} patients have less than {} observation values'.format(removed_to_few_obs,obs_min))
    mask = np.logical_and(mask, min_mask)
    total_removed = np.sum(~mask)
    total_remain = np.sum(mask)
    statistics = {'total_remain': total_remain, 'total_removed': total_removed, 'removed_to_few_obs': removed_to_few_obs}
    print('---> Removing {} patients'.format(total_removed))
    for i in np.arange(n):
        result_data.append(data[i][mask])
    if static is not None:
        result_static = static[mask]
        return statistics, result_data, result_static
    else:
        return statistics, result_data

#utility funciton to extract selected horizon on the fly from compact data format
def select_horizon(data, horizon):
    #initialize data compononents of result:
    values = [] # values[i][j] stores lab/vital value of patient i as jth record (all lab,vital variable types in same array!) 
    times = [] # times[i][:] stores all distinct record times of pat i (hours since icu-intime) (sorted)
    ind_lvs = [] # ind_lvs[i][j] stores index (actually id: e.g. 0-9) to determine the lab/vital type of values[i][j]. 
    ind_times = [] # ind_times[i][j] stores index for times[i][index], to get observation time of values[i][j].
    labels = [] # binary label if case/control
    num_rnn_grid_times = [] # array with numb of grid_times for each patient
    rnn_grid_times = [] # array with explicit grid points: np.arange of num_rnn_grid_times
    num_obs_times = [] #number of times observed for each encounter; how long each row of T (times) really is
    num_obs_values = [] #number of lab values observed per encounter; how long each row of Y (values) really is
    onset_hour = [] # hour, when sepsis (or control onset) occurs (practical for horizon analysis)
    
    n_samples = len(data[0])
    #loop over all samples in dataset to process:
    for i in np.arange(n_samples):    
        #for each patient, we need onset_hour!
        onset = data[9][i]
        #determine which indices of times should be used, which not -> return selected times
        time_mask = data[1][i] <= (onset - horizon +0.01) #add small buffer due to floating point rounding issues..
        #select ind_times indices that remain when masking times
        ind_times_mask = time_mask[data[3][i]]
        # apply the ind_times_mask / or timemask to all components of respective dimensionality
        pat_values = data[0][i][ind_times_mask]
        values.append(pat_values)
        pat_times = data[1][i][time_mask]
        times.append(pat_times)
        ind_lvs.append(data[2][i][ind_times_mask])
        ind_times.append(data[3][i][ind_times_mask])
        labels.append(data[4][i])
        #update the remaining components:
        end_time = max(data[1][i])
        num_rnn_grid_time = int(np.floor(end_time)+1)
        num_rnn_grid_times.append(num_rnn_grid_time)
        rnn_grid_times.append(np.arange(num_rnn_grid_time))
        num_obs_times.append(len(pat_times))
        num_obs_values.append(len(pat_values))
        onset_hour.append(onset)
    results = [np.array(item) for item in [values,times,ind_lvs,ind_times,labels,num_rnn_grid_times,rnn_grid_times,num_obs_times,num_obs_values,onset_hour]]
    return results
