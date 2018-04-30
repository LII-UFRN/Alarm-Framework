from math import inf

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, newton

from alarm_framework.alarm_generator.alarm_setting import AlarmSetting

alpha = 0.01
beta = 0.01
sample_period = 1


def fap_map_threshold(proc_df, low_threshold, high_threshold, var_list=None, r_far=0.05, r_mar=0.05, r_aad=10):
    if var_list is None:
        var_list = proc_df.columns

    if (low_threshold is None) and (high_threshold is None):
        raise ValueError('Reference threshold not provided')
    else:
        if len(var_list) != len(low_threshold):
            raise ValueError('var_list length expected to be equal to low_threshold length')
        if len(var_list) != len(high_threshold):
            raise ValueError('var_list length expected to be equal to high_threshold length')

    select_proc_df = proc_df[var_list]

    alarm_settings = list()
    for i, col in enumerate(var_list):
        print('Start tunning %s' % col)
        signal = select_proc_df[col]
        t = change_mean_points(signal)
        t.sort()

        low_seq, norm_l_seq, norm_h_seq, high_seq = mean_groups(signal, t, low_threshold[i], high_threshold[i])
        low_kde, norm_l_kde, norm_h_kde, high_kde = density_estimation(low_seq, norm_l_seq, norm_h_seq, high_seq)

        if low_kde is not None:
            lower, upper = low_seq.mean(), norm_l_seq.mean()
            threshold = opt_threshold(low_kde, norm_l_kde, r_mar, r_far, r_aad, lower, upper, high=False)
            alm = AlarmSetting(threshold, "LOW", col)
            alarm_settings.append(alm)

        if high_kde is not None:
            lower, upper = norm_h_seq.mean(), high_seq.mean()
            threshold = opt_threshold(norm_h_kde, high_kde, r_far, r_mar, r_aad, lower, upper, high=True)
            alm = AlarmSetting(threshold, "HIGH", col)
            alarm_settings.append(alm)

        print('End tunning %s' % col)

    return alarm_settings


def change_mean_points(signal):
    """
            Parameters
            ----------
            signal: pd.Series
    """
    u = statistic_u(signal)

    t = u.abs().idxmax()
    p_value = 2 * np.exp((-6 * (max(abs(u)) ** 2)) / ((len(signal) ** 2) + (len(signal) ** 3)))

    if p_value < alpha:
        signal_left = signal.loc[:t]
        signal_right = signal.loc[t:]

        ts = change_mean_points(signal_left) + change_mean_points(signal_right)
        ts.append(t)
        return ts
    else:
        return []


def statistic_u(signal):
    """
            Parameters
            ----------
            signal: pd.Series
    """
    return np.cumsum(signal.apply(lambda s: np.sum(np.sign(s - signal))))


def mean_groups(signal, t, low_threshold, high_threshold):
    low_seq = pd.Series()
    norm_l_seq = pd.Series()
    norm_h_seq = pd.Series()
    high_seq = pd.Series()

    t_zero = signal.index[0]
    for t_one in t:
        x = signal[t_zero:t_one]

        tl_statistic, pl_value = stats.ttest_1samp(x, low_threshold)
        if pl_value < beta:
            if tl_statistic < 0:
                low_seq = low_seq.append(x)
            else:
                norm_l_seq = norm_l_seq.append(x)

        th_statistic, ph_value = stats.ttest_1samp(x, high_threshold)

        if ph_value < beta:
            if th_statistic > 0:
                high_seq = high_seq.append(x)
            else:
                norm_h_seq = norm_h_seq.append(x)

        t_zero = t_one

    x = signal[t_zero:]

    tl_statistic, pl_value = stats.ttest_1samp(x, low_threshold)
    if pl_value < beta:
        if tl_statistic < 0:
            low_seq = low_seq.append(x)
        else:
            norm_l_seq = norm_l_seq.append(x)

    th_statistic, ph_value = stats.ttest_1samp(x, high_threshold)

    if ph_value < beta:
        if th_statistic > 0:
            high_seq = high_seq.append(x)
        else:
            norm_h_seq = norm_h_seq.append(x)

    return low_seq, norm_l_seq, norm_h_seq, high_seq


def density_estimation(low_seq, norm_l_seq, norm_h_seq, high_seq):
    low_kde = stats.gaussian_kde(low_seq) if not low_seq.empty else None
    norm_l_kde = stats.gaussian_kde(norm_l_seq) if not norm_l_seq.empty else None
    norm_h_kde = stats.gaussian_kde(norm_h_seq) if not norm_h_seq.empty else None
    high_kde = stats.gaussian_kde(high_seq) if not high_seq.empty else None
    return low_kde, norm_l_kde, norm_h_kde, high_kde


def opt_threshold(lower_kde, upper_kde, r_lower, r_upper, r_aad, lower, upper, w=(1, 1, 1), high=True):
    x_zero = np.random.uniform(lower, upper)
    args = (w, lower_kde, upper_kde, r_lower, r_upper, r_aad, high)
    threshold = minimize(fun=loss_function, x0=x_zero, args=args, bounds=[(lower, upper)])
    return threshold.x[0]


def loss_function(x, *args):
    w = args[0]
    lower_rate = args[1].integrate_box_1d(x, inf)
    upper_rate = args[2].integrate_box_1d(-inf, x)
    if args[6]:
        aad = sample_period * (upper_rate/(1 - upper_rate))
    else:
        aad = sample_period * (lower_rate/(1 - lower_rate))

    r_lower, r_upper, r_aad = args[3], args[4], args[5]
    j = w[0]*(lower_rate/r_lower) + w[1]*(upper_rate/r_upper) + w[2]*(aad/r_aad)
    return j
