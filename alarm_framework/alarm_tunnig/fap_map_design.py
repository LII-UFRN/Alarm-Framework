from alarm_framework.alarm_generator.alarm_setting import AlarmSetting

from scipy import stats
from scipy.optimize import minimize
from math import inf
import numpy as np
import pandas as pd


alpha = 0.01


def var_limits(far_kde, mar_kde, aad_kde, r_far, r_mar, r_aad):
    # Define limits based on estimations
    return None, None


def fap_map_threshold(proc_df, normal_mean, var_list=None, r_far=0.05, r_mar=0.05, r_aad=10):
    if var_list is None:
        var_list = proc_df.columns
    if normal_mean is None:
        raise ValueError('Normal mean not provided')
    elif len(var_list) != len(normal_mean):
        raise ValueError('Var list length expected to be equal to normal mean length')
    alarm_settings = list()
    for i, col in enumerate(proc_df.columns):
        signal = proc_df[col]
        t = change_mean_points(signal)
        groups = mean_groups(signal, t)

        density_estimations = density_estimation(groups)
        far_kde = None
        mar_kde = None
        aad_kde = None

        lower, upper = var_limits(far_kde, mar_kde, aad_kde, r_far, r_mar, r_aad)
        threshold = opt_threshold(far_kde, mar_kde, aad_kde, r_far, r_mar, r_aad, lower, upper)
        alm = AlarmSetting(threshold[i], "HIGH" if normal_mean is None or normal_mean[i] < threshold else "LOW", col)
        alarm_settings.append(alm)
    return alarm_settings


def change_mean_points(signal):
    """
            Parameters
            ----------
            signal: pd.Series
    """
    u = statistic_u(signal)

    t = u.abs().idxmax()
    p_value = 2 * np.exp(-6 * max(abs(u) ** 2) / ((len(signal) ** 2) + (len(signal) ** 3)))

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


def mean_groups(signal, t):
    groups = []

    return groups


# Return refactor
def density_estimation(groups):
    density_estimations = []
    for g in groups:
        kde_g = stats.gaussian_kde(g)
        density_estimations.append(kde_g)
    return density_estimations


def opt_threshold(far_kde, mar_kde, aad_kde, r_far, r_mar, r_aad, lower, upper, w=(1, 1, 1)):
    x_zero = np.random.uniform(lower, upper)
    args = (w, far_kde, mar_kde, aad_kde, r_far, r_mar, r_aad)
    threshold = minimize(fun=loss_function, x0=x_zero, args=args)
    return threshold


def loss_function(x, *args):
    w = args[0]
    # Correct integration calculus
    far = args[1].integrate_box_1d(-inf, x)
    mar = args[2].integrate_box_1d(-inf, x)
    aad = args[3].integrate_box_1d(-inf, x)
    r_far, r_mar, r_aad = args[4], args[5], args[6]
    j = w[0]*(far/r_far) + w[1]*(mar/r_mar) + w[2]*(aad/r_aad)
    return j
