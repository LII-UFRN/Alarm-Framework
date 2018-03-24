from alarm_framework.alarm_generator.alarm_setting import AlarmSetting

from scipy import stats
from scipy.optimize import minimize
from math import inf
import numpy as np
import pandas as pd


def correlation_threshold(proc_df, var_list=None, normal_mean=None):
    if var_list is None:
        var_list = proc_df.columns
    if len(var_list) < 2:
        raise ValueError('Variable list must be greater than 1')

    select_proc_df = proc_df[var_list]

    proc_corr = select_proc_df.corr()

    kde_df = select_proc_df.apply(func=stats.gaussian_kde, axis=0)
    joint_kde_df = pd.DataFrame(index=select_proc_df.columns, columns=select_proc_df.columns)

    for x in joint_kde_df.index:
        for y in joint_kde_df.columns:
            if x is not y:
                joint_kde_df[x][y] = stats.gaussian_kde(select_proc_df[[x, y]].T)

    opt = minimize(fun=corr_fit, x0=select_proc_df.mean(), args=(joint_kde_df, kde_df, proc_corr))
    threshold = opt.x

    alarm_settings = list()
    for i, var in enumerate(var_list):
        alarm_settings.append(AlarmSetting(threshold[i], "HIGH" if normal_mean is None or normal_mean[i] < threshold[i] else "LOW", var))

    return alarm_settings


def alarm_corr(x, joint_kde, kde_df):
    corr_mat = np.matrix(np.zeros((len(x), len(x))))
    for i, x_tp in enumerate(x):
        for j, y_tp in enumerate(x[i:]):
            if i is not j:
                x_kde = kde_df.iloc[i]
                y_kde = kde_df.iloc[j]
                p_xy = joint_kde.iloc[i, j].integrate_box([x_tp, y_tp], [inf, inf])
                p_x = x_kde.integrate_box_1d(x_tp, inf)
                p_y = y_kde.integrate_box_1d(y_tp, inf)
                corr_mat[i, j] = (p_xy - p_x * p_y) / (np.sqrt(p_x - p_x ** 2) * np.sqrt(p_y - p_y ** 2))

    return corr_mat


def corr_fit(x, *args):
    alm_corr = alarm_corr(x, args[0], args[1])
    proc_corr = args[2]
    diff_corr = alm_corr - proc_corr.as_matrix()
    fit_value = np.sum(np.abs(diff_corr))
    return fit_value
