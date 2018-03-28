from alarm_framework.alarm_generator.alarm_setting import AlarmSetting

from scipy import stats
from scipy.optimize import minimize
from math import inf
import numpy as np
import pandas as pd


alpha = 0.01

def fap_map_threshold(proc_df, var_list=None, normal_mean=None):

    pass


def change_mean_points(signal):
    u = statistic_u(signal)
    # t = (u.abs() ** 2).idxmax()
    # p_value = 2*np.exp(-6*t/((u.size**2)+(u.size**3)))

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
    return np.cumsum(signal.apply(lambda s: np.sum(np.sign(s - signal))))
