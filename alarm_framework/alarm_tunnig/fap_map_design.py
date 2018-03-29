from alarm_framework.alarm_generator.alarm_setting import AlarmSetting

from scipy import stats
from scipy.optimize import minimize, newton
from math import inf
import numpy as np
import pandas as pd


alpha = 0.01
beta = 0.01
sample_period = 1


def var_limits(lower_kde, upper_kde, r_lower, r_upper, r_aad, mean, high=True):
    lower = newton(lambda x: lower_kde.integrate_box_1d(x, inf) - r_lower, x0=mean)
    upper = newton(lambda x: upper_kde.integrate_box_1d(-inf, x) - r_upper, x0=mean)

    if high:
        aad_limit = newton(lambda x: (sample_period * (upper_kde.integrate_box_1d(-inf, x)/ (1 - upper_kde.integrate_box_1d(-inf, x)))) - r_aad, x0=mean)
        upper = aad_limit if upper > aad_limit else upper

    else:
        aad_limit = newton(lambda x: (sample_period * (lower_kde.integrate_box_1d(x, inf)/ (1 - lower_kde.integrate_box_1d(x, inf)))) - r_aad, x0=mean)
        lower = aad_limit if lower < aad_limit else lower

    return lower, upper


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
        t.sort()

        low_seq, norm_seq, high_seq = mean_groups(signal, t, normal_mean[i])
        low_kde, norm_kde, high_kde = density_estimation(low_seq, norm_seq, high_seq)

        if low_kde is not None:
            lower, upper = var_limits(low_kde, norm_kde, r_mar, r_far, r_aad, normal_mean[i])
            threshold = opt_threshold(low_kde, norm_kde, r_mar, r_far, r_aad, lower, upper)
            alm = AlarmSetting(threshold, "LOW", col)
            alarm_settings.append(alm)

        if high_kde is not None:
            lower, upper = var_limits(norm_kde, high_kde, r_far, r_mar, r_aad, normal_mean[i])
            threshold = opt_threshold(norm_kde, high_kde, r_far, r_mar, r_aad, lower, upper)
            alm = AlarmSetting(threshold[i], "HIGH", col)
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


def mean_groups(signal, t, mean):
    low_seq = pd.Series()
    norm_seq = pd.Series()
    high_seq = pd.Series()

    t_zero = 0
    for t_one in t:
        x = signal[t_zero:t_one]

        t_statistic, p_value = stats.ttest_1samp(x, mean)
        t_beta_low = stats.t.ppf(beta, len(x))
        t_beta_high = stats.t.ppf(1 - beta, len(x))

        if t_statistic < t_beta_low:
            low_seq.append(x)
        elif t_statistic > t_beta_high:
            high_seq.append(x)
        else:
            norm_seq.append(x)
        t_zero = t_one

    x = signal[t_zero:]

    t_statistic, p_value = stats.ttest_1samp(x, mean)
    t_beta_low = stats.t.ppf(beta, len(x))
    t_beta_high = stats.t.ppf(1 - beta, len(x))

    if t_statistic < t_beta_low:
        low_seq.append(x)
    elif t_statistic > t_beta_high:
        high_seq.append(x)
    else:
        norm_seq.append(x)

    return low_seq, norm_seq, high_seq


# Return refactor
def density_estimation(low_seq, norm_seq, high_seq):
    low_kde = stats.gaussian_kde(low_seq) if not low_seq.empty else None
    norm_kde = stats.gaussian_kde(norm_seq) if not norm_seq.empty else None
    high_kde = stats.gaussian_kde(high_seq) if not high_seq.empty else None
    return low_kde, norm_kde, high_kde


def opt_threshold(lower_kde, upper_kde, r_lower, r_upper, r_aad, lower, upper, w=(1, 1, 1)):
    x_zero = np.random.uniform(lower, upper)
    args = (w, lower_kde, upper_kde, r_lower, r_upper, r_aad)
    threshold = minimize(fun=loss_function, x0=x_zero, args=args)
    return threshold


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
