import numpy as np
import pandas as pd


def alarm_generate(series, alm_set):
    """alarm_generate is a function when return a vector """

    count_on = 0
    count_off = 0
    log_alarm = np.zeros(len(series))
    if alm_set.alm_type == 'HIGH':
        for i in np.arange(len(series)):
            if series.iloc[i] > alm_set.limit:
                count_off = 0
                count_on = count_on + 1
                if count_on >= alm_set.on_delay:
                    log_alarm[i] = 1
                else:
                    log_alarm[i] = 0
            else:
                count_on = 0
                count_off = count_off + 1
                if count_off >= alm_set.off_delay:
                    log_alarm[i] = 0
                else:
                    log_alarm[i] = 1
    else:
        for i in np.arange(len(series)):
            if series.iloc[i] < alm_set.limit:
                count_off = 0
                count_on = count_on + 1
                if count_on >= alm_set.on_delay:
                    log_alarm[i] = 1
                else:
                    log_alarm[i] = 0
            else:
                count_on = 0
                count_off = count_off + 1
                if count_off >= alm_set.off_delay:
                    log_alarm[i] = 0
                else:
                    log_alarm[i] = 1

    return pd.Series(log_alarm, index=series.index)


def alarm_seq(df, alm_settings):
    seq = pd.DataFrame(index=df.index)
    for alm_set in alm_settings:
        seq[(alm_set.proc_var + '_' + alm_set.alm_type).upper()] = alarm_generate(df[alm_set.proc_var], alm_set)
    return seq


def alarm_log(df):
    diff_df = df.diff()
    diff_df.fillna(value=0, inplace=True)
    occ_df = diff_df[diff_df != 0]
    log = list()
    index = list()
    for t, r in occ_df.iterrows():
        # t = Time
        # r = Linha
        for i, v in r.iteritems():
            # i = Coluna
            # v = Valor
            proc_var, alm_type = i.split('_')
            if ~np.isnan(v):
                index.append(t)
                log.append([proc_var, proc_var, alm_type, alm_type, 'ALM' if v > 0 else 'RTN'])

    log_df = pd.DataFrame(log, columns=['TAG', 'TAG_DESC', 'ALM', 'ALM_DESC', 'STATE'], index=index)
    return log_df
