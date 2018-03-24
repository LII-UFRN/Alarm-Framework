import numpy as np
import pandas as pd


def alarm_generate(df, alm_set):
    """alarm_generate is a function when return a vector """

    t_delay = 0
    t_dead = 0
    log_alarm = np.zeros(len(df))
    if alm_set.alm_type == 'high':
        for i in np.arange(len(df)):
            if df.iloc[i] > alm_set.limit:
                t_dead = 0
                t_delay = t_delay + 3/len(df)
                if t_delay >= alm_set.on_delay:
                    log_alarm[i] = 1
                else:
                    log_alarm[i] = 0
            else:
                if log_alarm[i-1] == 1:
                    t_dead = t_dead + 3/len(df)
                    if t_dead >= alm_set.off_delay:
                        log_alarm[i] = 0
                        t_delay = 0
                        t_dead = 0
                    else:
                        log_alarm[i] = 1
                else:
                    t_delay = 0
                    t_dead = 0
                    log_alarm[i] = 0
    else:
        for i in np.arange(len(df)):
            if df.iloc[i] < alm_set.limit:
                t_dead = 0
                t_delay = t_delay + 3/len(df)
                if t_delay >= alm_set.on_delay:
                    log_alarm[i] = 1
                else:
                    log_alarm[i] = 0
            else:
                if log_alarm[i-1] == 1:
                    t_dead = t_dead + 3/len(df)
                    if t_dead >= alm_set.off_delay:
                        log_alarm[i] = 0
                        t_delay = 0
                        t_dead = 0
                    else:
                        log_alarm[i] = 1
                else:
                    t_delay = 0
                    t_dead = 0
                    log_alarm[i] = 0

    return pd.Series(log_alarm, index=df.index)


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
