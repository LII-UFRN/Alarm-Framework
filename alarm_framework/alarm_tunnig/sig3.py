from alarm_framework.alarm_generator.alarm_setting import AlarmSetting


def sigma3_threshold(proc_df, var_list=None):
    if var_list is None:
        var_list = proc_df.columns

    mu = proc_df[var_list].mean()
    sigma = proc_df[var_list].std()
    low_threshold = mu - 3 * sigma
    high_threshold = mu + 3 * sigma
    alarm_settings = list()
    if type(var_list) is str:
        var_list = list(var_list)
        alarm_settings.append(AlarmSetting(low_threshold, 'LOW', var_list))
        alarm_settings.append(AlarmSetting(high_threshold, 'HIGH', var_list))
    else:
        for i, var in enumerate(var_list):
            alarm_settings.append(AlarmSetting(low_threshold[i], 'LOW', var))
            alarm_settings.append(AlarmSetting(high_threshold[i], 'HIGH', var))

    return alarm_settings
