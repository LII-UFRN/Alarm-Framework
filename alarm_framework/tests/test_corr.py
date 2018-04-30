import unittest
from time import time

from alarm_framework.alarm_tunnig.corr import *


class TestCorr(unittest.TestCase):
    def setUp(self):
        pass

    def test_generate_corr_two_proc_var_two_alarm_var(self):
        proc_vars = ['XMEAS01', 'XMEAS02']
        x1 = np.concatenate([np.random.normal(0, 1, size=100), np.random.normal(3, 1, size=100)])
        x2 = np.roll(x1, 10)
        proc_data = np.transpose([x1, x2])
        proc_index = pd.DatetimeIndex(start=time(), periods=len(proc_data), freq='36S')
        proc_df = pd.DataFrame(data=proc_data, index=proc_index, columns=proc_vars)

        t = correlation_threshold(proc_df, proc_vars, proc_df.mean())

        self.assertEqual(2, len(t))
        pass

    def atest_divide_zero(self):
        proc_hist_dist_df = pd.read_csv("/home/ryuga/Documentos/Artigo CBA Sintonia de "
                                        "Alarmes/CBA-Threshold/cba_data/dados para avaliação/proc_history_dist.csv",
                                        index_col='tout')
        proc_hist_dist_df.index = pd.to_datetime(proc_hist_dist_df.index, unit='s')
        proc_hist_dist_df.columns = map(str.upper, proc_hist_dist_df.columns)

        proc_hist_norm_df = pd.read_csv("/home/ryuga/Documentos/Artigo CBA Sintonia de "
                                        "Alarmes/CBA-Threshold/cba_data/dados para avaliação/proc_history_norm.csv",
                                        index_col='tout')
        proc_hist_norm_df.index = pd.to_datetime(proc_hist_norm_df.index, unit='s')
        proc_hist_norm_df.columns = map(str.upper, proc_hist_norm_df.columns)

        var_list = ["XMEAS%02d" % i for i in [1, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16, 18, 20, 21, 22]]

        corr_settings = correlation_threshold(proc_hist_dist_df, normal_mean=proc_hist_norm_df[var_list].mean(), var_list=var_list)
        corr_df = pd.DataFrame([s.as_dict() for s in corr_settings], columns=['proc_var', 'alm_type', 'limit', 'on_delay', 'off_delay'])


if __name__ == "__main__":
    unittest.main()
