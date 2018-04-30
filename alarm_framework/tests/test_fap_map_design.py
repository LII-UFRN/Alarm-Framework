import unittest

from ..alarm_tunnig.fap_map_design import *


class TestFAPMAPDesign(unittest.TestCase):

    def setUp(self):
        s = np.concatenate([np.random.normal(0, 1, 600), np.random.normal(5, 1, 200), np.random.normal(0, 1, 1000),
                            np.random.normal(5, 1, 150), np.random.normal(0, 1, 300)])
        self.signal = pd.Series(s)
        pass

    def atest_change_mean_point(self):
        ts = change_mean_points(self.signal)
        self.assertEqual(4, len(ts))

        self.assertAlmostEqual(np.array([600, 800, 1800, 1950]), np.array(ts), delta=10)
        pass

    def test_threshold(self):
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

        high_df = proc_hist_norm_df.mean() + 3 * proc_hist_norm_df.std()
        low_df = proc_hist_norm_df.mean() - 3 * proc_hist_norm_df.std()

        fap_map_settings = fap_map_threshold(proc_hist_dist_df, low_df, high_df,
                                             var_list=var_list)
        fap_map_df = pd.DataFrame([s.as_dict() for s in fap_map_settings],
                                  columns=['proc_var', 'alm_type', 'limit', 'on_delay', 'off_delay'])
        fap_map_df.to_csv("/home/ryuga/Documentos/Artigo CBA Sintonia de Alarmes/CBA-Threshold/thresholds/fap_map.csv")


if __name__ == '__main__':
    unittest.main()
