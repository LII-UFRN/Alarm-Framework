import unittest
from time import time

from alarm_framework.alarm_tunnig.corr import *


class TestCorr(unittest.TestCase):
    def setUp(self):
        pass

    def test_generate_corr_two_proc_var_two_alarm_var(self):
        proc_vars = ['XMEAS01', 'XMEAS02']
        proc_data = np.random.normal(0, 1, size=(100, len(proc_vars)))
        proc_index = pd.DatetimeIndex(start=time(), periods=len(proc_data), freq='36S')
        proc_df = pd.DataFrame(data=proc_data, index=proc_index, columns=proc_vars)

        t = correlation_threshold(proc_df, proc_vars, proc_df.mean())

        self.assertEqual(2, len(t))
        pass


if __name__ == "__main__":
    unittest.main()
