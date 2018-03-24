import unittest

from time import time

import numpy as np
import pandas as pd

from ..alarm_tunnig.sig3 import *


class TestSigma(unittest.TestCase):

    def setUp(self):
        pass

    def test_generate_sgi3_one_proc_var(self):
        proc_vars = ['XMEAS01']
        proc_data = np.random.normal(0, 1, size=(100, len(proc_vars)))
        proc_index = pd.DatetimeIndex(start=time(), periods=len(proc_data), freq='36S')
        proc_df = pd.DataFrame(data=proc_data, index=proc_index, columns=proc_vars)

        t = sigma3_threshold(proc_df, proc_vars)

        self.assertEqual(len(t), 2)
        pass

    def test_generate_sgi3_two_proc_var_two_alarm_var(self):
        proc_vars = ['XMEAS01', 'XMEAS02']
        proc_data = np.random.normal(0, 1, size=(100, len(proc_vars)))
        proc_index = pd.DatetimeIndex(start=time(), periods=len(proc_data), freq='36S')
        proc_df = pd.DataFrame(data=proc_data, index=proc_index, columns=proc_vars)

        t = sigma3_threshold(proc_df, proc_vars)

        self.assertEqual(4, len(t))
        pass

    def test_generate_sgi3_two_proc_var_one_alarm_var(self):
        proc_vars = ['XMEAS01', 'XMEAS02']
        proc_data = np.random.normal(0, 1, size=(100, len(proc_vars)))
        proc_index = pd.DatetimeIndex(start=time(), periods=len(proc_data), freq='36S')
        proc_df = pd.DataFrame(data=proc_data, index=proc_index, columns=proc_vars)

        t = sigma3_threshold(proc_df, [proc_vars[0]])

        self.assertEqual(2, len(t))
        pass

    def test_generate_sgi3_two_proc_var_one_alarm_var_no_list(self):
        proc_vars = ['XMEAS01', 'XMEAS02']
        proc_data = np.random.normal(0, 1, size=(100, len(proc_vars)))
        proc_index = pd.DatetimeIndex(start=time(), periods=len(proc_data), freq='36S')
        proc_df = pd.DataFrame(data=proc_data, index=proc_index, columns=proc_vars)

        t = sigma3_threshold(proc_df, proc_vars[0])

        self.assertEqual(2, len(t))
        pass

    def test_generate_sgi3_two_proc_var_no_alarm_var(self):
        proc_vars = ['XMEAS01', 'XMEAS02']
        proc_data = np.random.normal(0, 1, size=(100, len(proc_vars)))
        proc_index = pd.DatetimeIndex(start=time(), periods=len(proc_data), freq='36S')
        proc_df = pd.DataFrame(data=proc_data, index=proc_index, columns=proc_vars)

        t = sigma3_threshold(proc_df)

        self.assertEqual(4, len(t))
        pass


if __name__ == '__main__':
    unittest.main()
