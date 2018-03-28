import unittest

from ..alarm_tunnig.fap_map_design import *


class TestFAPMAPDesign(unittest.TestCase):

    def setUp(self):
        s = np.concatenate([np.random.normal(0, 1, 600), np.random.normal(5, 1, 200), np.random.normal(0, 1, 1000), np.random.normal(5, 1, 150), np.random.normal(0, 1, 300)])
        self.signal = pd.Series(s)
        pass

    def test_change_mean_point(self):
        ts = change_mean_points(self.signal)
        self.assertEqual(4, len(ts))

        self.assertAlmostEqual(np.array([600, 800, 1800, 1950]), np.array(ts), delta=10)
        pass


if __name__ == '__main__':
    unittest.main()
