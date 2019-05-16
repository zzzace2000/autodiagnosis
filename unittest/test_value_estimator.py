import unittest
import copy
from policy_training_and_evaluation.run_value_estimator_regression_based import _evalaute_exp
import numpy as np


class TestValueEstimator(unittest.TestCase):
    def test__evalaute_exp(self):
        exp_loader = [dict(patient_inds=np.ones((2,)), mortality=np.ones((2,))),
                      dict(patient_inds=2 * np.ones((2,)), mortality=np.zeros((2,)))]

        def evaluate_exp_fn(exp):
            result = copy.deepcopy(exp)
            result['rand_info_gains'] = np.ones((2,))
            return result

        df = _evalaute_exp(exp_loader, evaluate_exp_fn)

        self.assertEqual(df['patient_inds'].values.tolist(), [1., 2.])
        self.assertEqual(df['mortality'].values.tolist(), [1., 0.])
        self.assertEqual(df['rand_info_gains'].values.tolist(), [2., 2.])

if __name__ == '__main__':
    unittest.main()
