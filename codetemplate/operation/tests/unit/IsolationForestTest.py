# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:30:43 2021

@author: gebha
"""

from unittest import TestCase
import pandas as pd
import numpy as np

import os
import sys
# Import files from src as relative import
module_path = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../../../src'))
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

from src import data_processing as dp
from src import evaluate as ev


class PreprocessingTest(TestCase):
    """ Test related to the preprocessing of data

    Test:
        - convert data (timestamp, heating cooling) to sextupel
        (current and previous values)
    """
    def test_0_data_generation(self):

        pd_data = pd.DataFrame([["2019-12-13 15:00:00", 33.5, 31.5],
                                ["2019-12-13 16:00:02", 34.5, 15.0],
                                ["2019-12-13 17:00:03", 35.5, 14.9]],
                               columns=["datetime","heating_temperature", "cooling_temperature"])
        pd_data["datetime"] = pd.to_datetime(pd_data["datetime"])
        x = dp.generate_evaluation_data(pd_data, na_heating=51, na_cooling=-11, max_t=20000)
        test = np.array([[20000, 33.5, 31.5, 20000, 51, -11],
                         [3602, 34.5, 15, 20000, 33.5, 31.5],
                         [3601, 35.5, 14.9, 3602, 34.5, 15]])
        np.testing.assert_array_equal(x, test,
                                      err_msg="[test 0] generated values do not match (test failed)")

    #TODO more test cases

class EvaluationTest(TestCase):
    """ Test related to the evaluation of data (with a trained model)

    Test:
        - predict outliers for a small test set
    """
    def test_0_prediction(self):

        data = np.array([[20000, 33.5, 31.5, 20000, 51, -11],
                         [3602, 34.5, 15, 20000, 33.5, 31.5],
                         [3601, 35.5, 14.9, 3602, 34.5, 15]])

        modelpath = os.path.join(os.path.realpath(__file__), '../isolation_tree_20210804')

        x = ev.evaluate_isolation_forest(data, modelpath)
        test = np.array([-1, 1, 1])
        np.testing.assert_array_equal(x, test,
                                      err_msg="[test 0] prediction values do not match (test failed)")

    #TODO more test cases