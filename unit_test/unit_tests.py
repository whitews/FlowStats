"""
Unit test framework
"""

import unittest

# include all the TestCase imports here
from .test_hdp import HDPMixtureModelTestCase
from .test_dp import DPMixtureModelTestCase

if __name__ == "__main__":
    unittest.main()