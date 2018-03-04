"""
Copyright (c) 2017 SPARKL Limited. All Rights Reserved.
For inclusion with BlueSky upstream code:
https://github.com/ProfHoekstra/bluesky/, distributed under
GNU General Public License v3.

Author <ahfarrell@sparkl.com> Andrew Farrell
Common test functionality.

NOTE - The test suite is written in Python3 only.
It tests BlueSky running in either Python2 or 3.
"""


def assert_fl(result, reference, threshold=0.49):
    """
    Tests whether the absolute (i.e. distance from zero) difference
    between a result and a reference value is less than a provided threshold.

    If not provided, the default threshold of 0.49 is used.
    """
    assert abs(result - reference) < threshold
