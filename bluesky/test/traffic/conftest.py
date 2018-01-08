"""
Copyright (c) 2017 SPARKL Limited. All Rights Reserved.
For inclusion with BlueSky upstream code:
https://github.com/ProfHoekstra/bluesky/, distributed under
GNU General Public License v3.

Author <ahfarrell@sparkl.com> Andrew Farrell
Common test fixtures.

NOTE - The test suite is written in Python3 only.
"""
from __future__ import print_function
import pytest
import bluesky


@pytest.fixture(scope="session")
def traffic_():
    """
    Suite-level setup and teardown function, for those test functions
    naming `traffic_` in their parameter lists.
    """
    bluesky.settings.is_sim = True
    bluesky.init()
    yield bluesky.traf


@pytest.fixture(scope="session")
def route_(traffic_):
    """
    Suite-level setup and teardown function, for those test functions
    naming `route_` in their parameter lists.
    """
    from bluesky.traffic import route
    yield route
