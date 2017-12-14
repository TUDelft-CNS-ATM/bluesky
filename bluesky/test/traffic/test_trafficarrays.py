"""
Copyright (c) 2017 SPARKL Limited. All Rights Reserved.
For inclusion with BlueSky upstream code:
https://github.com/ProfHoekstra/bluesky/, distributed under
GNU General Public License v3.

Author <ahfarrell@sparkl.com> Andrew Farrell
Tests TrafficArrays
"""

import pytest
import numpy as np
import bluesky.tools.trafficarrays as ta


@pytest.fixture(scope="module")
def t_a(pytestconfig):
    """
    Suite-level setup and teardown function, for those test functions
    naming `t_a` in their parameter lists.
    """

    class TestChild(ta.TrafficArrays):

        def __init__(self):
            super(TestChild, self).__init__()

            with ta.RegisterElementParameters(self):
                self.np_array_bool = np.array([], dtype=np.bool)
                self.np_array_int = np.array([], dtype=np.int)

    class TestRoot(ta.TrafficArrays):

        def __init__(self):
            super(TestRoot, self).__init__()

            # Traffic is the toplevel trafficarrays object
            ta.TrafficArrays.SetRoot(self)

            with ta.RegisterElementParameters(self):
                self.fl_list = []
                self.int_list = []
                self.bool_list = []
                self.str_list = []

                self.test_child = TestChild()

    yield TestRoot(), TestChild.__name__


def test_trafficarrays_init(t_a):
    root, tcclass = t_a

    assert len(root.children) == 1
    assert not root.ArrVars

    assert root.children[0].__class__.__name__ == tcclass
    assert len({'str_list', 'bool_list', 'fl_list', 'int_list'}.intersection(
        set(root.LstVars))) == 4
    assert len(root.children[0].ArrVars) == 2
    assert root.children[0].np_array_bool.__class__.__name__ == 'ndarray'


def test_trafficarrays_create(t_a):
    root, _tcclass = t_a
    root.create()
    root.create_children()

    assert len(root.fl_list) == 1
    assert len(root.children[0].np_array_bool) == 1

    root.fl_list[0] = 1.0
    root.int_list[0] = 1
    root.bool_list[0] = True
    root.str_list[0] = ""

    root.create(2)
    root.create_children(2)

    assert len(root.fl_list) == 3
    assert len(root.children[0].np_array_bool) == 3

    assert root.fl_list[-1] == 0.0
    assert root.int_list[-1] == 0
    assert not root.bool_list[-1]
    assert root.str_list[-1] == ""

    assert not root.children[0].np_array_bool[-1]
    assert root.children[0].np_array_int[-1] == 0


def test_trafficarrays_delete(t_a):
    root, _tcclass = t_a

    root.int_list = [0, 1, 2]
    root.delete(1)

    assert root.int_list == [0, 2]
    assert len(root.children[0].np_array_bool) == 2


def test_trafficarrays_reset(t_a):
    root, _tcclass = t_a

    root.reset()

    assert not root.fl_list
    assert not root.children[0].np_array_bool
