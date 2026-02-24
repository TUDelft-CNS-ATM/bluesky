from types import SimpleNamespace

import numpy as np
import pytest

from bluesky.traffic.performance.openap import coeff
from bluesky.traffic.performance.openap import phase as ph
from bluesky.traffic.performance.openap.perfoap import OpenAP


def test_construct_v_limits_keeps_initial_climb_bounds():
    perf = SimpleNamespace(
        actype=np.array(["A320"]),
        lifttype=np.array([coeff.LIFT_FIXWING]),
        phase=np.array([ph.IC]),
        vminic=np.array([155.0]),
        vminer=np.array([125.0]),
        vminap=np.array([110.0]),
        vmaxic=np.array([210.0]),
        vmaxer=np.array([250.0]),
        vmaxap=np.array([160.0]),
        vmin=np.array([0.0]),
        vmax=np.array([0.0]),
    )

    vmin, vmax = OpenAP._construct_v_limits(perf, np.array([True], dtype=bool))

    assert vmin[0] == pytest.approx(155.0)
    assert vmax[0] == pytest.approx(210.0)
