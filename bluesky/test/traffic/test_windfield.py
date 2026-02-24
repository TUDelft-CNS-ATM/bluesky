import pytest

from bluesky.traffic.windfield import Windfield


def test_addpointvne_static_scalar_wind_does_not_crash_getdata():
    wind = Windfield()
    wind.addpointvne([52.0], [4.0], 0.0, 0.0)

    vnorth, veast = wind.getdata(52.1, 4.1, 1000.0)

    assert vnorth == pytest.approx(0.0)
    assert veast == pytest.approx(0.0)
