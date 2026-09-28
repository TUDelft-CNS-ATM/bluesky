''' Tests for bluesky.tools.shapes area classes. '''
import numpy as np
import pytest

from bluesky.tools.shapes import Poly

# A simple square: lat 51-52, lon 6-7, alt 0-40000 ft
COORDS = [52.0, 6.0, 52.0, 7.0, 51.0, 7.0, 51.0, 6.0]


@pytest.fixture
def poly():
    return Poly('TESTAREA', COORDS, top=40000, bottom=0)


def test_poly_checkinside_batch(poly):
    ''' Multiple aircraft at once (array input). '''
    lat = np.array([51.5, 50.0])
    lon = np.array([6.5, 6.5])
    alt = np.array([10000.0, 10000.0])
    inside = poly.checkInside(lat, lon, alt)
    assert list(inside) == [True, False]


def test_poly_checkinside_single_scalar(poly):
    ''' Single aircraft selected by index (scalar input), issue #506.

    np scalars as obtained from e.g. bs.traf.lat[idx] must not raise
    ValueError, and must give the same answer as the batch query.
    '''
    lat = np.float64(51.5)
    lon = np.float64(6.5)
    alt = np.float64(10000.0)
    inside = poly.checkInside(lat, lon, alt)
    assert np.all(inside) == True  # noqa: E712


def test_poly_checkinside_single_altitude_limits(poly):
    ''' Scalar input outside the altitude band must be False, not crash. '''
    assert np.all(poly.checkInside(np.float64(51.5), np.float64(6.5),
                                   np.float64(50000.0))) == False  # noqa: E712
    assert np.all(poly.checkInside(np.float64(51.5), np.float64(6.5),
                                   np.float64(-1000.0))) == False  # noqa: E712


def test_poly_checkinside_single_matches_batch(poly):
    ''' Scalar query result must equal the corresponding batch query result. '''
    lats = np.array([51.5, 50.0])
    lons = np.array([6.5, 6.5])
    alts = np.array([10000.0, 10000.0])
    batch = poly.checkInside(lats, lons, alts)
    for i in range(2):
        single = poly.checkInside(lats[i], lons[i], alts[i])
        assert bool(np.all(single)) == bool(batch[i])
