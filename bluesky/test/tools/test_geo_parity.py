''' Consistency of the compiled (_cgeo) and Python (_geo) geo functions. '''
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from bluesky.tools.geo import _geo

_cgeo = pytest.importorskip('bluesky.tools.geo._cgeo')

BACKENDS = [pytest.param(_geo, id='python'), pytest.param(_cgeo, id='compiled')]

# Element-wise functions of four arguments, and their outputs
ELEMENTWISE = ['qdrdist', 'latlondist', 'kwikdist', 'kwikqdrdist', 'qdrpos']
MATRIX = ['qdrdist_matrix', 'latlondist_matrix', 'kwikdist_matrix', 'kwikqdrdist_matrix']
BEARINGS = {'qdrdist': 0, 'kwikqdrdist': 0, 'qdrdist_matrix': 0, 'kwikqdrdist_matrix': 0}


def outputs(result):
    return result if isinstance(result, tuple) else (result,)


def wrap180(angle):
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def assert_close(name, a, b, dist_rtol=1e-9, qdr_atol=1e-9):
    ''' Compare the outputs of geo function name, taking bearings modulo 360. '''
    a, b = outputs(a), outputs(b)
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert np.shape(x) == np.shape(y)
        if BEARINGS.get(name) == i:
            # The bearing between coincident points is undefined
            defined = np.asarray(b[-1]) > 0.0
            diff = np.where(defined, wrap180(np.asarray(x) - np.asarray(y)), 0.0)
            np.testing.assert_allclose(diff, 0.0, atol=qdr_atol)
        elif name == 'qdrpos':
            # Positions in degrees
            np.testing.assert_allclose(wrap180(np.asarray(x) - np.asarray(y)), 0.0, atol=qdr_atol)
        else:
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=dist_rtol)


def random_positions(n, seed):
    ''' Global positions, including both hemispheres and the antimeridian. '''
    rng = np.random.default_rng(seed)
    lat = np.degrees(np.arcsin(rng.uniform(-0.999, 0.999, n)))
    lon = rng.uniform(-180.0, 180.0, n)
    return lat, lon


@pytest.fixture(scope='module')
def global_inputs():
    lat1, lon1 = random_positions(100_000, 1)
    lat2, lon2 = random_positions(100_000, 2)
    # Add pairs on either side of the antimeridian
    lat2[:1000], lon2[:1000] = lat1[:1000] + 0.5, np.where(lon1[:1000] > 0, -179.5, 179.5)
    lon1[:1000] = np.where(lon1[:1000] > 0, 179.5, -179.5)
    return lat1, lon1, lat2, lon2


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['qdrdist', 'latlondist', 'kwikdist', 'kwikqdrdist'])
def test_parity_global(name, global_inputs):
    assert_close(name, getattr(_cgeo, name)(*global_inputs), getattr(_geo, name)(*global_inputs))


def test_parity_qdrpos(global_inputs):
    lat, lon, _, _ = global_inputs
    rng = np.random.default_rng(3)
    qdr, dist = rng.uniform(-180.0, 180.0, lat.size), rng.uniform(0.0, 3000.0, lat.size)
    assert_close('qdrpos', _cgeo.qdrpos(lat, lon, qdr, dist), _geo.qdrpos(lat, lon, qdr, dist))


@pytest.mark.parametrize('name', ['rwgs84', 'rwgs84_matrix', 'wgsg'])
def test_parity_single_argument(name, global_inputs):
    lat = global_inputs[0]
    np.testing.assert_allclose(getattr(_cgeo, name)(lat), getattr(_geo, name)(lat), rtol=1e-12)
    assert getattr(_cgeo, name)(52.0) == pytest.approx(getattr(_geo, name)(52.0), rel=1e-12)


@pytest.mark.parametrize('name', MATRIX)
def test_parity_matrix(name, global_inputs):
    lat1, lon1, lat2, lon2 = (x[:300] for x in global_inputs)
    lat2, lon2 = lat2[:200], lon2[:200]
    assert_close(name, getattr(_cgeo, name)(lat1, lon1, lat2, lon2),
                 getattr(_geo, name)(lat1, lon1, lat2, lon2))


# ---------------------------------------------------------------------------
# Call shapes
# ---------------------------------------------------------------------------
_lat = np.array([52.0, 52.1, 52.2])
_lon = np.array([4.0, 4.1, 4.2])
_big = np.linspace(-60.0, 60.0, 12)

CALLS = {
    'scalar/scalar': (52.0, 4.0, 52.5, 4.5),
    'int scalars': (52, 4, 53, 5),
    'array/array': (_lat, _lon, _lat + 0.5, _lon - 0.5),
    'array/scalar': (_lat, _lon, 52.5, 4.5),
    'scalar/array': (52.5, 4.5, _lat, _lon),
    'mixed within pair': (_lat, 4.0, _lat + 0.5, _lon),
    'length-1 array': (_lat, _lon, np.array([52.0]), np.array([4.0])),
    '0-d arrays': (np.array(52.0), np.array(4.0), np.array(52.5), np.array(4.5)),
    '0-d with scalars': (np.array(52.0), 4.0, 52.5, 4.5),
    '2-d column': (_lat[:, None], _lon[:, None], 52.5, 4.5),
    '2-d broadcast': (_lat[:, None], _lon[:, None], _big[None, :4], _big[None, 4:8]),
    'float32': (_lat.astype(np.float32), _lon.astype(np.float32), 52.5, 4.5),
    'non-contiguous': (_big[::3], _big[1::3], _big[2::3], _big[::-3]),
    'empty': (np.array([]), np.array([]), 52.0, 4.0),
}


@pytest.mark.parametrize('call', CALLS.values(), ids=CALLS.keys())
@pytest.mark.parametrize('name', ELEMENTWISE)
def test_call_shapes(name, call):
    # numpy calculates float32 input in float32, while the compiled functions
    # always use float64, so compare with the same values in float64.
    ref = [x.astype(float) if getattr(x, 'dtype', None) == np.float32 else x for x in call]
    assert_close(name, getattr(_cgeo, name)(*call), getattr(_geo, name)(*ref))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', ELEMENTWISE)
def test_mismatched_lengths(backend, name):
    with pytest.raises(ValueError):
        getattr(backend, name)(_lat, _lon, _lat[:2], _lon[:2])


@pytest.mark.parametrize('name', ELEMENTWISE + ['rwgs84', 'wgsg'])
@pytest.mark.parametrize('bad', ['abc', None, [52.0, 'x']])
def test_non_numeric_input_raises_typeerror(name, bad):
    nargs = 1 if name in ('rwgs84', 'wgsg') else 4
    with pytest.raises(TypeError):
        getattr(_cgeo, name)(bad, *[52.0] * (nargs - 1))


@pytest.mark.parametrize('name', MATRIX)
def test_matrix_mismatched_lat_lon(name):
    with pytest.raises(ValueError):
        getattr(_cgeo, name)(_lat, _lon[:2], _lat, _lon)


def run_isolated(code):
    ''' Run code in a separate interpreter, so that a crash fails the test
        instead of killing the test session. '''
    proc = subprocess.run([sys.executable, '-c', textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f'exit code {proc.returncode}\n{proc.stdout}\n{proc.stderr}'
    return proc.stdout


@pytest.mark.parametrize('name', MATRIX)
def test_matrix_two_arguments_raises_typeerror(name):
    run_isolated(f'''
        import numpy as np
        from bluesky.tools.geo import _geo, _cgeo
        for backend in (_geo, _cgeo):
            try:
                backend.{name}(np.array([52.0, 53.0]), np.array([4.0, 5.0]))
            except TypeError:
                pass
            else:
                raise AssertionError(f'{{backend.__name__}}: no TypeError')
    ''')


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------
def pairwise(backend, name, lat1, lon1, lat2, lon2):
    ''' Matrix from the element-wise function, one pair at a time. '''
    func = getattr(backend, name.replace('_matrix', ''))
    rows = [[outputs(func(lat1[i], lon1[i], lat2[j], lon2[j])) for j in range(lat2.size)]
            for i in range(lat1.size)]
    res = tuple(np.array([[p[k] for p in row] for row in rows]) for k in range(len(rows[0][0])))
    if name == 'latlondist_matrix':
        # latlondist returns meters, latlondist_matrix nautical miles
        res = (res[0] / _geo.nm,)
    return res if len(res) > 1 else res[0]


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', MATRIX)
def test_matrix_two_sets(backend, name):
    lat1, lon1 = random_positions(5, 4)
    lat2, lon2 = random_positions(3, 5)
    # Also check a second set that coincides with the first on the diagonal
    for lat2, lon2 in ((lat2, lon2), (lat1[:3] + 1.0, lon1[:3] + 1.0)):
        res = getattr(backend, name)(np.asmatrix(lat1), np.asmatrix(lon1),
                                     np.asmatrix(lat2), np.asmatrix(lon2))
        for out in outputs(res):
            assert np.shape(out) == (5, 3)
        assert_close(name, tuple(np.asarray(x) for x in outputs(res)),
                     outputs(pairwise(backend, name, lat1, lon1, lat2, lon2)), dist_rtol=1e-12)
        # No forced zeros on the diagonal of two different sets
        assert np.all(np.diagonal(np.asarray(outputs(res)[-1])) > 1.0)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', MATRIX)
def test_matrix_same_set(backend, name):
    lat, lon = random_positions(6, 6)
    mlat, mlon = np.asmatrix(lat), np.asmatrix(lon)
    res = tuple(np.asarray(x) for x in outputs(getattr(backend, name)(mlat, mlon, mlat, mlon)))
    ref = outputs(pairwise(backend, name, lat, lon, lat, lon))
    offdiag = ~np.eye(6, dtype=bool)
    assert_close(name, tuple(x[offdiag] for x in res), tuple(x[offdiag] for x in ref), dist_rtol=1e-12)
    np.testing.assert_array_equal(np.diagonal(res[-1]), 0.0)


def test_matrix_scalar_reference():
    ''' A single reference position against a set, as in traffic/metric.py. '''
    for backend in (_geo, _cgeo):
        qdr, dist = backend.qdrdist_matrix(np.float64(52.0), np.float64(4.0),
                                           np.asmatrix(_lat), np.asmatrix(_lon))
        assert np.shape(dist) == (1, 3)
        np.testing.assert_allclose(np.asarray(dist).ravel(),
                                   _geo.qdrdist(52.0, 4.0, _lat, _lon)[1], rtol=1e-12)


# ---------------------------------------------------------------------------
# Units and accuracy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('backend', BACKENDS)
def test_latlondist_returns_meters(backend):
    # One degree of latitude at 52N is about 111 km
    assert backend.latlondist(52.0, 4.0, 53.0, 4.0) == pytest.approx(111085.3, rel=1e-5)
    assert backend.latlondist_matrix(np.array([52.0]), np.array([4.0]),
                                     np.array([53.0]), np.array([4.0]))[0, 0] == \
        pytest.approx(111085.3 / 1852.0, rel=1e-5)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', ['qdrdist', 'latlondist'])
def test_equator(backend, name):
    ''' Both points on the equator, where the hemisphere check divides by zero. '''
    d = outputs(getattr(backend, name)(0.0, 0.0, 0.0, 1.0))[-1]
    d = d if name == 'latlondist' else d * _geo.nm
    assert d == pytest.approx(111319.49, rel=1e-6)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('meters', [0.01, 0.1, 1.0, 10.0])
def test_qdrdist_short_range(backend, meters):
    lat1, lon1 = 52.0, 4.0
    lat2 = lat1 + np.degrees(meters / _geo.rwgs84(lat1))
    _, dist = backend.qdrdist(lat1, lon1, lat2, lon1)
    assert dist * _geo.nm == pytest.approx(meters, rel=1e-6)


@pytest.mark.parametrize('backend', BACKENDS)
def test_antipodal(backend):
    _, dist = backend.qdrdist(np.array([10.0]), np.array([20.0]), np.array([-10.0]), np.array([-160.0]))
    assert np.isfinite(dist).all()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def test_backend_selected_after_settings_are_read(tmp_path):
    ''' Importing geo before bs.init() must not fix the backend to the default
        (https://github.com/TUDelft-CNS-ATM/bluesky/issues/624). '''
    from bluesky.pathfinder import resource
    cfg = tmp_path / 'settings.cfg'
    cfg.write_text(resource('default.cfg').read_text() + '\nprefer_compiled = True\n')
    run_isolated(f'''
        import bluesky as bs
        from bluesky.tools import geo
        from bluesky.tools.geo import qdrdist, _geo, _cgeo

        # Calls before bs.init() use the default backend
        qdrdist(52.0, 4.0, 53.0, 4.0)
        assert geo._impl is _geo

        bs.init(mode='sim', detached=True, configfile={str(cfg)!r})
        assert geo._impl is _cgeo, geo._impl
        # Also through the name imported before bs.init()
        assert qdrdist(52.0, 4.0, 53.0, 4.0) == _cgeo.qdrdist(52.0, 4.0, 53.0, 4.0)
    ''')
