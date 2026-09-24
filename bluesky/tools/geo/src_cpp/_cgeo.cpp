#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include "geo.hpp"
#include <vector>
#define DEG2RAD 0.017453292519943295
#define RAD2DEG 57.29577951308232
#define M2NM 0.0005399568034557236
#define NM2M 1852.0

// Owned reference to a Python object, released when it goes out of scope
struct PyRef {
    PyObject* p;
    PyRef(PyObject* p = NULL) : p(p) {}
    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;
    ~PyRef() {Py_XDECREF(p);}
    PyArrayObject* arr() const {return (PyArrayObject*)p;}
    PyObject* release() {PyObject* r = p; p = NULL; return r;}
};

// Convert obj to an array of doubles. Only boolean, integer and floating point
// input is accepted, so that e.g. a string raises a TypeError instead of being
// parsed as a number.
static PyObject* as_double_array(PyObject* obj, int flags)
{
    PyRef arr(PyArray_FROM_O(obj));
    if (arr.p == NULL)
        return NULL;
    int type = PyArray_TYPE(arr.arr());
    if (!PyTypeNum_ISBOOL(type) && !PyTypeNum_ISINTEGER(type) && !PyTypeNum_ISFLOAT(type)) {
        PyErr_Format(PyExc_TypeError, "expected a number or numeric array, got %R",
                     (PyObject*)PyArray_DESCR(arr.arr()));
        return NULL;
    }
    return PyArray_FROM_OTF(arr.p, NPY_DOUBLE, flags | NPY_ARRAY_FORCECAST);
}

// Apply func element-wise to NIN arguments, which are broadcast against each
// other as numpy does. func(in, out) calculates the NOUT outputs of one element.
// Scalar arguments give scalar results, arrays give arrays of the broadcast shape.
template<int NIN, int NOUT, typename F>
static PyObject* elementwise(const char* name, PyObject* args, F func)
{
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs != NIN) {
        PyErr_Format(PyExc_TypeError, "%s() takes exactly %d argument%s (%zd given)",
                     name, NIN, NIN == 1 ? "" : "s", nargs);
        return NULL;
    }

    double in[NIN], out[NOUT];

    // Fast path for Python (and numpy float64) scalars
    bool scalars = true;
    for (int k = 0; k < NIN && scalars; ++k) {
        PyObject* arg = PyTuple_GET_ITEM(args, k);
        scalars = PyFloat_Check(arg) || PyLong_Check(arg);
    }
    if (scalars) {
        for (int k = 0; k < NIN; ++k) {
            in[k] = PyFloat_AsDouble(PyTuple_GET_ITEM(args, k));
            if (in[k] == -1.0 && PyErr_Occurred())
                return NULL;
        }
        func(in, out);
        return (NOUT == 1) ? PyFloat_FromDouble(out[0]) : Py_BuildValue("dd", out[0], out[NOUT - 1]);
    }

    PyRef arrs[NIN];
    PyObject* ops[NIN];
    for (int k = 0; k < NIN; ++k) {
        arrs[k].p = ops[k] = as_double_array(PyTuple_GET_ITEM(args, k), NPY_ARRAY_ALIGNED);
        if (ops[k] == NULL)
            return NULL;
    }

    // Common case: every argument either has the shape of the largest one and
    // is contiguous, or has a single element. The broadcast shape is then the
    // shape of the largest argument, and the data can be read directly.
    PyArrayObject* ref = arrs[0].arr();
    for (int k = 1; k < NIN; ++k) {
        PyArrayObject* arr = arrs[k].arr();
        if (PyArray_SIZE(arr) > PyArray_SIZE(ref) ||
                (PyArray_SIZE(arr) == PyArray_SIZE(ref) && PyArray_NDIM(arr) > PyArray_NDIM(ref)))
            ref = arr;
    }
    npy_intp size = PyArray_SIZE(ref), step[NIN];
    const double* pin[NIN];
    bool direct = true;
    for (int k = 0; k < NIN && direct; ++k) {
        PyArrayObject* arr = arrs[k].arr();
        pin[k] = (const double*)PyArray_DATA(arr);
        if (PyArray_SIZE(arr) == 1 && PyArray_NDIM(arr) <= PyArray_NDIM(ref))
            step[k] = 0;
        else if (PyArray_SAMESHAPE(arr, ref) && PyArray_IS_C_CONTIGUOUS(arr))
            step[k] = 1;
        else
            direct = false;
    }

    // Otherwise use numpy's broadcasting iterator. This raises a ValueError
    // when the argument shapes can't be broadcast together.
    PyRef multi(direct ? NULL : PyArray_MultiIterFromObjects(ops, NIN, 0));
    if (!direct && multi.p == NULL)
        return NULL;
    PyArrayMultiIterObject* it = (PyArrayMultiIterObject*)multi.p;

    PyRef res[NOUT];
    double* pres[NOUT];
    for (int m = 0; m < NOUT; ++m) {
        res[m].p = direct ? PyArray_SimpleNew(PyArray_NDIM(ref), PyArray_DIMS(ref), NPY_DOUBLE) :
                            PyArray_SimpleNew(PyArray_MultiIter_NDIM(it), PyArray_MultiIter_DIMS(it), NPY_DOUBLE);
        if (res[m].p == NULL)
            return NULL;
        pres[m] = (double*)PyArray_DATA(res[m].arr());
    }

    if (direct) {
        for (npy_intp i = 0; i < size; ++i) {
            for (int k = 0; k < NIN; ++k)
                in[k] = pin[k][i * step[k]];
            func(in, out);
            for (int m = 0; m < NOUT; ++m)
                pres[m][i] = out[m];
        }
    } else {
        // The iterator runs in C order over the broadcast shape, which is the
        // memory order of the newly created output arrays.
        while (PyArray_MultiIter_NOTDONE(it)) {
            for (int k = 0; k < NIN; ++k)
                in[k] = *(double*)PyArray_MultiIter_DATA(it, k);
            func(in, out);
            for (int m = 0; m < NOUT; ++m)
                *pres[m]++ = out[m];
            PyArray_MultiIter_NEXT(it);
        }
    }

    // PyArray_Return turns 0-d results into numpy scalars
    if (NOUT == 1)
        return PyArray_Return((PyArrayObject*)res[0].release());
    PyRef r0(PyArray_Return((PyArrayObject*)res[0].release()));
    if (r0.p == NULL)
        return NULL;
    PyRef r1(PyArray_Return((PyArrayObject*)res[NOUT - 1].release()));
    if (r1.p == NULL)
        return NULL;
    return Py_BuildValue("NN", r0.release(), r1.release());
}

// Convert a lat/lon pair of arrays [deg] to a vector of points [rad]
static bool to_points(const PyRef& lat, const PyRef& lon, std::vector<qdr_d_in>& points)
{
    npy_intp size = PyArray_SIZE(lat.arr());
    if (PyArray_SIZE(lon.arr()) != size) {
        PyErr_Format(PyExc_ValueError, "lat and lon have different sizes (%zd and %zd)",
                     (Py_ssize_t)size, (Py_ssize_t)PyArray_SIZE(lon.arr()));
        return false;
    }
    const double *plat = (double*)PyArray_DATA(lat.arr()),
                 *plon = (double*)PyArray_DATA(lon.arr());
    points.resize(size);
    for (npy_intp i = 0; i < size; ++i)
        points[i].init(DEG2RAD * plat[i], DEG2RAD * plon[i]);
    return true;
}

// Optional: derive the (j, i) outputs from the (i, j) outputs, in place
typedef void (*Mirror)(double* out);

// Calculate an n1 x n2 matrix for each of the NOUT outputs of func(ll1, ll2, out),
// between every point in lat1/lon1 and every point in lat2/lon2.
template<int NOUT, typename F>
static PyObject* matrix(const char* name, PyObject* args, F func, Mirror mirror)
{
    PyObject *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
    if (!PyArg_UnpackTuple(args, name, 4, 4, &arg1, &arg2, &arg3, &arg4))
        return NULL;

    PyRef lat1(as_double_array(arg1, NPY_ARRAY_IN_ARRAY));
    if (lat1.p == NULL) return NULL;
    PyRef lon1(as_double_array(arg2, NPY_ARRAY_IN_ARRAY));
    if (lon1.p == NULL) return NULL;
    PyRef lat2(as_double_array(arg3, NPY_ARRAY_IN_ARRAY));
    if (lat2.p == NULL) return NULL;
    PyRef lon2(as_double_array(arg4, NPY_ARRAY_IN_ARRAY));
    if (lon2.p == NULL) return NULL;

    std::vector<qdr_d_in> ll1, ll2;
    if (!to_points(lat1, lon1, ll1) || !to_points(lat2, lon2, ll2))
        return NULL;

    npy_intp n1 = ll1.size(), n2 = ll2.size();

    // The same set of points passed twice (in the same memory)
    bool same = n1 == n2 &&
                PyArray_DATA(lat1.arr()) == PyArray_DATA(lat2.arr()) &&
                PyArray_DATA(lon1.arr()) == PyArray_DATA(lon2.arr());

    npy_intp shape[] = {n1, n2};
    PyRef res[NOUT];
    double* pres[NOUT];
    for (int m = 0; m < NOUT; ++m) {
        res[m].p = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
        if (res[m].p == NULL)
            return NULL;
        pres[m] = (double*)PyArray_DATA(res[m].arr());
    }

    double out[NOUT];
    for (npy_intp i = 0; i < n1; ++i) {
        // With a mirror function, only the upper triangle is calculated
        for (npy_intp j = (same && mirror) ? i : 0; j < n2; ++j) {
            if (same && i == j) {
                // Distance from a point to itself: skip the calculation
                for (int m = 0; m < NOUT; ++m)
                    pres[m][i * n2 + j] = 0.0;
                continue;
            }
            func(ll1[i], ll2[j], out);
            for (int m = 0; m < NOUT; ++m)
                pres[m][i * n2 + j] = out[m];
            if (same && mirror) {
                mirror(out);
                for (int m = 0; m < NOUT; ++m)
                    pres[m][j * n2 + i] = out[m];
            }
        }
    }

    if (NOUT == 1)
        return res[0].release();
    return Py_BuildValue("NN", res[0].release(), res[NOUT - 1].release());
}

static PyObject* cgeo_rwgs84(PyObject* self, PyObject* args)
{
    return elementwise<1, 1>("rwgs84", args, [](const double* in, double* out) {
        double lat = DEG2RAD * in[0];
        out[0] = rwgs84(sin(lat), cos(lat));
    });
}

static PyObject* cgeo_qdrdist(PyObject* self, PyObject* args)
{
    return elementwise<4, 2>("qdrdist", args, [](const double* in, double* out) {
        qdr_d_in ll1, ll2;
        ll1.init(DEG2RAD * in[0], DEG2RAD * in[1]);
        ll2.init(DEG2RAD * in[2], DEG2RAD * in[3]);
        out[0] = RAD2DEG * qdr(ll1, ll2);
        out[1] = M2NM * dist(ll1, ll2);
    });
}

static PyObject* cgeo_qdrdist_matrix(PyObject* self, PyObject* args)
{
    // The bearing from j to i can't be derived from the bearing from i to j
    return matrix<2>("qdrdist_matrix", args, [](const qdr_d_in& ll1, const qdr_d_in& ll2, double* out) {
        out[0] = RAD2DEG * qdr(ll1, ll2);
        out[1] = M2NM * dist(ll1, ll2);
    }, NULL);
}

static PyObject* cgeo_latlondist(PyObject* self, PyObject* args)
{
    return elementwise<4, 1>("latlondist", args, [](const double* in, double* out) {
        qdr_d_in ll1, ll2;
        ll1.init(DEG2RAD * in[0], DEG2RAD * in[1]);
        ll2.init(DEG2RAD * in[2], DEG2RAD * in[3]);
        out[0] = dist(ll1, ll2);
    });
}

static PyObject* cgeo_latlondist_matrix(PyObject* self, PyObject* args)
{
    return matrix<1>("latlondist_matrix", args, [](const qdr_d_in& ll1, const qdr_d_in& ll2, double* out) {
        out[0] = M2NM * dist(ll1, ll2);
    }, [](double* out) {});
}

static PyObject* cgeo_wgsg(PyObject* self, PyObject* args)
{
    return elementwise<1, 1>("wgsg", args, [](const double* in, double* out) {
        out[0] = wgsg(DEG2RAD * in[0]);
    });
}

static PyObject* cgeo_qdrpos(PyObject* self, PyObject* args)
{
    return elementwise<4, 2>("qdrpos", args, [](const double* in, double* out) {
        pos newpos = qdrpos(DEG2RAD * in[0], DEG2RAD * in[1], DEG2RAD * in[2], NM2M * in[3]);
        out[0] = RAD2DEG * newpos.lat;
        out[1] = RAD2DEG * newpos.lon;
    });
}

static PyObject* cgeo_kwikdist(PyObject* self, PyObject* args)
{
    return elementwise<4, 1>("kwikdist", args, [](const double* in, double* out) {
        out[0] = M2NM * kwikdist(kwik_in(DEG2RAD * in[0], DEG2RAD * in[1],
                                         DEG2RAD * in[2], DEG2RAD * in[3]));
    });
}

static PyObject* cgeo_kwikdist_matrix(PyObject* self, PyObject* args)
{
    return matrix<1>("kwikdist_matrix", args, [](const qdr_d_in& ll1, const qdr_d_in& ll2, double* out) {
        out[0] = M2NM * kwikdist(kwik_in(ll1.lat, ll1.lon, ll2.lat, ll2.lon));
    }, [](double* out) {});
}

static PyObject* cgeo_kwikqdrdist(PyObject* self, PyObject* args)
{
    return elementwise<4, 2>("kwikqdrdist", args, [](const double* in, double* out) {
        kwik_in kin(DEG2RAD * in[0], DEG2RAD * in[1], DEG2RAD * in[2], DEG2RAD * in[3]);
        out[0] = RAD2DEG * kwikqdr(kin);
        out[1] = M2NM * kwikdist(kin);
    });
}

static PyObject* cgeo_kwikqdrdist_matrix(PyObject* self, PyObject* args)
{
    return matrix<2>("kwikqdrdist_matrix", args, [](const qdr_d_in& ll1, const qdr_d_in& ll2, double* out) {
        kwik_in kin(ll1.lat, ll1.lon, ll2.lat, ll2.lon);
        out[0] = RAD2DEG * kwikqdr(kin);
        out[1] = M2NM * kwikdist(kin);
    }, [](double* out) {
        out[0] = fmod(out[0] + 180.0, 360.0);
    });
}

static struct PyMethodDef methods[] = {
    {"rwgs84", cgeo_rwgs84, METH_VARARGS, "Get local earth radius [m] using WGS'84 spec."},
    {"rwgs84_matrix", cgeo_rwgs84, METH_VARARGS, "Get local earth radius [m] using WGS'84 spec (for vectors)."},
    {"qdrdist", cgeo_qdrdist, METH_VARARGS, "Calculate bearing [deg] and distance [nm] between lat1+lon1 and lat2+lon2"},
    {"qdrdist_matrix", cgeo_qdrdist_matrix, METH_VARARGS, "Calculate bearing [deg] and distance [nm] matrices between vectors lat1+lon1/lat2+lon2"},
    {"latlondist", cgeo_latlondist, METH_VARARGS, "Calculate distance [m] between lat1+lon1 and lat2+lon2"},
    {"latlondist_matrix", cgeo_latlondist_matrix, METH_VARARGS, "Calculate distance matrix [nm] between vectors lat1+lon1/lat2+lon2"},
    {"wgsg", cgeo_wgsg, METH_VARARGS, "Gravity acceleration at a given latitude according to WGS'84"},
    {"qdrpos", cgeo_qdrpos, METH_VARARGS, "Calculate position from reference position, bearing and distance"},
    {"kwikdist", cgeo_kwikdist, METH_VARARGS, "Quick and dirty dist [nm]"},
    {"kwikdist_matrix", cgeo_kwikdist_matrix, METH_VARARGS, "Quick and dirty dist [nm] (for vectors)"},
    {"kwikqdrdist", cgeo_kwikqdrdist, METH_VARARGS, "Quick and dirty dist [nm] and bearing [deg]"},
    {"kwikqdrdist_matrix", cgeo_kwikqdrdist_matrix, METH_VARARGS, "Quick and dirty dist [nm] and bearing [deg] (for vectors)"},
    {NULL, NULL, 0, NULL}
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

static struct PyModuleDef cgeodef =
{
    PyModuleDef_HEAD_INIT,
    "_cgeo",      /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods
};

PyMODINIT_FUNC PyInit__cgeo(void)
{
    import_array();
    return PyModule_Create(&cgeodef);
};
