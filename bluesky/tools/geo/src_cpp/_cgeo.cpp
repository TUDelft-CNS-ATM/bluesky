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
    PyObject      *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
    PyArrayObject *lat1 = NULL, *lon1 = NULL, *lat2 = NULL, *lon2 = NULL;
    if (!PyArg_ParseTuple(args, "OO|OO", &arg1, &arg2, &arg3, &arg4))
        return NULL;

    lat1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lon1 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lat2 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lon2 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (lat1 == NULL || lon1 == NULL) return NULL;

    double *plat1 = (double*)PyArray_DATA(lat1),
           *plon1 = (double*)PyArray_DATA(lon1);

    double *plat2 = (lat2 == NULL ? plat1 : (double*)PyArray_DATA(lat2)),
           *plon2 = (lon2 == NULL ? plon1 : (double*)PyArray_DATA(lon2));

    // Determine sizes
    npy_intp  size  = PyArray_SIZE(lat1);

    int i = 0, j = 0;

    // Create ll2 data for efficient nested loop
    std::vector<qdr_d_in> ll2(size);
    std::vector<qdr_d_in>::iterator pll2 = ll2.begin();
    while (i < size) {
        pll2->init(DEG2RAD * *plat2, DEG2RAD * *plon2);
        ++i; ++plat2; ++plon2; ++pll2;
    }

    // Create output matrices
    npy_intp shape[] = {size, size};
    PyObject* vqdr = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
    PyObject* vdst = PyArray_SimpleNew(2, shape, NPY_DOUBLE);

    // Nested loop to calculate qdr and dist matrices
    i = 0;
    double *pqdr = (double*)PyArray_DATA((PyArrayObject*)vqdr);
    double *pdst = (double*)PyArray_DATA((PyArrayObject*)vdst);

    qdr_d_in ll1;
    while (i < size) {
        ll1.init(DEG2RAD * *plat1, DEG2RAD * *plon1);
        pll2 = ll2.begin();
        while (j < size) {
            if (i == j) {
                *pqdr = 0.0;
                *pdst = 0.0;
            } else {
                *pqdr = RAD2DEG * qdr(ll1, *pll2);
                *pdst = M2NM * dist(ll1, *pll2);
            }
            ++j; ++pll2; ++pqdr; ++pdst;
        }
        ++i; ++plat1; ++plon1;
        j = 0;
    }
    //}
    Py_DECREF(lat1);
    Py_DECREF(lon1);
    Py_XDECREF(lat2);// Py_XDECREF checks for NULL
    Py_XDECREF(lon2);

    return Py_BuildValue("NN", vqdr, vdst);
};

static PyObject* cgeo_latlondist(PyObject* self, PyObject* args)
{
    return elementwise<4, 1>("latlondist", args, [](const double* in, double* out) {
        qdr_d_in ll1, ll2;
        ll1.init(DEG2RAD * in[0], DEG2RAD * in[1]);
        ll2.init(DEG2RAD * in[2], DEG2RAD * in[3]);
        out[0] = M2NM * dist(ll1, ll2);
    });
}

static PyObject* cgeo_latlondist_matrix(PyObject* self, PyObject* args)
{
    PyObject      *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
    PyArrayObject *lat1 = NULL, *lon1 = NULL, *lat2 = NULL, *lon2 = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4))
        return NULL;

    lat1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lon1 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lat2 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lon2 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (lat1 == NULL || lon1 == NULL) return NULL;

    double *plat1 = (double*)PyArray_DATA(lat1),
           *plon1 = (double*)PyArray_DATA(lon1);

    double *plat2 = (lat2 == NULL ? plat1 : (double*)PyArray_DATA(lat2)),
           *plon2 = (lon2 == NULL ? plon1 : (double*)PyArray_DATA(lon2));

    bool equal_latlon_arrays = (plat1 == plat2);

    // Determine sizes
    npy_intp  size  = PyArray_SIZE(lat1);

    int i = 0, j = 0;

    // Create ll2 data for efficient nested loop
    std::vector<qdr_d_in> ll2(size);
    std::vector<qdr_d_in>::iterator pll2 = ll2.begin();
    while (i < size) {
        pll2->init(DEG2RAD * *plat2, DEG2RAD * *plon2);
        ++i; ++plat2; ++plon2; ++pll2;
    }

    // Create output matrices
    npy_intp shape[] = {size, size};
    PyObject* dst = PyArray_SimpleNew(2, shape, NPY_DOUBLE);

    // Nested loop to calculate dist matrix
    i = 0;
    double *pdst = (double*)PyArray_DATA((PyArrayObject*)dst);
    if (equal_latlon_arrays) {
        double *pdst_T = pdst;
        std::vector<qdr_d_in>::iterator pll1 = ll2.begin();
        pll2 = ll2.begin();
        while (i < size) {
            while (j < size) {
                if (i == j) {
                    *pdst = 0.0;
                } else {
                    *pdst = *pdst_T = M2NM * dist(*pll1, *pll2);
                }
                ++j; ++pll2; ++pdst;
                pdst_T += size;
            }
            ++i; ++pll1;
            pdst += i;
            pdst_T = pdst;
            j = i;
            pll2 = ll2.begin() + j;
        }
    } else {
        qdr_d_in ll1;
        while (i < size) {
            ll1.init(DEG2RAD * *plat1, DEG2RAD * *plon1);
            pll2 = ll2.begin();
            while (j < size) {
                if (i == j) {
                    *pdst = 0.0;
                } else {
                    *pdst = M2NM * dist(ll1, *pll2);
                }
                ++j; ++pll2; ++pdst;
            }
            ++i; ++plat1; ++plon1;
            j = 0;
        }
    }
    Py_DECREF(lat1);
    Py_DECREF(lon1);
    Py_XDECREF(lat2);// Py_XDECREF checks for NULL
    Py_XDECREF(lon2);

    return Py_BuildValue("N", dst);
};

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
    PyObject      *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
    PyArrayObject *lat1 = NULL, *lon1 = NULL, *lat2 = NULL, *lon2 = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4))
        return NULL;

    lat1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lon1 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lat2 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lon2 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (lat1 == NULL || lon1 == NULL) return NULL;

    double *plat1 = (double*)PyArray_DATA(lat1),
           *plon1 = (double*)PyArray_DATA(lon1);

    double *plat2 = (lat2 == NULL ? plat1 : (double*)PyArray_DATA(lat2)),
           *plon2 = (lon2 == NULL ? plon1 : (double*)PyArray_DATA(lon2));

    bool equal_latlon_arrays = (plat1 == plat2);

    // Determine sizes
    npy_intp  size  = PyArray_SIZE(lat1);

    // Create output matrices
    npy_intp shape[] = {size, size};
    PyObject* dst = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
    double *pdst  = (double*)PyArray_DATA((PyArrayObject*)dst);
    // Nested loop to calculate dist matrix
    int i = 0, j = 0;
    if (equal_latlon_arrays) {
        double *pdst_T = pdst;
        while (i < size) {
            while (j < size) {
                if (i == j) {
                    *pdst = 0.0;
                } else {
                    *pdst = *pdst_T = M2NM * kwikdist(
                        kwik_in(DEG2RAD * *plat1, DEG2RAD * *plon1, DEG2RAD * *plat2, DEG2RAD * *plon2));
                }
                ++j; ++plat2; ++plon2; ++pdst;
                pdst_T += size;
            }
            ++i; ++plat1; ++plon1;
            pdst += i;
            pdst_T = pdst;
            j = i;
            plat2 = (double*)PyArray_DATA(lat2) + j;
            plon2 = (double*)PyArray_DATA(lon2) + j;
        }
    } else {
        while (i < size) {
            while (j < size) {
                if (i == j) {
                    *pdst = 0.0;
                } else {
                    *pdst = M2NM * kwikdist(
                        kwik_in(DEG2RAD * *plat1, DEG2RAD * *plon1, DEG2RAD * *plat2, DEG2RAD * *plon2));
                }
                ++j; ++plat2; ++plon2; ++pdst;
            }
            ++i; ++plat1; ++plon1;
            j = 0;
            plat2 = (double*)PyArray_DATA(lat2);
            plon2 = (double*)PyArray_DATA(lon2);
        }
    }
    Py_DECREF(lat1);
    Py_DECREF(lon1);
    Py_XDECREF(lat2);// Py_XDECREF checks for NULL
    Py_XDECREF(lon2);

    return Py_BuildValue("N", dst);
};

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
    PyObject      *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
    PyArrayObject *lat1 = NULL, *lon1 = NULL, *lat2 = NULL, *lon2 = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4))
        return NULL;

    lat1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lon1 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lat2 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    lon2 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (lat1 == NULL || lon1 == NULL) return NULL;

    double *plat1 = (double*)PyArray_DATA(lat1),
           *plon1 = (double*)PyArray_DATA(lon1);

    double *plat2 = (lat2 == NULL ? plat1 : (double*)PyArray_DATA(lat2)),
           *plon2 = (lon2 == NULL ? plon1 : (double*)PyArray_DATA(lon2));

    bool equal_latlon_arrays = (plat1 == plat2);

    // Determine sizes
    npy_intp  size  = PyArray_SIZE(lat1);

    // Create output matrices
    npy_intp shape[] = {size, size};
    PyObject *qdr  = PyArray_SimpleNew(2, shape, NPY_DOUBLE),
             *dst  = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
    double   *pqdr = (double*)PyArray_DATA((PyArrayObject*)qdr),
             *pdst = (double*)PyArray_DATA((PyArrayObject*)dst);
    // Nested loop to calculate dist matrix
    int i = 0, j = 0;
    if (equal_latlon_arrays) {
        double *pqdr_T = pqdr,
               *pdst_T = pdst;
        while (i < size) {
            while (j < size) {
                if (i == j) {
                    *pqdr = 0.0;
                    *pdst = 0.0;
                } else {
                    kwik_in in(DEG2RAD * *plat1, DEG2RAD * *plon1, DEG2RAD * *plat2, DEG2RAD * *plon2);
                    *pqdr = RAD2DEG * kwikqdr(in);
                    *pqdr_T = fmod(*pqdr + 180.0, 360.0);
                    *pdst = *pdst_T = M2NM * kwikdist(in);
                }
                ++j; ++plat2; ++plon2; ++pqdr; ++pdst;
                pqdr_T += size; pdst_T += size;
            }
            ++i; ++plat1; ++plon1;
            pqdr += i; pdst += i;
            pqdr_T = pqdr; pdst_T = pdst;
            j = i;
            plat2 = (double*)PyArray_DATA(lat2) + j;
            plon2 = (double*)PyArray_DATA(lon2) + j;
        }
    } else {
        while (i < size) {
            while (j < size) {
                if (i == j) {
                    *pqdr = 0.0;
                    *pdst = 0.0;
                } else {
                    kwik_in in(DEG2RAD * *plat1, DEG2RAD * *plon1, DEG2RAD * *plat2, DEG2RAD * *plon2);
                    *pqdr = RAD2DEG * kwikqdr(in);
                    *pdst = M2NM * kwikdist(in);
                }
                ++j; ++plat2; ++plon2; ++pqdr; ++pdst;
            }
            ++i; ++plat1; ++plon1;
            j = 0;
            plat2 = (double*)PyArray_DATA(lat2);
            plon2 = (double*)PyArray_DATA(lon2);
        }
    }
    Py_DECREF(lat1);
    Py_DECREF(lon1);
    Py_XDECREF(lat2);// Py_XDECREF checks for NULL
    Py_XDECREF(lon2);

    return Py_BuildValue("NN", qdr, dst);
};

static struct PyMethodDef methods[] = {
    {"rwgs84", cgeo_rwgs84, METH_VARARGS, "Get local earth radius using WGS'84 spec."},
    {"rwgs84_matrix", cgeo_rwgs84, METH_VARARGS, "Get local earth radius using WGS'84 spec (for vectors)."},
    {"qdrdist", cgeo_qdrdist, METH_VARARGS, "Calculate bearing and distance between lat1+lon1 and lat2+lon2"},
    {"qdrdist_matrix", cgeo_qdrdist_matrix, METH_VARARGS, "Calculate bearing and distance matrices between vectors lat1+lon1/lat2+lon2"},
    {"latlondist", cgeo_latlondist, METH_VARARGS, "Calculate distance between lat1+lon1 and lat2+lon2"},
    {"latlondist_matrix", cgeo_latlondist_matrix, METH_VARARGS, "Calculate distance matrix between vectors lat1+lon1/lat2+lon2"},
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
