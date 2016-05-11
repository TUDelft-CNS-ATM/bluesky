#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include "geo.hpp"
#include <iostream>
#define DEG2RAD 0.017453292519943295
#define RAD2DEG 57.29577951308232
#define M2NM 0.0005399568034557236
#define NM2M 1852.0

static PyObject* rwgs84(PyObject* self, PyObject* args)
{
    PyObject *arg1 = NULL;
    double lat;
    if (!PyArg_ParseTuple(args, "O", &arg1))
        return NULL;

    // Check if arg is an array
    if (PyArray_Check(arg1)) {
        PyArrayObject *arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        int       nd    = PyArray_NDIM(arr1);
        npy_intp  size  = PyArray_SIZE(arr1);
        npy_intp* shape = PyArray_DIMS(arr1);

        PyObject* radii = PyArray_SimpleNew(nd, shape, NPY_DOUBLE);

        double* pLatd = (double*)PyArray_DATA(arr1);
        double* pR    = (double*)PyArray_DATA((PyArrayObject*)radii);

        while (--size >= 0) {
            lat = DEG2RAD * *pLatd;
            *pR = _rwgs84(sin(lat), cos(lat));
            ++pLatd; ++pR;
        }

        Py_DECREF(arr1);
        return radii;
    } else {
        // arg is a scalar
        lat = DEG2RAD * PyFloat_AsDouble(arg1);
        return Py_BuildValue("d", _rwgs84(sin(lat), cos(lat)));
    }
};

static PyObject* qdrdist(PyObject* self, PyObject* args)
{
    PyObject *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4))
        return NULL;

    qdr_d_in ll1, ll2;

    // First check if lat1/lon1 are arrays
    if (PyArray_Check(arg1) && PyArray_Check(arg2)) {
        PyArrayObject* arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        PyArrayObject* arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        double *plat1 = (double*)PyArray_DATA(arr1),
               *plon1 = (double*)PyArray_DATA(arr2);
        npy_intp size = PyArray_SIZE(arr1);

        // Create return vectors
        PyObject* qdr = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
        PyObject* dst = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
        double *pqdr = (double*)PyArray_DATA((PyArrayObject*)qdr);
        double *pdst = (double*)PyArray_DATA((PyArrayObject*)dst);

        // Check if lat2/lon2 are also arrays
        if (PyArray_Check(arg3) && PyArray_Check(arg4)) {
            PyArrayObject* arr3 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            PyArrayObject* arr4 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            double *plat2 = (double*)PyArray_DATA(arr3),
                   *plon2 = (double*)PyArray_DATA(arr4);
            while (--size >= 0) {
                ll1.init(DEG2RAD * *plat1, DEG2RAD * *plon1);
                ll2.init(DEG2RAD * *plat2, DEG2RAD * *plon2);
                *pqdr = RAD2DEG * _qdr(ll1, ll2);
                *pdst = M2NM * _dist(ll1, ll2);
                ++plat1; ++plon1; ++plat2; ++plon2; ++pqdr; ++pdst;
            }
            Py_DECREF(arr3);
            Py_DECREF(arr4);
        } else {
            // lat2/lon2 are scalars
            ll2.init(DEG2RAD * PyFloat_AsDouble(arg3), DEG2RAD * PyFloat_AsDouble(arg4));
            while (--size >= 0) {
                ll1.init(DEG2RAD * *plat1, DEG2RAD * *plon1);
                *pqdr = RAD2DEG * _qdr(ll1, ll2);
                *pdst = M2NM * _dist(ll1, ll2);
                ++plat1; ++plon1; ++pqdr; ++pdst;
            }
        }
        Py_DECREF(arr1);
        Py_DECREF(arr2);
        return Py_BuildValue("NN", qdr, dst);
    }
    // Arguments should be all scalars
    ll1.init(DEG2RAD * PyFloat_AsDouble(arg1), DEG2RAD * PyFloat_AsDouble(arg2));
    ll2.init(DEG2RAD * PyFloat_AsDouble(arg3), DEG2RAD * PyFloat_AsDouble(arg4));
    return Py_BuildValue("dd", RAD2DEG * _qdr(ll1, ll2), M2NM * _dist(ll1, ll2));
};

static PyObject* qdrdist_vector(PyObject* self, PyObject* args)
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
    qdr_d_in ll2[size];
    qdr_d_in* pll2 = ll2;
    while (i < size) {
        pll2->init(DEG2RAD * *plat2, DEG2RAD * *plon2);
        ++i; ++plat2; ++plon2; ++pll2;
    }

    // Create output matrices
    npy_intp shape[] = {size, size};
    PyObject* qdr = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
    PyObject* dst = PyArray_SimpleNew(2, shape, NPY_DOUBLE);

    // Nested loop to calculate qdr and dist matrices
    i = 0;
    double *pqdr = (double*)PyArray_DATA((PyArrayObject*)qdr);
    double *pdst = (double*)PyArray_DATA((PyArrayObject*)dst);

    qdr_d_in ll1;
    while (i < size) {
        ll1.init(DEG2RAD * *plat1, DEG2RAD * *plon1);
        pll2 = ll2;
        while (j < size) {
            if (i == j) {
                *pqdr = 0.0;
                *pdst = 0.0;
            } else {
                *pqdr = RAD2DEG * _qdr(ll1, *pll2);
                *pdst = M2NM * _dist(ll1, *pll2);
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

    return Py_BuildValue("NN", qdr, dst);
};

static PyObject* latlondist(PyObject* self, PyObject* args)
{
    PyObject *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4))
        return NULL;

    qdr_d_in ll1, ll2;

    // First check if lat1/lon1 are arrays
    if (PyArray_Check(arg1) && PyArray_Check(arg2)) {
        PyArrayObject* arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        PyArrayObject* arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        double *plat1 = (double*)PyArray_DATA(arr1),
               *plon1 = (double*)PyArray_DATA(arr2);
        npy_intp size = PyArray_SIZE(arr1);

        // Create return vector
        PyObject* dst = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
        double *pdst = (double*)PyArray_DATA((PyArrayObject*)dst);

        // Check if lat2/lon2 are also arrays
        if (PyArray_Check(arg3) && PyArray_Check(arg4)) {
            PyArrayObject* arr3 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            PyArrayObject* arr4 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            double *plat2 = (double*)PyArray_DATA(arr3),
                   *plon2 = (double*)PyArray_DATA(arr4);
            while (--size >= 0) {
                ll1.init(DEG2RAD * *plat1, DEG2RAD * *plon1);
                ll2.init(DEG2RAD * *plat2, DEG2RAD * *plon2);
                *pdst = M2NM * _dist(ll1, ll2);
                ++plat1; ++plon1; ++plat2; ++plon2; ++pdst;
            }
            Py_DECREF(arr3);
            Py_DECREF(arr4);
        } else {
            // lat2/lon2 are scalars
            ll2.init(DEG2RAD * PyFloat_AsDouble(arg3), DEG2RAD * PyFloat_AsDouble(arg4));
            while (--size >= 0) {
                ll1.init(DEG2RAD * *plat1, DEG2RAD * *plon1);
                *pdst = M2NM * _dist(ll1, ll2);
                ++plat1; ++plon1; ++pdst;
            }
        }
        Py_DECREF(arr1);
        Py_DECREF(arr2);
        return dst;
    }
    // Arguments should be all scalars
    ll1.init(DEG2RAD * PyFloat_AsDouble(arg1), DEG2RAD * PyFloat_AsDouble(arg2));
    ll2.init(DEG2RAD * PyFloat_AsDouble(arg3), DEG2RAD * PyFloat_AsDouble(arg4));
    return Py_BuildValue("d", M2NM * _dist(ll1, ll2));
};

static PyObject* latlondist_vector(PyObject* self, PyObject* args)
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
    qdr_d_in ll2[size];
    qdr_d_in* pll2 = ll2;
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
        qdr_d_in *pll1 = ll2;
        pll2 = ll2;
        while (i < size) {
            while (j < size) {
                if (i == j) {
                    *pdst = 0.0;
                } else {
                    *pdst = *pdst_T = M2NM * _dist(*pll1, *pll2);
                }
                ++j; ++pll2; ++pdst;
                pdst_T += size;
            }
            ++i; ++pll1;
            pdst += i;
            pdst_T = pdst;
            j = i;
            pll2 = ll2 + j;
        }
    } else {
        qdr_d_in ll1;
        while (i < size) {
            ll1.init(DEG2RAD * *plat1, DEG2RAD * *plon1);
            pll2 = ll2;
            while (j < size) {
                if (i == j) {
                    *pdst = 0.0;
                } else {
                    *pdst = M2NM * _dist(ll1, *pll2);
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

static PyObject* wgsg(PyObject* self, PyObject* args)
{
    PyObject *arg1 = NULL;
    if (!PyArg_ParseTuple(args, "O", &arg1))
        return NULL;

    // Check if arg is an array
    if (PyArray_Check(arg1)) {
        PyArrayObject *arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        int       nd    = PyArray_NDIM(arr1);
        npy_intp  size  = PyArray_SIZE(arr1);
        npy_intp* shape = PyArray_DIMS(arr1);
        PyObject* g     = PyArray_SimpleNew(nd, shape, NPY_DOUBLE);
        double* pLatd   = (double*)PyArray_DATA(arr1);
        double* pg      = (double*)PyArray_DATA((PyArrayObject*)g);

        while (--size >= 0) {
            *pg = _wgsg(DEG2RAD * *pLatd);
            ++pLatd; ++pg;
        }

        Py_DECREF(arr1);
        return g;
    } else {
        // arg is a scalar
        return Py_BuildValue("d", _wgsg(DEG2RAD * PyFloat_AsDouble(arg1)));
    }
};

static PyObject* qdrpos(PyObject* self, PyObject* args)
{
    PyObject      *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4))
        return NULL;

    // Check if args are arrays
    if (PyArray_Check(arg1) && PyArray_Check(arg2) &&
        PyArray_Check(arg3) && PyArray_Check(arg4)) {
        PyArrayObject *arr1  = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
                      *arr2  = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
                      *arr3  = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
                      *arr4  = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        double        *plat1 = (double*)PyArray_DATA((PyArrayObject*)arr1),
                      *plon1 = (double*)PyArray_DATA((PyArrayObject*)arr2),
                      *pqdr  = (double*)PyArray_DATA((PyArrayObject*)arr3),
                      *pdst  = (double*)PyArray_DATA((PyArrayObject*)arr4);

        // Determine sizes
        npy_intp  size  = PyArray_SIZE(arr1);
        
        // Create output matrices
        PyObject *lat2  = PyArray_SimpleNew(1, &size, NPY_DOUBLE),
                 *lon2  = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
        double   *plat2 = (double*)PyArray_DATA((PyArrayObject*)lat2),
                 *plon2 = (double*)PyArray_DATA((PyArrayObject*)lon2);

        // Calculate the output vectors
        pos newpos;
        while (--size >= 0) {
            newpos = _qdrpos(DEG2RAD * *plat1, DEG2RAD * *plon1, DEG2RAD * *pqdr, NM2M * *pdst);
            *plat2 = RAD2DEG * newpos.lat;
            *plon2 = RAD2DEG * newpos.lon;
            ++plat1; ++plon1; ++pdst; ++pqdr; ++plat2; ++plon2;
        }

        Py_DECREF(arr1);
        Py_DECREF(arr2);
        Py_DECREF(arr3);
        Py_DECREF(arr4);

        return Py_BuildValue("NN", lat2, lon2);
    } else {
        // Args should be scalars
        pos newpos = _qdrpos(DEG2RAD * PyFloat_AsDouble(arg1), DEG2RAD * PyFloat_AsDouble(arg2),
                             DEG2RAD * PyFloat_AsDouble(arg3), NM2M    * PyFloat_AsDouble(arg4));
        return Py_BuildValue("dd", RAD2DEG * newpos.lat, RAD2DEG * newpos.lon);
    }
};

static PyObject* kwikdist(PyObject* self, PyObject* args)
{
    PyObject *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4))
        return NULL;

    // First check if lat1/lon1 are arrays
    if (PyArray_Check(arg1) && PyArray_Check(arg2)) {
        PyArrayObject* arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        PyArrayObject* arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        double *plat1 = (double*)PyArray_DATA(arr1),
               *plon1 = (double*)PyArray_DATA(arr2);
        npy_intp size = PyArray_SIZE(arr1);

        // Create return vector
        PyObject* dst = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
        double *pdst = (double*)PyArray_DATA((PyArrayObject*)dst);

        // Check if lat2/lon2 are also arrays
        if (PyArray_Check(arg3) && PyArray_Check(arg4)) {
            PyArrayObject* arr3 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            PyArrayObject* arr4 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            double *plat2 = (double*)PyArray_DATA(arr3),
                   *plon2 = (double*)PyArray_DATA(arr4);
            while (--size >= 0) {
                *pdst = M2NM * _kwikdist(kwik_in(
                    DEG2RAD * *plat1, DEG2RAD * *plon1,
                    DEG2RAD * *plat2, DEG2RAD * *plon2));
                ++plat1; ++plon1; ++plat2; ++plon2; ++pdst;
            }
            Py_DECREF(arr3);
            Py_DECREF(arr4);
        } else {
            // lat2/lon2 are scalars
            double lat2 = DEG2RAD * PyFloat_AsDouble(arg3),
                   lon2 = DEG2RAD * PyFloat_AsDouble(arg4);
            while (--size >= 0) {
                *pdst = *pdst = M2NM * _kwikdist(kwik_in(
                    DEG2RAD * *plat1, DEG2RAD * *plon1, lat2, lon2));
                ++plat1; ++plon1; ++pdst;
            }
        }
        Py_DECREF(arr1);
        Py_DECREF(arr2);
        return dst;
    }
    // Arguments should be all scalars
    return Py_BuildValue("d", M2NM * _kwikdist(
                kwik_in(DEG2RAD * PyFloat_AsDouble(arg1),
                        DEG2RAD * PyFloat_AsDouble(arg2),
                        DEG2RAD * PyFloat_AsDouble(arg3),
                        DEG2RAD * PyFloat_AsDouble(arg4))));
};

static PyObject* kwikdist_vector(PyObject* self, PyObject* args)
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
                    *pdst = *pdst_T = M2NM * _kwikdist(
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
                    *pdst = M2NM * _kwikdist(
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

static PyObject* kwikqdrdist(PyObject* self, PyObject* args)
{
    PyObject *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4))
        return NULL;

    // First check if lat1/lon1 are arrays
    if (PyArray_Check(arg1) && PyArray_Check(arg2)) {
        PyArrayObject* arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        PyArrayObject* arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        double *plat1 = (double*)PyArray_DATA(arr1),
               *plon1 = (double*)PyArray_DATA(arr2);
        npy_intp size = PyArray_SIZE(arr1);

        // Create return vectors
        PyObject *qdr  = PyArray_SimpleNew(1, &size, NPY_DOUBLE),
                 *dst  = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
        double   *pqdr = (double*)PyArray_DATA((PyArrayObject*)qdr),
                 *pdst = (double*)PyArray_DATA((PyArrayObject*)dst);

        // Check if lat2/lon2 are also arrays
        if (PyArray_Check(arg3) && PyArray_Check(arg4)) {
            PyArrayObject* arr3 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            PyArrayObject* arr4 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            double *plat2 = (double*)PyArray_DATA(arr3),
                   *plon2 = (double*)PyArray_DATA(arr4);
            while (--size >= 0) {
                kwik_in in(DEG2RAD * *plat1, DEG2RAD * *plon1, DEG2RAD * *plat2, DEG2RAD * *plon2);
                *pqdr = RAD2DEG * _kwikqdr(in);
                *pdst = M2NM * _kwikdist(in);
                ++plat1; ++plon1; ++plat2; ++plon2; ++pqdr; ++pdst;
            }
            Py_DECREF(arr3);
            Py_DECREF(arr4);
        } else {
            // lat2/lon2 are scalars
            double lat2 = DEG2RAD * PyFloat_AsDouble(arg3),
                   lon2 = DEG2RAD * PyFloat_AsDouble(arg4);
            while (--size >= 0) {
                kwik_in in(DEG2RAD * *plat1, DEG2RAD * *plon1, lat2, lon2);
                *pqdr = RAD2DEG * _kwikqdr(in);
                *pdst = M2NM * _kwikdist(in);
                ++plat1; ++plon1; ++pqdr; ++pdst;
            }
        }
        Py_DECREF(arr1);
        Py_DECREF(arr2);
        return dst;
    }
    // Arguments should be all scalars
    kwik_in in(DEG2RAD * PyFloat_AsDouble(arg1),
               DEG2RAD * PyFloat_AsDouble(arg2),
               DEG2RAD * PyFloat_AsDouble(arg3),
               DEG2RAD * PyFloat_AsDouble(arg4));
    return Py_BuildValue("dd", RAD2DEG * _kwikqdr(in), M2NM * _kwikdist(in));
};

static PyObject* kwikqdrdist_vector(PyObject* self, PyObject* args)
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
                    *pqdr = *pqdr_T = RAD2DEG * _kwikqdr(in);
                    *pdst = *pdst_T = M2NM * _kwikdist(in);
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
                    *pqdr = RAD2DEG * _kwikqdr(in);
                    *pdst = M2NM * _kwikdist(in);
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
    {"rwgs84", rwgs84, METH_VARARGS, "Get local earth radius using WGS'84 spec."},
    {"rwgs84_vector", rwgs84, METH_VARARGS, "Get local earth radius using WGS'84 spec (for vectors)."},
    {"qdrdist", qdrdist, METH_VARARGS, "Calculate bearing and distance between lat1+lon1 and lat2+lon2"},
    {"qdrdist_vector", qdrdist_vector, METH_VARARGS, "Calculate bearing and distance matrices between vectors lat1+lon1/lat2+lon2"},
    {"latlondist", latlondist, METH_VARARGS, "Calculate distance between lat1+lon1 and lat2+lon2"},
    {"latlondist_vector", latlondist_vector, METH_VARARGS, "Calculate distance matrix between vectors lat1+lon1/lat2+lon2"},
    {"wgsg", wgsg, METH_VARARGS, "Gravity acceleration at a given latitude according to WGS'84"},
    {"qdrpos", qdrpos, METH_VARARGS, "Calculate position from reference position, bearing and distance"},
    {"kwikdist", kwikdist, METH_VARARGS, "Quick and dirty dist [nm]"},
    {"kwikdist_vector", kwikdist_vector, METH_VARARGS, "Quick and dirty dist [nm] (for vectors)"},
    {"kwikqdrdist", kwikqdrdist, METH_VARARGS, "Quick and dirty dist [nm] and bearing [deg]"},
    {"kwikqdrdist_vector", kwikqdrdist_vector, METH_VARARGS, "Quick and dirty dist [nm] and bearing [deg] (for vectors)"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcgeo()
{
    Py_InitModule("cgeo", methods);
    import_array();
};
