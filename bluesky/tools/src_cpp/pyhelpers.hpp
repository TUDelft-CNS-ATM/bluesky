#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include <Python.h>
#include "structmember.h"
#include "numpy/arrayobject.h"

template<typename T> int atype();
template<> int atype<double>() {return NPY_DOUBLE;};
template<> int atype<npy_bool>() {return NPY_BOOL;};

struct PyAttr {
    PyObject* attr;
    PyAttr(PyObject* attr) : attr(attr) {}
    PyAttr(PyObject* parent, const char* name) :
        attr(PyObject_GetAttrString(parent, name)) {}
    PyAttr(const PyAttr& parent, const char* name) :
        attr(PyObject_GetAttrString(parent.attr, name)) {}
    ~PyAttr() {Py_XDECREF(attr);}
};

template<typename T, int T_ENUM>
struct PyArrayAttr: public PyAttr {
    T *ptr, *ptr_start;
    PyArrayObject* arr;
    PyArrayAttr(PyObject* parent, const char* name) :
        PyAttr(parent, name) {init();}
    PyArrayAttr(const PyAttr& parent, const char* name) :
        PyAttr(parent, name) {init();}
    PyArrayAttr(PyObject* attr) : PyAttr(attr) {init();}

    PyArrayAttr(const int length) : PyAttr(NULL) {
        int nd = 1;
        npy_intp dims[] = {length};
        arr = (PyArrayObject*)PyArray_SimpleNew(nd, dims, T_ENUM);
        if (arr != NULL) {
            ptr_start = ptr = (T*)PyArray_DATA(arr);
        }
    }

    void init() {
        if (attr != NULL) {
            arr = (PyArrayObject*)PyArray_FROM_OTF(attr, atype<T>(), NPY_ARRAY_IN_ARRAY);
            if (arr != NULL) {
                ptr_start = ptr = (T*)PyArray_DATA(arr);
            }
        }
    }
    ~PyArrayAttr()
    {
        Py_XDECREF(arr);
    }

    operator bool() const {return (attr != NULL);}
    npy_intp size() const {return PyArray_SIZE(arr);}
};

typedef PyArrayAttr<double, NPY_DOUBLE> PyDoubleArrayAttr;
typedef PyArrayAttr<npy_bool, NPY_BOOL> PyBoolArrayAttr;

struct PyListAttr: public PyAttr {
    PyListAttr(int size=0) : PyAttr(PyList_New(size)) {}
    PyListAttr(PyObject* attr) : PyAttr(attr) {}
    PyListAttr(PyObject* parent, const char* name) : PyAttr(parent, name) {}
    PyListAttr(const PyAttr& parent, const char* name) : PyAttr(parent, name) {}
    PyObject* operator[](Py_ssize_t idx) const {return PyList_GetItem(attr, idx);}
    inline int setItem(const Py_ssize_t& idx, const int& item) {return PyList_SetItem(attr, idx, PyLong_FromLong(item));}
    inline int setItem(const Py_ssize_t& idx, const double& item) {return PyList_SetItem(attr, idx, PyFloat_FromDouble(item));}
    inline int setItem(const Py_ssize_t& idx, PyObject* item) {return PyList_SetItem(attr, idx, item);}
    inline int append(const int& item) {
        PyObject* o = PyLong_FromLong(item);
        int index = PyList_Append(attr, o);
        Py_DECREF(o);
        return index;}
    inline int append(const double& item) {
        PyObject* o = PyFloat_FromDouble(item);
        int index = PyList_Append(attr, o);
        Py_DECREF(o);
        return index;}
    inline int append(PyObject* item) {return PyList_Append(attr, item);}
};

double GetAttrDouble(PyObject* parent, const char* name) {
    PyAttr a(parent, name);
    return PyFloat_AsDouble(a.attr);
};

int GetAttrInt(PyObject* parent, const char* name) {
    PyAttr a(parent, name);
    return PyLong_AsLong(a.attr);
};
