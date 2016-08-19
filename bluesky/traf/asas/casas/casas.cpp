#include <iostream>
#include <vector>
#include "asas.hpp"
#define DEG2RAD 0.017453292519943295
#define RAD2DEG 57.29577951308232
#define M2NM 0.0005399568034557236
#define NM2M 1852.0
#define KTS2MS 0.514444
#define FT2M 0.3048
#define FPM2MS 0.00508

static PyObject* casas_detect(PyObject* self, PyObject* args)
{
    PyObject *pyasas = NULL,
             *traf   = NULL;
    double   simt;
    if (!PyArg_ParseTuple(args, "OOd", &pyasas, &traf, &simt))
        return NULL;

    PyDoubleArrayAttr lat1(traf, "lat"),     lon1(traf, "lon"),     trk1(traf, "trk"),
                      gs1 (traf, "gs"),      alt1(traf, "alt"),     vs1 (traf, "vs"),
                      lat2(traf, "adsblat"), lon2(traf, "adsblon"), trk2(traf, "adsbtrk"),
                      gs2 (traf, "adsbgs"),  alt2(traf, "adsbalt"), vs2 (traf, "adsbvs");

    PyListAttr  acid(traf, "id");

    // Only continue if all arrays exist
    if (lat1 && lon1 && trk1 && gs1  && alt1 && vs1  && lat2 && lon2 && trk2 && gs2  && alt2 && vs2)
    {
        // Assume all arrays are the same size; only get the size of lat1
        npy_intp  size  = lat1.size();

        // Wrap dbconf in C struct
        Dbconf dbconf(pyasas);

        // Loop over all combinations of aircraft to detect conflicts
        conflict confhor, confver;
        double tin, tout;
        double dalt, dvs;
        qdr_d_in ll1;
        std::vector<qdr_d_in> ll2(size);
        std::vector<qdr_d_in>::iterator pll2 = ll2.begin();
        npy_bool asasactive = NPY_FALSE;

        // Pre-calculate intruder data
        for (unsigned int i = 0; i < size; ++i) {
            pll2->init(*lat2.ptr * DEG2RAD, *lon2.ptr * DEG2RAD);
            lat2.ptr++; lon2.ptr++; pll2++;
        }
        
        for (unsigned int i = 0; i < size; ++i) {
            ll1.init(*lat1.ptr * DEG2RAD, *lon1.ptr * DEG2RAD);
            pll2 = ll2.begin();
            PyListAttr acconfids;
            for (unsigned int j = 0; j < size; ++j) {
                if (i != j) {
                    // Vectical detection first
                    dalt = *alt1.ptr - *alt2.ptr;
                    dvs  = *vs1.ptr  - *vs2.ptr;
                    if (detect_ver(dbconf, confver, dalt, dvs)) {
                        // Horizontal detection
                        if (detect_hor(dbconf, confhor, 
                                       ll1,   *gs1.ptr, *trk1.ptr * DEG2RAD,
                                       *pll2, *gs2.ptr, *trk2.ptr * DEG2RAD)) {
                            tin  = std::max(confhor.tin, confver.tin);
                            tout = std::min(confhor.tout, confver.tout);
                            // Combined conflict?
                            if (tin <= dbconf.dtlookahead && tin < tout && tout > 0.0) {
                                // Add AC id to conflict list
                                dbconf.confpairs.append(PyTuple_Pack(2, acid[i], acid[j]));
                                dbconf.latowncpa.append(confhor.latcpa * RAD2DEG);
                                dbconf.lonowncpa.append(confhor.loncpa * RAD2DEG);

                                // Keep index for current ownship and increase conflict count
                                acconfids.append(dbconf.nconf);
                                dbconf.nconf++;
                                asasactive = NPY_TRUE;// ASAS should be active for this aircraft
                            }
                        }
                    }
                }
                pll2++; trk2.ptr++; gs2.ptr++; alt2.ptr++; vs2.ptr++;
            }
            dbconf.iconf.append(acconfids.attr);
            *dbconf.asasactive.ptr = asasactive;
            dbconf.asasactive.ptr++;
            asasactive = NPY_FALSE;
            trk2.ptr = trk2.ptr_start; gs2.ptr  = gs2.ptr_start;
            alt2.ptr = alt2.ptr_start; vs2.ptr  = vs2.ptr_start;
            lat1.ptr++; lon1.ptr++; trk1.ptr++; gs1.ptr++; alt1.ptr++; vs1.ptr++;
        }
        // Copy new lists back to python dbconf object
        dbconf.copyback();
    }

    Py_INCREF(Py_None);
    return Py_None;
};

static PyMethodDef methods[] = {
    {"detect", casas_detect, METH_VARARGS, "Detect conflicts for traffic"},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initcasas(void) 
{
    Py_InitModule("casas", methods);
    import_array();
}
