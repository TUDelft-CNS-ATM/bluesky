#include <cmath>
#include <pyhelpers.hpp>
#include <geo.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

struct Dbconf {
    PyObject* self;
    double dtlookahead, R, R2, dh;
    int nconf;
    // Per aircraft arrays/lists
    PyBoolArrayAttr asasactive;// not recreated with each call to detect
    PyListAttr  iconf;// iconf is a list of (empty) lists
    // Per conflict lists
    PyListAttr latowncpa, lonowncpa, altowncpa, confpairs,
               LOSlist_now, LOSlist_exp, LOSlist_all, 
               conflist_now, conflist_exp, conflist_all;
    // Also used in CR methods, but not yet created here:
    // qdr, dist, u, v, dx, dy, tcpa, dalt, tinconf, toutconf
    // LOSmaxsev, LOShmaxsev, LOSvmaxsev

    Dbconf(PyObject* self) : self(self),
        dtlookahead(GetAttrDouble(self, "dtlookahead")), R(GetAttrDouble(self, "R")),
        dh(GetAttrDouble(self, "dh")), nconf(0), asasactive(self, "asasactive")
        {R2 = R * R;}

    void copyback() {
        PyObject_SetAttrString(self, "latowncpa", latowncpa.attr);
        PyObject_SetAttrString(self, "lonowncpa", lonowncpa.attr);
        PyObject_SetAttrString(self, "altowncpa", altowncpa.attr);
        PyObject_SetAttrString(self, "confpairs", confpairs.attr);
        PyObject_SetAttrString(self, "iconf", iconf.attr);
        PyObject_SetAttrString(self, "nconf", PyInt_FromLong(nconf));
    }
};

struct conflict {double tin, tout, latcpa, loncpa; bool LOS; };
inline bool detect_hor(const Dbconf& params, conflict& conf,
                       const qdr_d_in& ll1, const double& gs1, const double& trk1,
                       const qdr_d_in& ll2, const double& gs2, const double& trk2)
{
    double u1        = gs1 * sin(trk1),
           v1        = gs1 * cos(trk1),
           u2        = gs2 * sin(trk2),
           v2        = gs2 * cos(trk2);
    double du        = u1 - u2,
           dv        = v1 - v2;

   // First a coarse flat-earth check to skip the most unlikely candidates for a conflict
   if (std::max((fabs(ll2.lon - ll1.lon) * std::min(ll1.coslat, ll2.coslat) * re - params.R) / fabs(du),
                 (fabs(ll2.lat - ll1.lat) * re - params.R) / fabs(dv))
        > 1.05 * params.dtlookahead)
     return false;

    double d         = dist(ll1, ll2),
           q         = qdr(ll1, ll2);
    double dx        = d * sin(q),
           dy        = d * cos(q);
           

    double vreldotdx = du * dx + dv * dy;
    conf.LOS         = d < params.R;
    
    // If diverging and separated, no horizontal conflict
    if (vreldotdx < 0.0 && !conf.LOS) return false;

    double vrel2     = (du * du + dv * dv);
    if (conf.LOS) {
        // Horizontal separation is already lost: calculate PZ exit time
        conf.tin     = 0.0;
        conf.tout    = (sqrt(vreldotdx * vreldotdx - d * d * vrel2) - vreldotdx) / (d * d);
        conf.latcpa  = ll1.lat;
        conf.loncpa  = ll1.lon;
        return true;
    } else {
        double t_cpa = vreldotdx / vrel2;
        double CPA2  = d * d - vreldotdx * t_cpa;
        double dt    = sqrt((params.R2 - CPA2) / vrel2);
        conf.tin     = t_cpa - dt;
        conf.tout    = t_cpa + dt;
        // Calculate CPA position of AC1 if conflict is in detection range
        if (conf.tin <= params.dtlookahead) {
          conf.latcpa = ll1.lat + v1 * t_cpa / re;
          conf.loncpa = ll1.lon + u1 * t_cpa / re / ll1.coslat;
          return true;
        }
    }
    return false;
}

inline bool detect_ver(const Dbconf& params, conflict& conf,
                       const double& dalt, const double& dvs)
{
    conf.LOS = fabs(dalt) < params.dh;
    if (fabs(dvs) > 1e-6) {
        double tcrosshi = (dalt + params.dh) / -dvs,
               tcrosslo = (dalt - params.dh) / -dvs;

        conf.tin  = std::max(0.0, std::min(tcrosslo, tcrosshi));
        conf.tout = std::max(tcrosslo, tcrosshi);

        // Vertical conflict if t_in is within lookahead horizon, and t_out > 0
        return (conf.tin <= params.dtlookahead && conf.tout > 0.0);
    } else {
        conf.tin  = 0.0;
        conf.tout = 1e4;
        return conf.LOS;
    }
}
