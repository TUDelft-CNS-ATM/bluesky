#include <cmath>
#include <pyhelpers.hpp>
#include <geo.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

struct conflict {
    double tin, tout, dcpa, tcpa, q, d; bool LOS;
    conflict() : tin(0.0), tout(0.0), dcpa(0.0), tcpa(0.0), q(0.0), d(0.0), LOS(false) {}
};
inline bool detect_hor(conflict& conf, const double& RPZ, const double& tlookahead,
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
   if (std::max((fabs(ll2.lon - ll1.lon) * std::min(ll1.coslat, ll2.coslat) * re - RPZ) / fabs(du),
                 (fabs(ll2.lat - ll1.lat) * re - RPZ) / fabs(dv))
        > 1.05 * tlookahead)
     return false;

    conf.d = dist(ll1, ll2),
    conf.q = qdr(ll1, ll2);
    double dx        = conf.d * sin(conf.q),
           dy        = conf.d * cos(conf.q);


    double vreldotdx = du * dx + dv * dy;
    conf.LOS         = conf.d < RPZ;

    // If diverging and separated, no horizontal conflict
    if (vreldotdx < 0.0 && !conf.LOS) return false;

    double vrel2     = (du * du + dv * dv);
    conf.tcpa = vreldotdx / vrel2;

    double CPA2  = conf.d * conf.d - vreldotdx * conf.tcpa;
    conf.dcpa = sqrt(CPA2);
    double dt    = sqrt(std::max(RPZ * RPZ - CPA2, 0.0) / vrel2);
    conf.tin     = conf.tcpa - dt;
    conf.tout    = conf.tcpa + dt;
    // Conflict is in detection range?
    return (conf.LOS || conf.tin <= tlookahead);
}

inline bool detect_ver(conflict& conf, const double& HPZ, const double& tlookahead,
                       const double& dalt, const double& dvs)
{
    conf.LOS = fabs(dalt) < HPZ;
    if (fabs(dvs) > 1e-6) {
        double tcrosshi = (dalt + HPZ) / -dvs,
               tcrosslo = (dalt - HPZ) / -dvs;

        conf.tin  = std::max(0.0, std::min(tcrosslo, tcrosshi));
        conf.tout = std::max(tcrosslo, tcrosshi);

        // Vertical conflict if t_in is within lookahead horizon, and t_out > 0
        return (conf.tin <= tlookahead && conf.tout > 0.0);
    } else {
        conf.tin  = 0.0;
        conf.tout = 1e4;
        return conf.LOS;
    }
}
