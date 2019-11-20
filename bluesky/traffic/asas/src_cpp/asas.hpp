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
inline bool detect_hor(conflict &conf, const double &RPZ, const double &tlookahead,
                       const double &lat1, const double &lon1, const double &gs1, const double &trk1,
                       const double &lat2, const double &lon2, const double &gs2, const double &trk2)
{
    // The groundspeed of ownship and intruder as vectors
    double u1        = gs1 * sin(trk1),
           v1        = gs1 * cos(trk1),
           u2        = gs2 * sin(trk2),
           v2        = gs2 * cos(trk2);
    // The relative velocity vector
    double du        = u1 - u2,
           dv        = v1 - v2;
    kwik_in kin = kwik_in(lat1, lon1, lat2, lon2);
    conf.d = kwikdist(kin),
    conf.q = kwikqdr(kin);
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
    // Check whether there is currently a vertical loss of separation
    conf.LOS = fabs(dalt) < HPZ;
    /* If there is a nonzero vertical speed, check the start and end time of
       a vertical loss of separation (negative if diverging) */
    if (fabs(dvs) > 1e-6) {
        /* tcrosshi: time until crossing the top of the intruder PZ
           tcrosslo: time until crossing the bottom of the intruder PZ */
        double tcrosshi = (dalt + HPZ) / -dvs,
               tcrosslo = (dalt - HPZ) / -dvs;

        // Determine time in/out from tcrosslo,tcrosshi
        conf.tin  = std::max(0.0, std::min(tcrosslo, tcrosshi));
        conf.tout = std::max(tcrosslo, tcrosshi);

        // Vertical conflict if t_in is within lookahead horizon, and t_out > 0
        return (conf.tin <= tlookahead && conf.tout > 0.0);
    } else {
        // If VS is zero, tin, tout are relevant if there is already a LoS
        conf.tin  = 0.0;
        conf.tout = 1e4;
        return conf.LOS;
    }
}
