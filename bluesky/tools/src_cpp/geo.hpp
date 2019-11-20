#include <cmath>

// Earth major and minor axes
static const double a  = 6378137.0,      // [m] Major semi-axis WGS-84
                    b  = 6356752.314245, // [m] Minor semi-axis WGS-84
                    re = 6371000.0,      // [m] average earth radius
                    a2 = a * a,
                    b2 = b * b,
                    a4 = a2 * a2,
                    b4 = b2 * b2;

// Return the distance from the Earth's center to a point on the spheroid
// surface at geodetic latitude, lat (radians). Pass cos(lat) and sin(lat)
inline double rwgs84(const double& sinlat, const double& coslat)
{
    // See https://en.wikipedia.org/wiki/Earth_radius#Geocentric_radius
    double  sinlat2 = sinlat * sinlat,
            coslat2 = coslat * coslat;

    // Calculate and return the radius in meters:
    return sqrt((a4 * coslat2 + b4 * sinlat2) /
                (a2 * coslat2 + b2 * sinlat2));
}

struct qdr_d_in  {
    double lat, lon, sinlat, coslat;
    void init(const double& lat, const double& lon) {
        this->lat = lat; this->lon = lon;
        this->sinlat = sin(lat); this->coslat = cos(lat);
    }};

// return great-circle distance between two points.
// The implementation uses the Haversine formula:
//   a = sin²(Δφ/2) + cos φ1 * cos φ2 * sin²(Δλ/2)
//   c = 2 * atan2( sqrt(a), sqrt(1−a) )
//   d = R * c
// where
//   φ is latitude (radians)
//   λ is longitude
//   R is Earth’s radius
// see http://www.movable-type.co.uk/scripts/latlong.html#ortho-dist
inline double dist(const qdr_d_in& ll1, const qdr_d_in& ll2)
{
    double  sindlat2 = sin(0.5 * (ll2.lat - ll1.lat)),
            sindlon2 = sin(0.5 * (ll2.lon - ll1.lon));

    double r;
    if (ll1.lat * ll2.lat >= 0.0) {
        r = rwgs84(sin(0.5 * (ll1.lat + ll2.lat)), cos(0.5 * (ll1.lat + ll2.lat)));
    } else {
        r = 0.5 * ( fabs(ll1.lat) * (rwgs84(ll1.sinlat, ll1.coslat) + a) + 
                    fabs(ll2.lat) * (rwgs84(ll2.sinlat, ll2.coslat) + a))
                / ( fabs(ll1.lat) + fabs(ll2.lat));
    }

    double root = sindlat2 * sindlat2 + ll1.coslat * ll2.coslat * sindlon2 * sindlon2;
    return 2.0 * r * atan2(sqrt(root), sqrt(1.0 - root));
}

// initial bearing (radians) initial heading for a great-circle route from 
// point ll1, the start point, to point ll2, the end point.
//    θ = atan2( sin Δλ * cos φ2 , cos φ1 * sin φ2 − sin φ1 * cos φ2 * cos Δλ )
// where
//    φ1 is the latitude (radians) of the start point
//    λ1 is the longitude (radians) of the start point
//    φ2 is the latitude (radians) of the end point
//    λ2 is the longitude (radians) of the end point
//    Δλ is the difference in longitude
// see http://www.movable-type.co.uk/scripts/latlong.html#bearing
// see http://williams.best.vwh.net/avform.htm
inline double qdr(const qdr_d_in& ll1, const qdr_d_in& ll2)
{
    return atan2(sin(ll2.lon - ll1.lon) * ll2.coslat, 
                    ll2.sinlat * ll1.coslat - ll1.sinlat * ll2.coslat * cos(ll2.lon - ll1.lon));
}

inline double wgsg(const double& lat)
{
    static const double geq = 9.7803;  // m/s2 g at equator
    static const double e2 = 6.694e-3; // eccentricity
    static const double k  = 0.001932; // derived from flattening f, 1/f = 298.257223563

    double sinlat = sin(lat);
    return geq * (1.0 + k * sinlat * sinlat) / sqrt(1.0 - e2 * sinlat * sinlat);
}

// position on Earth, lat, lon (radians)
struct pos {double lat, lon;};

// Return the new position starting from point (lon1, lat1)
// with initial bearing qdr, for a distance dist
//   φ2 = asin( sin φ1 * cos δ + cos φ1 * sin δ * cos θ )
//   λ2 = λ1 + atan2( sin θ * sin δ * cos φ1, cos δ − sin φ1 * sin φ2 )
// where
//   φ is latitude,
//   λ is longitude,
//   θ is the bearing (clockwise from north),
//   δ is the angular distance d/R
//   d is the distance travelled
//   R is Earth’s radius
// see http://www.movable-type.co.uk/scripts/latlong.html#destPoint
inline pos qdrpos(const double& lat1, const double& lon1, const double& qdr, const double& dist)
{
    // Calculate new position
    double sinlat = sin(lat1),
           coslat = cos(lat1);
    double R      = rwgs84(sinlat, coslat);
    double sdr    = sin(dist / R),
           cdr    = cos(dist / R);
    pos newpos;
    newpos.lat = asin(sinlat * cdr + coslat * sdr * cos(qdr));

    newpos.lon = lon1 + atan2(sin(qdr) * sdr * coslat,
                              cdr - sinlat * sin(newpos.lat));
    return newpos;
}

inline pos kwikpos(const double& lat1, const double& lon1, const double& qdr, const double& dist)
{
    pos newpos;
    newpos.lat = lat1 + cos(qdr) * dist / re;
    newpos.lon = lon1 + sin(qdr) * dist / re / cos(lat1);
    return newpos;
}

struct kwik_in {double dlat, dlon, cavelat;
    kwik_in(const double& lat1, const double& lon1, const double& lat2, const double& lon2) :
        dlat(lat2 - lat1), dlon(lon2 - lon1), cavelat(cos(0.5 * (lat1 + lat2))) {};
};

// quick distance calculation (using equirectangular distance approximation)
inline double kwikdist(const kwik_in& in)
{
    double dangle  = sqrt(in.dlat * in.dlat + in.dlon * in.dlon * in.cavelat * in.cavelat);
    return re * dangle;
}

// struct posvec {double dx, dy; posvec(const double& dx, const double& dy): dx(dx), dy(dy) {}};
// inline posvec kwikposvec(const double &lat1, const double &lon1, const double &lat2, const double &lon2)
// {
//     return posvec(
//         re * (lon1 - lon2) * cos(0.5 * (lat1 + lat2)),
//         re * (lat1 - lat2)
//     );
// };

inline double kwikqdr(const kwik_in &in)
{
    return fmod(atan2(in.dlon * in.cavelat, in.dlat), 360.0);
}