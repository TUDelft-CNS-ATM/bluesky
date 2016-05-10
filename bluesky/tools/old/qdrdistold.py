def qdrdistold(latd1,lond1,latd2,lond2):
# Using WGS'84 calculate (input in degrees!)
# qdr [deg] = heading from 1 to 2
# d [nm]    = distance from 1 to 2 in nm

# Haversine with average radius
    r = 0.5*(rwgs84(latd1)+rwgs84(latd2))

    lat1 = radians(latd1)
    lon1 = radians(lond1)
    lat2 = radians(latd2)
    lon2 = radians(lond2)

    sin1 = sin(0.5*(lat2-lat1))
    sin2 = sin(0.5*(lon2-lon1))

    dist =  2.*r*asin(sqrt(sin1*sin1 +  \
            cos(lat1)*cos(lat2)*sin2*sin2))  / nm   # dist is in [nm]

# Bearing from Ref. http://www.movable-type.co.uk/scripts/latlong.html

    qdr = degrees(atan2( sin(lon2-lon1) * cos(lat2), \
                         cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1)))

    return qdr,dist

