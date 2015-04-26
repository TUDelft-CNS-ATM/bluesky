class Waypoint():
    """
    Waypoint class definition: Waypoint element of FMS route (basic 
        FMS functionality)

    waypoint(name,lat,lon,spd,alt,wptype)  SPD,ALT! as in Legs page on CDU
        spd and alt are constraints, -999 if None (alt=-999)
        
    Currently stub. (Route-class takes care of waypoints.)
    """
    
    def __init__(self,name,lat,lon,spd=-999,alt=-999,wptype=0):
        self.name  = name
        self.lat   = lat
        self.lon   = lon
        self.alt   = alt   #[m] negative value means no alt specificied
        self.spd   = spd   #[m/s] negative value means no alt specificied
        self.type  = wptype
        
        self.normal      = 0
        self.origin      = 1
        self.destination = 2
        self.acposition  = 4
        
        return

