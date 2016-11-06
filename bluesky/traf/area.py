import numpy as np
from ..tools import areafilter

class Area:
    def __init__(self,traf):
        # Traffic area: delete traffic when it leaves this area (so not when outside)
        
        self.traf   = traf
        
        # Parameters of area
        self.active = False 
        self.dt     = 5.0     # [s] frequency of area check (simtime)
        self.t0     = -100.   # last time checked
        self.name   = None
        
        # Boolean array whether aircraft are in circle or not
        self.inside = np.array([],dtype = np.bool) # In test area or not
        
        # Taxi switch
        self.swtaxi = False  # Default OFF: Doesn't do anything. See comments of setTaxi fucntion below.

    def create(self):
        self.inside = np.append(self.inside,False)

    def delete(self,idx):
        self.inside = np.delete(self.inside,idx)

    def check(self,t):
        # ToDo: Add autodelete for descending with swTaxi:
        if self.swtaxi:
            pass # To be added!!!
        
        # Update area once per areadt seconds:
        if self.active and abs(t - self.t0) > self.dt:
            self.t0 = t
            
            # Find out which aircraft are inside the experiment area
            inside = areafilter.checkInside(self.name, self.traf.lat, self.traf.lon, self.traf.alt)
            
            # Determine the aircraft indexes that should be deleted
            delAircraftidx = np.intersect1d(np.where(np.array(self.inside)==True), np.where(np.array(inside)==False))

            # Update self.inside with the new inside
            self.inside = inside
            
            # delete all aicraft in delAircraftidx and log their flight statistics
            for acid in [self.traf.id[idx] for idx in delAircraftidx]:
                self.traf.delete(acid)

    def setArea(self, scr, args):
        ''' Set Experiment Area. Aicraft leaving the experiment area are deleted.
        Input can be exisiting shape name, or a box with optional altitude constrainsts.'''        
        
        # if all args are empty, then print out the current area status
        if len(args)==0:
            return True, "Area is currently " + ("ON" if self.active else "OFF") + \
                         "\nCurrent Area name is: " + str(self.name)   
        
        # start by checking if the first argument is a string -> then it is an area name
        if isinstance(args[0], str) and len(args)==1:
            if args[0] in areafilter.areas:
                # switch on Area, set it to the shape name
                self.name = args[0]
                self.active = True
                return True, "Area is set to " + str(self.name)
            elif args[0]=='OFF' or args[0]=='OF':
                # switch off the area                 
                areafilter.deleteArea(scr, self.name)
                self.active = False
                self.name = None
                return True, "Area is switched OFF"  
            else: 
                # shape name is unknown
                return False, "Shapename unknown. Please create shapename first or shapename is misspelled!"
        # if first argument is a float -> then make a box with the arguments
        elif (isinstance(args[0],float) or isinstance(args[0],int)) and 4<=len(args)<=6:
            self.active = True
            self.name = 'DELAREA'
            areafilter.defineArea(scr, self.name, 'BOX', args)
            return True, "Area is ON. Area name is: " + str(self.name)
        else:
            return False,  "Incorrect arguments" + \
                           "\nAREA Shapename/OFF or\n Area lat,lon,lat,lon,[top,bottom]"

    def setTaxi(self, flag):
        """ If you want to delete below 1500ft, make an box with the bottom at 1500ft and set it to Area. 
            This is because taxi does nothing """
        self.swtaxi = flag
        