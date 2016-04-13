""" 
ASAS classes
   
Created by  : Jacco M. Hoekstra (TU Delft)
Date        : November 2013

Modifation  : Added Conflict Resolution, Trajectory Recovery
By          : Jerom Maas
Date        : October 2014

"""

#----------------------------------------------------------------

# Create a confict database
# Inputs:
#    lat [deg]  = array with traffic latitude
#    lon [deg]  = array with traffic longitude
#    alt [m]    = array with traffic altitude 
#    trk [deg]  = array with traffic track angle
#    gs  [m/s]  = array with ground speed [m/s]
#    vs  [m/s]  = array with vertical speed [m/s]
#
# Outputs:
#    swconfl = 2D array with True/False for conflict
#    dtconfl = time to conflict
 
import numpy as np
import sys, os
from ..tools.aero_np import qdrdist_vector,nm,qdrpos
from ..tools.aero import ft

# Find a way to import the required Conflict Resolution Class
sys.path.append('bluesky/traf/CDRmethods/')


class Dbconf():

# Constructor of conflict database, call with SI units (meters and seconds)

    def __init__(self,traf,tlook, R, dh):
        self.swasas      = True   # [-] whether to perform CD&R
        self.dtlookahead = tlook  # [s] lookahead time
        
        mar              = 1.15   # [-] Safety margin for evasion
        self.R           = R      # [m] Horizontal separation minimum
        self.dh          = dh     # [m] Vertical separation minimum
        self.Rm          = R*mar  # [m] Horizontal separation minimum + margin
        self.dhm         = dh*mar # [m] Vertical separation minimum + margin
        
        self.traf        = traf   # Traffic database object
        
        self.deletenames =[]      # List of aircraft outside test region
        
        self.vmin        =100.     # [m/s] Minimum ASAS velocity
        self.vmax        =500.     # [m/s] Maximum ASAS velocity
        self.vsmax       = 3000./60.*ft # [m/s] Max vertical speed
        self.vsmin       = -3000./60.*ft # [m/s] Min vertical speed
        
        self.swresodir   = 'COMB'  # desired directions of resolution methods: 
                                   # combined (COMB), horizontal only (HORIZ), vertical only (VERT)
        
        self.swprio      = False   # If true, then cruising aircraft have priority and will not resolve

        self.reset()              # Reset database
        self.SetCRmethod("DoNothing")
        return
        
    def SetCRmethod(self,method):
        self.CRname ="Undefined"
        self.CRmethod    = __import__(method)
        self.CRmethod.start(self)
        
    def SetResoDirection(self,direction):
        self.swresodir = direction

# Reset conflict database

    def reset(self):
        self.conf        = []     # Start with emtpy database: no conflicts
        self.nconf       = 0      # Number of detected conflicts
        self.swconfl     = np.array([])
        self.latowncpa = np.array([])
        self.lonowncpa = np.array([])
        self.altowncpa = np.array([])
        self.latintcpa = np.array([])
        self.lonintcpa = np.array([])
        self.altintcpa = np.array([])

        self.idown     = []
        self.idoth     = []

        self.conflist_all= [] #Create a list of all Conflicts       
        self.LOSlist_all = [] #Create a list of all Losses Of Separation
        self.conflist_exp= [] #Create a list of all Conflicts in experiment time
        self.LOSlist_exp = [] #Create a list of all Losses Of Separation in experiment time
        self.conflist_now= [] #Create a list of current Conflicts       
        self.LOSlist_now = [] #Create a list of current Losses Of Separation
        
        # For keeping track of locations with most severe intrusions
        self.LOSmaxsev=[] 
        self.LOShmaxsev=[]
        self.LOSvmaxsev=[]
        
        return


# ==================== Conflict Detection based on state ======================

    def detect(self):   
        if not self.swasas:
            return

#        t0_ = time.clock()     # Timing of ASAS calculation

# Horizontal conflict ---------------------------------------------------------

# qdlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
        qdlst = qdrdist_vector(np.mat(self.traf.lat),np.mat(self.traf.lon),\
                                  np.mat(self.traf.adsblat),np.mat(self.traf.adsblon))

# Convert results from mat-> array
        self.qdr      = np.array(qdlst[0])  # degrees
        I             = np.eye(self.traf.ntraf) # Identity matric of order ntraf
        self.dist     = np.array(qdlst[1])*nm + 1e9*I # meters i to j
                    
# Transmission noise
        if self.traf.ADSBtransnoise:
            # error in the determined bearing between two a/c
            bearingerror=np.random.normal(0,self.traf.transerror[0],self.qdr.shape) #degrees
            self.qdr+=bearingerror
            # error in the perceived distance between two a/c
            disterror=np.random.normal(0,self.traf.transerror[1],self.dist.shape) #meters
            self.dist+=disterror

# Calculate horizontal closest point of approach (CPA)        
        qdrrad  = np.radians(self.qdr)
        self.dx      = self.dist * np.sin(qdrrad) # is pos j rel to i
        self.dy      = self.dist * np.cos(qdrrad) # is pos j rel to i
        
        trkrad = np.radians(self.traf.trk)
        self.u      = self.traf.gs*np.sin(trkrad).reshape((1,len(trkrad)))  # m/s
        self.v      = self.traf.gs*np.cos(trkrad).reshape((1,len(trkrad)))  # m/s
        
        # parameters received through ADSB
        adsbtrkrad = np.radians(self.traf.adsbtrk)
        adsbu  = self.traf.adsbgs*np.sin(adsbtrkrad).reshape((1,len(adsbtrkrad)))  # m/s
        adsbv  = self.traf.adsbgs*np.cos(adsbtrkrad).reshape((1,len(adsbtrkrad)))  # m/s
        
        self.du = self.u - adsbu.T  # Speed du[i,j] is perceived eastern speed of i to j
        self.dv = self.v - adsbv.T  # Speed dv[i,j] is perceived northern speed of i to j
        
        dv2  = self.du*self.du+self.dv*self.dv
        dv2  = np.where(np.abs(dv2)<1e-6,1e-6,dv2) # limit lower absolute value
        
        vrel = np.sqrt(dv2)
        
        self.tcpa = -(self.du*self.dx + self.dv*self.dy) / dv2   + 1e9*I
        
       # Calculate CPA positions
        xcpa = self.tcpa*self.du
        ycpa = self.tcpa*self.dv

# Calculate distance^2 at CPA (minimum distance^2)
        dcpa2 = self.dist*self.dist-self.tcpa*self.tcpa*dv2

# Check for horizontal conflict
        R2        = self.R*self.R
        self.swhorconf = dcpa2<R2 # conflict or not
        
# Calculate times of entering and leaving horizontal conflict        
        dxinhor   = np.sqrt(np.maximum(0.,R2-dcpa2)) # half the distance travelled inzide zone
        dtinhor   = dxinhor/vrel

        tinhor    = np.where(self.swhorconf,self.tcpa - dtinhor,1e8) # Set very large if no conf
        
        touthor   = np.where(self.swhorconf,self.tcpa + dtinhor,-1e8) # set very large if no conf
#        swhorconf = swhorconf*(touthor>0)*(tinhor<self.dtlook)
       
# Vertical conflict -----------------------------------------------------------

# Vertical crossing of disk (-dh,+dh)

        alt       = self.traf.alt.reshape((1,self.traf.ntraf))
        adsbalt   = self.traf.adsbalt.reshape((1,self.traf.ntraf))
        if self.traf.ADSBtransnoise:
            # error in the determined altitude of other a/c
            alterror=np.random.normal(0,self.traf.transerror[2],adsbalt.shape) #degrees
            adsbalt+=alterror        
        
        self.dalt      = alt - adsbalt.T

        vs  = self.traf.vs
        vs  = vs.reshape(1,len(vs))

        avs = self.traf.adsbvs
        avs = avs.reshape(1,len(avs))

        dvs = vs-avs.T

# Check for passing through each others zone       
        dvs       = np.where(np.abs(dvs)<1e-6,1e-6,dvs) # prevent division by zero
        tcrosshi  = (self.dalt + self.dh)/-dvs
        tcrosslo  = (self.dalt - self.dh)/-dvs
        
        tinver    = np.minimum(tcrosshi,tcrosslo)
        toutver   = np.maximum(tcrosshi,tcrosslo)
        
# Combine vertical and horizontal conflict-------------------------------------
        self.tinconf = np.maximum(tinver,tinhor)
        
        self.toutconf = np.minimum(toutver,touthor)
        
        self.swconfl = self.swhorconf*(self.tinconf<=self.toutconf)*    \
                       (self.toutconf>0.)*(self.tinconf<self.dtlookahead) \
                       *(1.-I)
                           
        return
    
    # ==================== Conflict Filter (User specific) ======================
    def conflictfilter(self):
        if not self.swasas:
            return
        ## Filter for conflicts: no conflicts are detected for aircraft with an altitude less than 1000 ft
        idx_conffilter                 = np.where(self.traf.alt<(1000*ft))[0]  # Search for the indices of aircraft within the altitude restriction
        self.swconfl[idx_conffilter,:] = 0.                                    # Make the rows of the 'restricted' aircraft zero
        self.swconfl[:,idx_conffilter] = 0.                                    # Make the columns of the 'restricted' aircraft zero
        
        return

    def conflictlist(self, simt):
        if len(self.swconfl) == 0:
            return
        # Calculate CPA positions of traffic in lat/lon?

        # Select conflicting pairs: each a/c gets their own record
        self.confidxs = np.where(self.swconfl)

        self.nconf = len(self.confidxs[0])
        self.iown = self.confidxs[0]
        self.ioth = self.confidxs[1]

        self.latowncpa = []
        self.lonowncpa = []
        self.altowncpa = []
        self.latintcpa = []
        self.lonintcpa = []
        self.altintcpa = []

# Store result
        self.traf.iconf = self.traf.ntraf*[-1]
        self.idown = []
        self.idoth = []
        
        self.LOSlist_now=[]
        self.conflist_now=[]
        

        for idx in range(self.nconf):
            i = self.iown[idx]
            j = self.ioth[idx]
            if i==j:
                continue

            self.idown.append(self.traf.id[i])
            self.idoth.append(self.traf.id[j])
            
            self.traf.iconf[i] = idx
            rng = self.tcpa[i,j]*self.traf.gs[i]/nm
            lato,lono = qdrpos(self.traf.lat[i],self.traf.lon[i], \
                                                        self.traf.trk[i],rng)
            alto = self.traf.alt[i]+self.tcpa[i,j]*self.traf.vs[i]
                                            
            rng = self.tcpa[i,j]*self.traf.adsbgs[j]/nm
            lati,loni = qdrpos(self.traf.adsblat[j],self.traf.adsblon[j], \
                                                       self.traf.adsbtrk[j],rng)
            alti=self.traf.adsbalt[j]+self.tcpa[i,j]*self.traf.adsbvs[j]
            
            self.latowncpa.append(lato)
            self.lonowncpa.append(lono)
            self.altowncpa.append(alto)
            self.latintcpa.append(lati)
            self.lonintcpa.append(loni)
            self.altintcpa.append(alti)
            
            dx = (self.traf.lat[i]-self.traf.lat[j])*111319.
            dy = (self.traf.lon[i]-self.traf.lon[j])*111319.
            
            hdist2 = dx**2+dy**2
            hLOS  = hdist2<self.R**2
            vdist = abs(self.traf.alt[i]-self.traf.alt[j])
            vLOS  = vdist<self.dh
            
            LOS = self.checkLOS(hLOS,vLOS,i,j)
            
# Add to Conflict and LOSlist, to count total conflicts and LOS

            # NB: if only one A/C detects a conflict, it is also added to these lists
            combi=str(self.traf.id[i])+" "+str(self.traf.id[j])
            combi2=str(self.traf.id[j])+" "+str(self.traf.id[i])
            
            experimenttime = simt>2100 and simt<5700 # These parameters may be 
            #changed to count only conflicts within a given expirement time window
            
            if combi not in self.conflist_all and combi2 not in self.conflist_all:
                self.conflist_all.append(combi)
                if self.traf.log.swcfl:
                    self.traf.log.write(1,simt,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
                                        % (self.traf.id[i],self.traf.id[j],self.tcpa[i][j],self.tinconf[i][j],self.toutconf[i][j], \
                                           self.latowncpa[idx],self.lonowncpa[idx],self.altowncpa[idx],self.traf.inconflict[i],\
                                           self.latintcpa[idx],self.lonintcpa[idx],self.altintcpa[idx],self.traf.inconflict[j],\
                                           self.traf.lat[i],self.traf.lon[i],self.traf.trk[i],self.traf.alt[i], \
                                           self.traf.tas[i],self.traf.gs[i],self.traf.vs[i],self.traf.type[i], \
                                           self.traf.lat[j],self.traf.lon[j],self.traf.trk[j],self.traf.alt[j], \
                                           self.traf.tas[j],self.traf.gs[j],self.traf.vs[j],self.traf.type[j]))
            if combi not in self.conflist_exp and combi2 not in self.conflist_exp and experimenttime:
                self.conflist_exp.append(combi)

            if combi not in self.conflist_now and combi2 not in self.conflist_now:
                self.conflist_now.append(combi)

            if LOS:
                if combi not in self.LOSlist_all and combi2 not in self.LOSlist_all:
                    self.LOSlist_all.append(combi)
                    self.LOSmaxsev.append(0.)
                    self.LOShmaxsev.append(0.)
                    self.LOSvmaxsev.append(0.)
                    if self.traf.log.swint:
                        self.traf.log.write(2,simt,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
                                            % (self.traf.id[i],self.traf.id[j], \
                                               self.tinconf[i][j],self.toutconf[i][j], \
                                               self.traf.lat[i],self.traf.lon[i],self.traf.trk[i],self.traf.alt[i], \
                                               self.traf.tas[i],self.traf.gs[i],self.traf.vs[i],self.traf.type[i], \
                                               self.traf.lat[j],self.traf.lon[j],self.traf.trk[j],self.traf.alt[j], \
                                               self.traf.tas[j],self.traf.gs[j],self.traf.vs[j],self.traf.type[j]))
            
                if combi not in self.LOSlist_exp and combi2 not in self.LOSlist_exp and experimenttime:
                    self.LOSlist_exp.append(combi)

                if combi not in self.LOSlist_now and combi2 not in self.LOSlist_now:
                    self.LOSlist_now.append(combi)
                    
                #Now, we measure intrusion and store it if it is the most severe
                Ih = 1.0 - np.sqrt(hdist2)/self.R
                Iv = 1.0 - vdist/self.dh
                severity = min(Ih,Iv)

                try:  # Only continue if combi is found in LOSlist (and not combi2)
                    idx = self.LOSlist_all.index(combi)
                except:
                    idx = -1
                                   
                if idx >=0:
                    if severity > self.LOSmaxsev[idx]:
                        self.LOSmaxsev[idx]  = severity
                        self.LOShmaxsev[idx] = Ih
                        self.LOSvmaxsev[idx] = Iv

        self.nconf = len(self.idown)

# Convert to numpy arrays for vectorisation
        self.latowncpa = np.array(self.latowncpa)
        self.lonowncpa = np.array(self.lonowncpa)
        self.altowncpa = np.array(self.altowncpa)
        self.latintcpa = np.array(self.latintcpa)
        self.lonintcpa = np.array(self.latintcpa)
        self.altintcpa = np.array(self.altintcpa)
  
        return

# ================ Conflict Resolution ========================================

    # Eby method for conflict resolution        
    def resolve(self):
        if self.swasas:
            self.CRmethod.resolve(self)

#============================= Trajectory Recovery ============================
        
# Decide for each aircraft whether the ASAS should be followed or not
    def APorASAS(self):

# Only use when ASAS is on
        if not self.swasas:
            return
            
# Indicate for all A/C that they should follow their Autopilot
        self.traf.asasactive.fill(False) 
        self.traf.inconflict.fill(False)

# Look at all conflicts, also the ones that are solved but CPA is yet to come
        for conflict in self.conflist_all: 
            id1,id2 = self.ConflictToIndices(conflict)
            
            if id1 != "Fail" and id2 != "Fail":
                pastCPA=self.ConflictIsPastCPA(self.traf,id1,id2)
                
                if not pastCPA:
                    # Indicate that the A/C must follow their ASAS
                    self.traf.asasactive[id1] = True 
                    self.traf.inconflict[id1] = True

                    self.traf.asasactive[id2] = True
                    self.traf.inconflict[id2] = True                   
                    
                    # if the next waypoint is the destination airport, and the 
                    # distance to the destination airport is less than dist2vs, don't detect conflicts anymore. 
                    # Do this only if the lnav is on
                    # This is to ensure that last minute conflicts don't deviate aircraft from their destinations too much.
                    if self.traf.swlnav[id1] == True:
                        dy = (self.traf.actwplat[id1]-self.traf.lat[id1])
                        dx = (self.traf.actwplon[id1]-self.traf.lon[id1])*self.traf.coslat[id1]
                        dist2wp = 60.*nm*np.sqrt(dx*dx+dy*dy)
                        if self.traf.route[id1].wptype[self.traf.route[id1].iactwp] == 3 and dist2wp<self.traf.dist2vs[id1]:
                            self.traf.asasactive[id1] = False    
                            iwpid1 = self.traf.route[id1].findact2(self.traf,id1)                        

                    # same as above for id2   
                    if self.traf.swlnav[id2] == True:                     
                        dy = (self.traf.actwplat[id2]-self.traf.lat[id2])
                        dx = (self.traf.actwplon[id2]-self.traf.lon[id2])*self.traf.coslat[id2]
                        dist2wp = 60.*nm*np.sqrt(dx*dx+dy*dy)                        
                        if self.traf.route[id2].wptype[self.traf.route[id2].iactwp] == 3 and dist2wp<self.traf.dist2vs[id2]:
                            self.traf.asasactive[id2] = False
                            iwpid2 = self.traf.route[id2].findact2(self.traf,id2)                        
                        
                else:
                    # Find the next active waypoint and delete the conflict from conflist_all
                    iwpid1 = self.traf.route[id1].findact2(self.traf,id1)
                    if iwpid1 != -1: # To avoid problems if there are no waypoints
                        self.traf.route[id1].direct(self.traf, id1, self.traf.route[id1].wpname[iwpid1])
                    iwpid2 = self.traf.route[id2].findact2(self.traf,id2)
                    if iwpid2 != -1: # To avoid problems if there are no waypoints
                        self.traf.route[id2].direct(self.traf, id2, self.traf.route[id2].wpname[iwpid2])
                    
                    # if conflict is solved, remove it from the conflist_all to pre
                    self.conflist_all.remove(conflict)
                    
            elif id1 == "Fail" and id2!= "Fail":
                 iwpid2 = self.traf.route[id2].findact2(self.traf,id2)
                 if iwpid2 != -1: # To avoid problems if there are no waypoints
                     self.traf.route[id2].direct(self.traf, id2, self.traf.route[id2].wpname[iwpid2])
                 self.conflist_all.remove(conflict)
            
            elif id2 == "Fail" and id1 != "Fail":
                iwpid1 = self.traf.route[id1].findact2(self.traf,id1)
                if iwpid1 != -1: # To avoid problems if there are no waypoints
                    self.traf.route[id1].direct(self.traf, id1, self.traf.route[id1].wpname[iwpid1])
                self.conflist_all.remove(conflict)
            
            # if both are fail, then remove the conflict from the conflist_all
            else:
                self.conflist_all.remove(conflict)
                     
                
            
        return

#========================= Check if past CPA ==================================
        
    def ConflictIsPastCPA(self, traf, id1, id2):

        d = np.array([traf.lon[id2]-traf.lon[id1],traf.lat[id2]-traf.lat[id1],traf.alt[id2]-traf.alt[id1]])

        # find track in degrees
        t1 = np.radians(traf.trk[id1])
        t2 = np.radians(traf.trk[id2])
        
        # write velocities as vectors and find relative velocity vector              
        v1 = np.array([np.sin(t1)*traf.tas[id1],np.cos(t1)*traf.tas[id1],traf.vs[id1]])
        v2 = np.array([np.sin(t2)*traf.tas[id2],np.cos(t2)*traf.tas[id2],traf.vs[id2]])
        v  = np.array(v2-v1) 
        
        # the conflict has past CPA if the horizontal
        # velocities of the two aircraft are not pointing at each other
        pastCPA = np.dot(d[:2],v[:2])>0.0

        return pastCPA

#====================== Give A/C indices of conflict pair =====================
        
    def ConflictToIndices(self,conflict):
        ac1,ac2 = conflict.split(" ")

        try:
            id1=self.traf.id.index(ac1)
            id2=self.traf.id.index(ac2)
        except:
            return "Fail","Fail"

        return id1,id2

#========== Method for cleaning up aircraft outside test region ===============
    def cleanup(self,traf):
        pass
    
    def checkLOS(self,HLOS,VLOS,i,j):
        return HLOS & VLOS
