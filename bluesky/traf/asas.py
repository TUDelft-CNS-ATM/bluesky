import numpy as np
import time
from ..tools.aero_np import qdrdist_vector,nm,qdrpos,vtas2eas
#from tools import kwikqdrdist

class Dbconf:
    """
    Constructor of conflict database, call with SI units (meters and seconds)
    Inputs:
       lat [deg]  = array with traffic latitude
       lon [deg]  = array with traffic longitude
       alt [m]    = array with traffic altitude 
       trk [deg]  = array with traffic track angle
       gs [m/s]   = array with ground speed [m/s]
       vs [m/s]   = array with vertical speed [m/s]

    Outputs:
       swconfl = 2D array with True/False for conflict
       dtconfl = time to conflict
    """

    def __init__(self,traf,tlook, R, dh):
        self.swasas = False
        self.dtlookahead = tlook  # [s] lookahead time
        self.R           = R      # [m] Horizontal separation minimum
        self.dh          = dh     # [m] Vertical separation minimum
        self.traf        = traf   # Traffic database object
        self.reset()              # Reset database
        return

    def reset(self):
        """Reset conflict database"""
        self.conf        = []     # Start with emtpy database: no conflicts
        self.nconf       = 0      # Number of detected conflicts

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
        self.conflist_now= [] #Create a list of current Conflicts       
        self.LOSlist_now = [] #Create a list of current Losses Of Separation
        return


    def cd_state(self):
        """Conflict Detection based on state"""

        # t0_ = time.clock()     # Timing of ASAS calculation

        # ---------- Horizontal conflict -----------
        # qdlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
        qdlst = qdrdist_vector(np.mat(self.traf.lat),np.mat(self.traf.lon),\
                                  np.mat(self.traf.adsblat),np.mat(self.traf.adsblon))

        # Convert results from mat-> array
        self.qdr = np.array(qdlst[0])   # degrees
        I = np.eye(self.traf.ntraf)     # Identity matric of order ntraf
        self.dist = np.array(qdlst[1])*nm + 1e9*I   # meters i to j
                    
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
        dx      = self.dist * np.sin(qdrrad) # is pos j rel to i
        dy      = self.dist * np.cos(qdrrad) # is pos j rel to i
        
        trkrad = np.radians(self.traf.trk)
        u      = self.traf.gs*np.sin(trkrad).reshape((1,len(trkrad)))  # m/s
        v      = self.traf.gs*np.cos(trkrad).reshape((1,len(trkrad)))  # m/s
        
        # parameters received through ADSB
        adsbtrkrad = np.radians(self.traf.adsbtrk)
        adsbu  = self.traf.adsbgs*np.sin(adsbtrkrad).reshape((1,len(adsbtrkrad)))  # m/s
        adsbv  = self.traf.adsbgs*np.cos(adsbtrkrad).reshape((1,len(adsbtrkrad)))  # m/s
        
        du = u - adsbu.T  # Speed du[i,j] is perceived eastern speed of i to j
        dv = v - adsbv.T  # Speed dv[i,j] is perceived northern speed of i to j
        
        dv2  = du*du+dv*dv
        dv2  = np.where(np.abs(dv2)<1e-6,1e-6,dv2) # limit lower absolute value
        
        vrel = np.sqrt(dv2)
        
        tcpa = -(du*dx + dv*dy) / dv2   + 1e9*I

        # Calculate CPA positions
        xcpa = tcpa*du
        ycpa = tcpa*dv

        # Calculate distance^2 at CPA (minimum distance^2)
        dcpa2 = self.dist*self.dist-tcpa*tcpa*dv2

        # Check for horizontal conflict
        R2        = self.R*self.R
        swhorconf = dcpa2<R2 # conflict or not

        # Calculate times of entering and leaving horizontal conflict        
        dxinhor   = np.sqrt(np.maximum(0.,R2-dcpa2)) # half the distance travelled inzide zone
        dtinhor   = dxinhor/vrel

        tinhor    = np.where(swhorconf,tcpa - dtinhor,1e8) # Set very large if no conf
        
        touthor   = np.where(swhorconf,tcpa + dtinhor,-1e8) # set very large if no conf

        # swhorconf = swhorconf*(touthor>0)*(tinhor<self.dtlook)
       
        # ----------- Vertical conflict ------------

        # Vertical crossing of disk (-dh,+dh)
        alt       = self.traf.alt.reshape((1,self.traf.ntraf))
        adsbalt   = self.traf.adsbalt.reshape((1,self.traf.ntraf))
        if self.traf.ADSBtransnoise:
            # error in the determined altitude of other a/c
            alterror=np.random.normal(0,self.traf.transerror[2],adsbalt.shape) #degrees
            adsbalt+=alterror        
        
        dalt = alt - adsbalt.T
        vs = self.traf.vs
        avs = self.traf.adsbvs
        vs = vs.reshape(1,len(vs))
        avs = avs.reshape(1,len(avs))
        dvs = vs-avs.T

        # Check for passing through each others zone       
        dvs       = np.where(np.abs(dvs)<1e-6,1e-6,dvs) # prevent division by zero
        tcrosshi  = (dalt + self.dh)/-dvs
        tcrosslo  = (dalt - self.dh)/-dvs
        
        tinver    = np.minimum(tcrosshi,tcrosslo)
        toutver   = np.maximum(tcrosshi,tcrosslo)
   
        # Check for level conflicts
        # levelconf = (np.abs(dvs)<1.)*(dalt<self.dh) # low V/S AND within zone dh
        # tinver    = np.where(levelconf,-10.,tinver) # Level conf: t in very small
        # toutver   = np.where(levelconf,1e6,toutver) # Level conf: t out very large
        

        # --------- Combine vertical and horizontal conflict ----------

        self.tinconf = np.maximum(tinver,tinhor)        
        self.toutconf = np.minimum(toutver,touthor)
        self.swconfl = swhorconf*(self.tinconf<=self.toutconf)*    \
           (self.toutconf>0.)*(self.tinconf<self.dtlookahead) \
           *(1.-I)
                       
        # Calculate CPA positions of traffic in lat/lon?
        # Select conflicting pairs: each a/c gets their own record
        self.confidxs = np.where(self.swconfl)
        self.nconf = len(self.confidxs[0])
        iown = self.confidxs[0]
        ioth = self.confidxs[1]
        
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
            i = iown[idx]
            j = ioth[idx]
            if i==j:
                continue

            self.idown.append(self.traf.id[i])
            self.idoth.append(self.traf.id[j])
            
            self.traf.iconf[i] = idx
            rng = tcpa[i,j]*self.traf.gs[i]/nm
            lato,lono = qdrpos(self.traf.lat[i],self.traf.lon[i], \
                                                        self.traf.trk[i],rng)
            alto=self.traf.alt[i]+tcpa[i,j]*self.traf.vs[i]
                                            
            rng = tcpa[i,j]*self.traf.adsbgs[j]/nm
            lati,loni = qdrpos(self.traf.adsblat[j],self.traf.adsblon[j], \
                                                       self.traf.adsbtrk[j],rng)
            alti=self.traf.adsbalt[j]+tcpa[i,j]*self.traf.adsbvs[j]
                                                       
            self.latowncpa.append(lato)
            self.lonowncpa.append(lono)
            self.altowncpa.append(alto)
            self.latintcpa.append(lati)
            self.lonintcpa.append(loni)
            self.altintcpa.append(alti)
            
            # Add to Conflict and LOSlist, to count total conflicts and LOS
            # NB: if only one A/C detects a conflict, it is also added to these lists
            combi=str(self.traf.id[i])+" "+str(self.traf.id[j])
            combi2=str(self.traf.id[j])+" "+str(self.traf.id[i])
            
            if combi not in self.conflist_all and combi2 not in self.conflist_all:
                self.conflist_all.append(combi)
            if combi not in self.conflist_now and combi2 not in self.conflist_now:
                self.conflist_now.append(combi)
            if self.tinconf[i,j]<0:
                if combi not in self.LOSlist_all and combi2 not in self.LOSlist_all:
                    self.LOSlist_all.append(combi)
                if combi not in self.LOSlist_now and combi2 not in self.LOSlist_now:
                    self.LOSlist_now.append(combi)
                    
        self.nconf = len(self.idown)

        # Convert to numpy arrays for vectorisation
        self.latowncpa = np.array(self.latowncpa)
        self.lonowncpa = np.array(self.lonowncpa)
        self.altowncpa = np.array(self.altowncpa)
        self.latintcpa = np.array(self.latintcpa)
        self.lonintcpa = np.array(self.latintcpa)
        self.altintcpa = np.array(self.altintcpa)
        return
    
    def cr_eby(self): 
        """Conflict Resolution with Eby's method"""

        # required change in velocity
        dv=np.zeros((self.traf.ntraf,3)) 
        
        #if possible, solve conflicts once and copy results for symmetrical conflicts,
        #if that is not possible, solve each conflict twice, once for each A/C
        if not self.traf.ADSBtrunc and not self.traf.ADSBtransnoise:
            for conflict in self.conflist_now:
                id1,id2=self.ConflictToIndices(conflict)
                if id1 != "Fail" and id2!= "Fail":
                    dv_eby=self.Eby_straight(self.traf,id1,id2,self.qdr[id1,id2],self.dist[id1,id2])
                    dv[id1]-=dv_eby
                    dv[id2]+=dv_eby
        else:
            for i in range(self.nconf):
                ac1=self.idown[i]
                ac2=self.idoth[i]
                id1=self.traf.id.index(ac1)
                id2=self.traf.id.index(ac2)
                dv_eby=self.Eby_straight(self.traf,id1,id2,self.qdr[id1,id2],self.dist[id1,id2])
                dv[id1]-=dv_eby

        # now we have the change in speed vector for each aircraft.
        dv=np.transpose(dv)
        # the old speed vector, cartesian coordinates
        trkrad=np.radians(self.traf.trk)
        v=np.array([np.sin(trkrad)*self.traf.tas,\
                    np.cos(trkrad)*self.traf.tas,\
                    self.traf.vs])
        # the new speed vector
        newv=dv+v
        # the new speed vector in polar coordinates
        newtrack=(np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
        newgs=np.sqrt(newv[0,:]**2+newv[1,:]**2)
        neweas=vtas2eas(newgs,self.traf.alt)
        
        # now assign in the traf class
        self.traf.asashdg=newtrack
        self.traf.asasspd=neweas
        self.traf.asasvsp=newv[2,:]


    #============================= ============================
        
    def APorASAS(self):
        """Trajectory Recovery. Decide for each aircraft whether 
        the ASAS should be followed or not"""
        
        # Indicate for all A/C that they should follow their Autopilot
        self.traf.asasactive.fill(False) 

        # Look at all conflicts, also the ones that are solved but CPA is yet to come
        for conflict in self.conflist_all: 
            id1,id2=self.ConflictToIndices(conflict)
            if id1 != "Fail":
                pastCPA=self.ConflictIsPastCPA(self.traf,id1,id2)
                if not pastCPA:
                    # Indicate that the A/C must follow their ASAS
                    self.traf.asasactive[id1]=True 
                    self.traf.asasactive[id2]=True        


    def Eby_straight(self, traf, id1, id2, qdr, dist):
        """
        Resolution: Eby method assuming aircraft move straight forward, 
        solving algebraically, only horizontally

        intrusion vector:
        i(t)=self.hsep-d(t)
        d(t)=sqrt((d[0]+v[0]*t)**2+(d[1]+v[1]*t)**2)
        find max(i(t)/t)
        - write the equation out
        - take derivative, set to zero
        - simplify, take square of everything so the sqrt disappears (creates two solutions)
        - write to the form a*t**2 + b*t + c = 0
        - Solve using the quadratic formula
        """
        
        # from degrees to radians
        qdr=np.radians(qdr)
        # relative position vector
        d=np.array([np.sin(qdr)*dist, \
                   np.cos(qdr)*dist, \
                   traf.alt[id2]-traf.alt[id1] ])

        # find track in degrees
        t1=np.radians(traf.trk[id1])
        t2=np.radians(traf.trk[id2])
        
        # write velocities as vectors and find relative velocity vector              
        v1=np.array([np.sin(t1)*traf.tas[id1],np.cos(t1)*traf.tas[id1],traf.vs[id1]])
        v2=np.array([np.sin(t2)*traf.tas[id2],np.cos(t2)*traf.tas[id2],traf.vs[id2]])
        v=np.array(v2-v1) 

        # bear in mind: the definition of vr (relative velocity) is opposite to 
        # the velocity vector in the LOS_nominal method, this just has consequences
        # for the derivation of tstar following Eby method, not more

        # These terms are used to construct a,b,c of the quadratic formula
        R2=self.R**2 # in meters
        d2=np.dot(d,d) # distance vector length squared
        v2=np.dot(v,v) # velocity vector length squared
        dv=np.dot(d,v) # dot product of distance and velocity
        
        # Solving the quadratic formula
        a=R2*v2 - dv**2
        b=2*dv* (R2 - d2)
        c=R2*d2 - d2**2
        discrim=b**2 - 4*a*c
        
        if discrim<0: # if the discriminant is negative, we're done as taking the square root will result in an error
            discrim=0

        time1=(-b+np.sqrt(discrim))/(2*a)
        time2=(-b-np.sqrt(discrim))/(2*a)

        #time when the size of the conflict is largest relative to time to solve
        tstar=min(time1,time2)
        
        #find drel and absolute distance at tstar
        drelstar=d+v*tstar
        dstarabs=np.linalg.norm(drelstar)
        #exception: if the two aircraft are on exact collision course 
        #(passing eachother within 10 meter), change drelstar
        exactcourse=10 #10 meter
        dif=exactcourse-dstarabs
        if dif>0:
            vperp=np.array([-v[1],v[0],0]) #rotate velocity 90 degrees in horizontal plane
            drelstar+=dif*vperp/np.linalg.norm(vperp) #normalize to 10 m and add to drelstar
            dstarabs=np.linalg.norm(drelstar)
        
        #intrusion at tstar
        i=self.R-dstarabs
        #desired change in the plane's speed vector:
        dv=i*drelstar/(dstarabs*tstar)
        return dv
        
    def ConflictIsPastCPA(self, traf, id1, id2):
        """Check if past CPA"""
        d=np.array([traf.lon[id2]-traf.lon[id1],traf.lat[id2]-traf.lat[id1],traf.alt[id2]-traf.alt[id1]])

        # find track in degrees
        t1=np.radians(traf.trk[id1])
        t2=np.radians(traf.trk[id2])
        
        # write velocities as vectors and find relative velocity vector              
        v1=np.array([np.sin(t1)*traf.tas[id1],np.cos(t1)*traf.tas[id1],traf.vs[id1]])
        v2=np.array([np.sin(t2)*traf.tas[id2],np.cos(t2)*traf.tas[id2],traf.vs[id2]])
        v=np.array(v2-v1) 
        
        pastCPA=np.dot(d,v)>0
           
        return pastCPA
        
    def ConflictToIndices(self,conflict):
        """Give A/C indices of conflict pair"""
        ac1,ac2 = conflict.split(" ")
        try:
            id1=self.traf.id.index(ac1)
            id2=self.traf.id.index(ac2)
        except:
            return "Fail","Fail"
        return id1,id2