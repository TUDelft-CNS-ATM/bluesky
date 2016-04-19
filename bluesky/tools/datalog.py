"""
    Datalog class definition : Data logging class
    
    Methods:
    Datalog(filename)          :  constructor
    
    write(txt)         : add a line to the datalogging buffer
    save()             : save data to file
    
    Created by  : Jacco M. Hoekstra (TU Delft)
    Date        : October 2013
    
    Modifation  : Added the logging funcitonality for SKY/CFL/INT/SNAP/FLST
    By          : M.A.P. Tra
    Date        : February 2016
    
    """
import os
from misc import tim2txt
from time import strftime,gmtime,localtime
from ..tools.aero import ft


#-----------------------------------------------------------------

class Datalog():
    def __init__(self,traf):
        # Create a buffer and save filename
        #self.fname = os.path.dirname(__file__) + "/../../data/output/" \
        #    + strftime("%Y-%m-%d-%H-%M-%S-BlueSky.sky", gmtime())
        
        self.traf = traf
        
        # LOG types:
        self.sky    = 0   # Traffic Data
        self.cfl    = 1   # Conflict Data
        self.int    = 2   # Intrusion Data
        self.snap   = 3   # Snapshot Data
        self.flst   = 4   # Flight Statistics Data
        self.inst   = 5   # Instantaneous number of conflicts (including position data, can be used in post-processing)
        
        self.reset()
        
    def reset(self):
        # Logging options
        self.swsky      = False # Traffic data logging
        self.swcfl      = False # Conflict data logging
        self.swint      = False # Intrusion data logging
        self.swsnap     = False # SNAP data logging
        self.swflst     = False # Flight Statistics logging
        self.swinst     = True # Instantaneous conflicts logging
        self.t0sky      = -999  # Last time SNAP was called
        self.dtsky      = 30.00 # Interval for snap
        self.t0snap     = -999  # Last time SNAP was called
        self.dtsnap     = 30.00 # Interval for snap
        self.t0writelog = -999  # Last time Writelog was called
        self.dtwritelog = 300.00 # Interval fot writing data and clear buffer
        
        # Create a buffer
        self.buffer_sky     = []
        self.buffer_cfl     = []
        self.buffer_int     = []
        self.buffer_snap    = []
        self.buffer_flst    = []
        self.buffer_inst    = []
        
        # Filename
        self.fname          = []
        self.scenfile       = []
    
    def writesettings(self,scenfile,buffertype):
        self.scenfile = scenfile
        # If scenfile includes the .scn extension, remove it
        if self.scenfile.lower().find(".scn") > 0:
            self.scenfile = self.scenfile[:-4]
        # Define fname if empty
        if self.scenfile == '':
            self.fname = os.path.dirname(__file__) + "/../../data/output/" \
            + strftime("%Y-%m-%d-%H-%M-%S-BlueSky", localtime())
        else:
            self.fname = os.path.dirname(__file__) + "/../../data/output/" + strftime(self.scenfile) + strftime("-%Y%m%d%H%M%S", localtime())
        # Create the .sky file and write the simulation settings to the sky log file
        if buffertype == self.sky:
            f = open('%s.%s' % (self.fname, 'sky'),"w" )
            f.write('==============================='+'\n')
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+'\n')
            f.write('Scenario: '+ str(self.scenfile) +'\n')
            f.write('SKY DATA' +'\n')
            f.write('==============================='+'\n')
            f.write('\n')
            f.write('time [hh:mm:ss], simulation time [hh:mm:ss.ss], ntraf [-], nconf [-], nLoS [-]'+'\n')
            f.write('\n')
            f.close()
        
        # Create the .cfl file an write the simulation settings to the cfl file
        elif buffertype == self.cfl:
            f = open('%s.%s' % (self.fname, 'cfl'),"w" )
            f.write('==============================='+'\n')
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+'\n')
            f.write('Scenario: '+ str(self.scenfile) +'\n')
            f.write('CONFLICT DATA' +'\n')
            f.write('==============================='+'\n')
            f.write('\n')
            f.write('time [hh:mm:ss], simulation time [hh:mm:ss:ss], id1 [-],id2 [-],tcpa [s],tinconf [s],toutconf [s],tcpa_lat1 [decimal degrees], tcpa_lon1 [decimal degrees], tcpa_alt1 [decimal degrees], in_conflict1 [-], tcpa_lat2 [decimal degrees], tcpa_lon2 [decimal degrees], tcpa_alt2 [decimal degrees], in_conflict2 [-], lat1 [decimal degrees],lon1 [decimal degrees],trk1 [deg],alt1 [m],tas1 [m/s],gs1 [m/s],vs1 [m/s],type1 [-],lat2 [decimal degrees],lon2 [decimal degrees],trk2 [deg],tas2 [m/s],[m/s],gs2 [m/s],vs2 [m/s],type2 [-]'+'\n')
            f.write('\n')
            f.close()
        
        # Create the .int file and write the simulation settings to the int file
        elif buffertype == self.int:
            f = open('%s.%s' % (self.fname, 'int'),"w" )
            f.write('==============================='+'\n')
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+'\n')
            f.write('Scenario: '+ str(self.scenfile) +'\n')
            f.write('INTRUSION DATA' +'\n')
            f.write('==============================='+'\n')
            f.write('\n')
            f.write('time [hh:mm:ss], simulation time [hh:mm:ss:ss], id1 [-],id2 [-], LOShmaxsev [-], LOSvmaxsev [-], tinconf [s], toutconf [s], lat1 [decimal degrees],lon1 [decimal degrees],trk1 [deg],alt1 [m],tas1 [m/s],gs1 [m/s],vs1 [m/s],type1 [-],lat2 [decimal degrees],lon2 [decimal degrees],trk2 [deg],tas2 [m/s],[m/s],gs2 [m/s],vs2 [m/s],type2 [-]'+'\n')
            f.write('\n')
            f.close()
        
        # Create the .snap file and write the simulation settings to the snap file
        elif buffertype == self.snap:
            f = open('%s.%s' % (self.fname, 'snap'),"w" )
            f.write('==============================='+'\n')
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+'\n')
            f.write('Scenario: '+ str(self.scenfile) +'\n')
            f.write('SNAP DATA' +'\n')
            f.write('==============================='+'\n')
            f.write('\n')
            f.write('time [hh:mm:ss], simulation time [hh:mm:ss.ss], id [-],type [-],lat [decimal degrees],lon [decimal degrees],alt [m],tas [m/s],gs [m/s],vs [m/s],trk [deg]'+'\n')
            f.write('\n')
            f.close()
        
        # Create the .flst file and write the simulation settings to the flst file
        elif buffertype == self.flst:
            f = open('%s.%s' % (self.fname, 'flst'),"w" )
            f.write('==============================='+'\n')
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+'\n')
            f.write('Scenario: '+ str(self.scenfile) +'\n')
            f.write('FLIGHT STATISTICS DATA' +'\n')
            f.write('==============================='+'\n')
            f.write('\n')
            f.write('time [hh:mm:ss], simulation time [hh:mm:ss.ss], id [-],orig [-],dest [-],type [-],distance-2D [m],distance-3D [m],flighttime [s],work [GJ],del-lat [decimal degrees],del-lon [decimal degrees],del-alt [m]'+'\n')
            f.write('\n')
            f.close()
        
        # Create the .inst file an write the simulation settings to the inst file
        elif buffertype == self.inst:
            f = open('%s.%s' % (self.fname, 'inst'),"w" )
            f.write('==============================='+'\n')
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+'\n')
            f.write('Scenario: '+ str(self.scenfile) +'\n')
            f.write('INSTANTANEOUS CONFLICT DATA' +'\n')
            f.write('==============================='+'\n')
            f.write('\n')
            f.write('time, simulation time, id1,id2,tcpa_lat1,tcpa_lon1,tcpa_alt1,trk1,vs1,type1,tcpa_lat2,tcpa_lon2,tcpa_alt2,trk2,vs2,type2'+'\n')
            f.write('\n')
            f.close()
        
        return

    def write(self,buffertype,t,txt):
        # Add text to buffer with timestamp t
        if buffertype == self.sky:
            self.buffer_sky.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+'\n')
        elif buffertype == self.cfl:
            self.buffer_cfl.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+'\n')
        elif buffertype == self.int:
            self.buffer_int.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+'\n')
        elif buffertype == self.snap:
            self.buffer_snap.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+'\n')
        elif buffertype == self.flst:
            self.buffer_flst.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+'\n')
        elif buffertype == self.inst:
            self.buffer_inst.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+'\n')
        return
    
    def save(self,buffertype):
        # Write the results to a text file
        if buffertype == self.sky:
            f = open('%s.%s' % (self.fname, 'sky'),"a" )
            f.writelines(self.buffer_sky)
            f.close()
            self.buffer_sky = []
        elif buffertype == self.cfl:
            f = open('%s.%s' % (self.fname, 'cfl'),"a" )
            f.writelines(self.buffer_cfl)
            f.close()
            self.buffer_cfl = []
        elif buffertype == self.int:
            f = open('%s.%s' % (self.fname, 'int'),"a" )
            f.writelines(self.buffer_int)
            f.close()
            self.buffer_int = []
        elif buffertype == self.snap:
            f = open('%s.%s' % (self.fname, 'snap'),"a" )
            f.writelines(self.buffer_snap)
            f.close()
            self.buffer_snap = []
        elif buffertype == self.flst:
            f = open('%s.%s' % (self.fname, 'flst'),"a" )
            f.writelines(self.buffer_flst)
            f.close()
            self.buffer_flst = []
        elif buffertype == self.inst:
            f = open('%s.%s' % (self.fname, 'inst'),"a" )
            f.writelines(self.buffer_inst)
            f.close()
            self.buffer_inst = []
        return

    def skysave(self,traf,simt):
        if self.swsky:
            if self.t0sky+self.dtsky<simt or simt<self.t0sky:
                self.t0sky = simt
                self.write(0,simt,'%s,%s,%s' \
                               % (self.traf.ntraf,len(self.traf.dbconf.conflist_now),len(self.traf.dbconf.LOSlist_now)))
        return

    def snapsave(self,traf,simt):
        if self.swsnap:
            if self.t0snap+self.dtsnap<simt or simt<self.t0snap:
                self.t0snap = simt
                i = 0
                while (i < self.traf.ntraf):
                    if self.traf.alt[i] > 5000*ft:
                        self.write(3,simt,'%s,%s,%s,%s,%s,%s,%s,%s,%s' % (self.traf.id[i],self.traf.type[i],self.traf.lat[i], \
                                                                              self.traf.lon[i],self.traf.alt[i],self.traf.tas[i], \
                                                                              self.traf.gs[i],self.traf.vs[i],self.traf.trk[i]))
                    i = i + 1
        return

    def clearbuffer(self,simt):
        # Write the saved buffer to the data files and clear the buffer, every dtwritelog seconds
        # -> Used to avoid any memory problems due to an overload of logging data
        if self.t0writelog+self.dtwritelog<simt or simt<self.t0writelog:
            self.t0writelog = simt
            if self.swsky:
                self.save(0)
            if self.swcfl:
                self.save(1)
            if self.swint:
                self.save(2)
            if self.swsnap:
                self.save(3)
            if self.swflst:
                self.save(4)
            if self.swinst:
                self.save(5)
        return


