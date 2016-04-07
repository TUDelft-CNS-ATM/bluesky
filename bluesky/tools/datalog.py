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


#-----------------------------------------------------------------

class Datalog():
    def __init__(self,traf):
        # Create a buffer and save filename
        #self.fname = os.path.dirname(__file__) + "/../../data/output/" \
        #    + strftime("%Y-%m-%d-%H-%M-%S-BlueSky.sky", gmtime())
        
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
        self.swinst     = False # Instantaneous conflicts logging
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
        # Define fname if empty
        if self.scenfile == '':
            self.fname = os.path.dirname(__file__) + "/../../data/output/" \
            + strftime("%Y-%m-%d-%H-%M-%S-BlueSky", localtime())
        else:
            self.fname = os.path.dirname(__file__) + "/../../data/output/" + strftime(self.scenfile)
        # Create the .sky file and write the simulation settings to the sky log file
        if buffertype == self.sky:
            f = open('%s.%s' % (self.fname, 'sky'),"wr" )
            f.write('==============================='+chr(13)+chr(10))
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+chr(13)+chr(10))
            f.write('Scenario: '+ str(self.scenfile) +chr(13)+chr(10))
            f.write('SKY DATA' +chr(13)+chr(10))
            f.write('==============================='+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.write('time, simulation time, ntraf, nconf, nLoS'+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.close()
        
        # Create the .cfl file an write the simulation settings to the cfl file
        elif buffertype == self.cfl:
            f = open('%s.%s' % (self.fname, 'cfl'),"wr" )
            f.write('==============================='+chr(13)+chr(10))
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+chr(13)+chr(10))
            f.write('Scenario: '+ str(self.scenfile) +chr(13)+chr(10))
            f.write('CONFLICT DATA' +chr(13)+chr(10))
            f.write('==============================='+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.write('time, simulation time, id1,id2,tcpa,tinconf,toutconf,lat1,lon1,trk1,alt1,tas1,gs1,vs1,type1,lat2,lon2,trk2,tas2,alt2,vs2,type2'+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.close()
        
        # Create the .int file and write the simulation settings to the int file
        elif buffertype == self.int:
            f = open('%s.%s' % (self.fname, 'int'),"wr" )
            f.write('==============================='+chr(13)+chr(10))
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+chr(13)+chr(10))
            f.write('Scenario: '+ str(self.scenfile) +chr(13)+chr(10))
            f.write('INTRUSION DATA' +chr(13)+chr(10))
            f.write('==============================='+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.write('time, simulation time, id1,id2,tinint,toutint,lat1,lon1,trk1,tas1,alt1,vs1,type1,lat2,lon2,trk2,tas2,alt2,vs2,type2'+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.close()
        
        # Create the .snap file and write the simulation settings to the snap file
        elif buffertype == self.snap:
            f = open('%s.%s' % (self.fname, 'snap'),"wr" )
            f.write('==============================='+chr(13)+chr(10))
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+chr(13)+chr(10))
            f.write('Scenario: '+ str(self.scenfile) +chr(13)+chr(10))
            f.write('SNAP DATA' +chr(13)+chr(10))
            f.write('==============================='+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.write('time, simulation time, id,type,lat,lon,alt,tas,gs,vs,trk'+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.close()
        
        # Create the .flst file and write the simulation settings to the flst file
        elif buffertype == self.flst:
            f = open('%s.%s' % (self.fname, 'flst'),"wr" )
            f.write('==============================='+chr(13)+chr(10))
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+chr(13)+chr(10))
            f.write('Scenario: '+ str(self.scenfile) +chr(13)+chr(10))
            f.write('FLIGHT STATISTICS DATA' +chr(13)+chr(10))
            f.write('==============================='+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.write('time, simulation time, id,orig,dest,type,distance-2D,distance-3D,flighttime,work,del-lat,del-lon,del-alt'+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.close()
        
        # Create the .inst file an write the simulation settings to the inst file
        elif buffertype == self.inst:
            f = open('%s.%s' % (self.fname, 'inst'),"wr" )
            f.write('==============================='+chr(13)+chr(10))
            f.write('New run at: '+strftime("%Y-%m-%d %H:%M:%S", localtime())+chr(13)+chr(10))
            f.write('Scenario: '+ str(self.scenfile) +chr(13)+chr(10))
            f.write('INSTANTANEOUS CONFLICT DATA' +chr(13)+chr(10))
            f.write('==============================='+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.write('time, simulation time, id1,id2,tcpa_lat1,tcpa_lon1,tcpa_alt1,trk1,vs1,type1,tcpa_lat2,tcpa_lon2,tcpa_alt2,trk2,vs2,type2'+chr(13)+chr(10))
            f.write(chr(13)+chr(10))
            f.close()
        
        return

    def write(self,buffertype,t,txt):
        # Add text to buffer with timestamp t
        if buffertype == self.sky:
            self.buffer_sky.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+chr(13)+chr(10))
        elif buffertype == self.cfl:
            self.buffer_cfl.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+chr(13)+chr(10))
        elif buffertype == self.int:
            self.buffer_int.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+chr(13)+chr(10))
        elif buffertype == self.snap:
            self.buffer_snap.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+chr(13)+chr(10))
        elif buffertype == self.flst:
            self.buffer_flst.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+chr(13)+chr(10))
        elif buffertype == self.inst:
            self.buffer_inst.append(strftime("%H:%M:%S", localtime()) + ", " + tim2txt(t)+", "+txt+chr(13)+chr(10))
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

