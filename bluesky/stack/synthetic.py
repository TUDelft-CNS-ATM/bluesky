import sys
sys.path.append('bluesky/tools/')
import random
import numpy as np
from aero import ft, eas2tas, qdrpos
from misc import txt2alt, txt2spd
'''
    Original version by Jerom Maas, fall 2014
    Adapted and expanded by Pieter Danneels, fall 2015
    
'''

savescenarios=False #whether to save a scenario as .scn file after generation via commands

def process(command, numargs, cmdargs, sim, traf, scr, cmd):
    # First, find by which callsign the CFreeFlight module is called in Cstack.py
    for sign,filename in cmd.extracmdmodules.iteritems():
        if filename==__name__:
            callsign = sign
    
    # change display settings and delete AC to generate own FF scenarios
    if command == "START":
        scr.swgeo=False         #don't draw coastlines and borders
        scr.swsat=False         #don't draw the satellite image
        scr.apsw=0              #don't draw airports
        scr.swlabel=0           #don't draw aircraft labels
        scr.wpsw=0              #don't draw waypoints
        scr.swfir=False         #don't show FIRs
        scr.swgrid=True         #do show a grid
        scr.pan([0,0],True)     #focus the map at the prime meridian and equator
        scr.redrawradbg=True    #draw the background again
        scr.swsep = True        #show circles of seperation between ac
        scr.swspd = True        #show speed vectors of aircraft
        #scr.zoom(1)       #set zoom level to the standard distance
        #cmd.scenlines=[]       #skip the rest of the scenario
        #cmd.scenlines.append("00:00:00.00>"+callsign+"TESTCIRCLE")
        #cmd.scenlines.append("00:00:00.00>DT 1")
        #cmd.scenlines.append("00:00:00.00>FIXDT ON")
        
        sim.mode=sim.init
    # display help    
    elif command == "HELP":
        scr.echo("This is the synthetic traffic scenario module")
        scr.echo("Possible subcommands: HELP, SIMPLE, SIMPLED, DIFG, SUPER, SPHERE, MATRIX, FLOOR, TAKEOVER, WALL, ROW, COLUMN, DISP")
    
    #create a perpendicular conflict between two aircraft
    elif command == "SIMPLE":
        scr.isoalt=0
        traf.deleteall()
        traf.create(["OWN", "GENERIC", -.5, 0, 0, 5000, 200])
        traf.create(["OTH", "GENERIC", 0, .5, 270, 5000, 200])
        
    #create a perpendicular conflict with slight deviations to aircraft speeds and places
    elif command == "SIMPLED":
        scr.isoalt=0
        traf.deleteall()
        ds=random.uniform(0.92,1.08)
        dd=random.uniform(0.92,1.08)
        traf.create(["OWNSHIP", "GENERIC", -.5*dd, 0, 0, 20000, 200*ds])
        traf.create(["INTRUDER", "GENERIC", 0, .5/dd, 270, 20000, 200/ds])   
     
    
    # used for testing the differential game resolution method
    elif command == "DIFG":
        if numargs<5:
            scr.echo("5 ARGUMENTS REQUIRED")
        else:
            scr.isoalt=0
            traf.deleteall()
            x=  traf.dbconf.xw[int(float(cmdargs[1]))]/111319.
            y=  traf.dbconf.yw[int(float(cmdargs[2]))]/111319.
            v_o=traf.dbconf.v_o[int(float(cmdargs[3]))]
            v_w=traf.dbconf.v_w[int(float(cmdargs[4]))]
            phi=np.degrees(traf.dbconf.phi[int(float(cmdargs[5]))])
            traf.create(["OWN", "GENERIC", 0, 0, 0, 5000, v_o])
            traf.create(["WRN", "GENERIC", y, x, phi, 5000, v_w])
    
    
    # create a superconflict of x aircraft in a circle towards the center
    elif command == "SUPER":
        if numargs ==0:
            scr.echo(callsign+"SUPER <NUMBER OF A/C>")
        else:
            scr.isoalt=0
            traf.deleteall()
            numac=int(float(cmdargs[1]))
            distance=0.50 #this is in degrees lat/lon, for now
            alt=20000 #meters
            spd=200 #kts
            for i in range(numac):
                angle=2*np.pi/numac*i
                acid="SUP"+str(i)
                traf.create([acid,"SUPER",distance*-np.cos(angle),distance*np.sin(angle),360-360/numac*i,alt,spd])
            if savescenarios:
                fname="super"+str(numac)
                cmd.saveic(fname,sim,traf)

    
    # create a sphereconflict of 3 layers of superconflicts
    elif command == "SPHERE":
        if numargs ==0:
            scr.echo(callsign+"SPHERE <NUMBER OF A/C PER LAYER>")
        else:
            scr.isoalt=1./200
            traf.deleteall()
            numac=int(float(cmdargs[1]))
            distance=0.5 #this is in degrees lat/lon, for now
            distancenm=distance*111319./1852
            alt=20000 #meters
            spd=150 #kts
            vs=4 #m/s          
            timetoimpact=distancenm/spd*3600 #seconds
            altdifference=vs*timetoimpact # m
            midalt=alt
            lowalt=alt-altdifference
            highalt=alt+altdifference
            hispd=eas2tas(spd,highalt)
            mispd=eas2tas(spd,midalt)
            lospd=eas2tas(spd,lowalt)
            hispd=spd
            mispd=spd
            lospd=spd
            for i in range(numac):
                angle=np.pi*(2./numac*i)
                lat=distance*-np.cos(angle)
                lon=distance*np.sin(angle)
                track=np.degrees(-angle)
                
                acidl="SPH"+str(i)+"LOW"
                traf.create([acidl,"SUPER",lat,lon,track,lowalt,lospd])    
                acidm="SPH"+str(i)+"MID"
                traf.create([acidm,"SUPER",lat,lon,track,midalt,mispd])    
                acidh="SPH"+str(i)+"HIG"
                traf.create([acidh,"SUPER",lat,lon,track,highalt,hispd])    
                
                idxl = traf.id.index(acidl)
                idxh = traf.id.index(acidh)
                
                traf.vs[idxl]=vs
                traf.vs[idxh]=-vs                
                
                traf.avs[idxl]=vs
                traf.avs[idxh]=-vs
                
                traf.aalt[idxl]=highalt
                traf.aalt[idxh]=lowalt
                
            if savescenarios:
                fname="sphere"+str(numac)
                cmd.saveic(fname,sim,traf)
             
    # create a conflict with several aircraft flying in a matrix formation    
    elif command == "MATRIX":
        if numargs ==0:
            scr.echo(callsign+"MATRIX <SIZE>")
        else:
            size=int(float(cmdargs[1]))
            scr.isoalt=0
            traf.deleteall()
            mperdeg=111319.
            hsep=traf.dbconf.R # [m] horizontal separation minimum
            hseplat=hsep/mperdeg
            matsep=1.1 #factor of extra space in the matrix
            hseplat=hseplat*matsep
            vel=200 #m/s
            extradist=(vel*1.1)*5*60/mperdeg #degrees latlon flown in 5 minutes
            for i in range(size):
                acidn="NORTH"+str(i)
                traf.create([acidn,"MATRIX",hseplat*(size-1.)/2+extradist,(i-(size-1.)/2)*hseplat,180,20000,vel])
                acids="SOUTH"+str(i)
                traf.create([acids,"MATRIX",-hseplat*(size-1.)/2-extradist,(i-(size-1.)/2)*hseplat,0,20000,vel])
                acide="EAST"+str(i)
                traf.create([acide,"MATRIX",(i-(size-1.)/2)*hseplat,hseplat*(size-1.)/2+extradist,270,20000,vel])
                acidw="WEST"+str(i)
                traf.create([acidw,"MATRIX",(i-(size-1.)/2)*hseplat,-hseplat*(size-1.)/2-extradist,90,20000,vel])
                
            if savescenarios:
                fname="matrix"+str(size)
                cmd.saveic(fname,sim,traf)

    # create a conflict with several aircraft flying in a floor formation    
    elif command == "FLOOR":
        scr.isoalt=1./50
        traf.deleteall()
        mperdeg=111319.
        altdif=3000 # meters
        hsep=traf.dbconf.R # [m] horizontal separation minimum
        floorsep=1.1 #factor of extra spacing in the floor
        hseplat=hsep/mperdeg*floorsep
        traf.create(["OWNSHIP","FLOOR",-1,0,90, 20000+altdif, 200])
        idx = traf.id.index("OWNSHIP")
        traf.avs[idx]=-10
        traf.aalt[idx]=20000-altdif
        for i in range(20):
            acid="OTH"+str(i)
            traf.create([acid,"FLOOR",-1,(i-10)*hseplat,90,20000,200])            
        if savescenarios:
            fname="floor"
            cmd.saveic(fname,sim,traf)            

    # create a conflict with several aircraft overtaking eachother    
    elif command == "TAKEOVER":
        if numargs ==0:
            scr.echo(callsign+"TAKEOVER <NUMBER OF A/C>")
        else:
            numac=int(float(cmdargs[1]))
            scr.isoalt=0
            traf.deleteall()
            mperdeg=111319.
            vsteps=50 #[m/s]
            for v in range(vsteps,vsteps*(numac+1),vsteps): #m/s
                acid="OT"+str(v)
                distancetofly=v*5*60 #m
                degtofly=distancetofly/mperdeg
                traf.create([acid,"OT",0,-degtofly,90,20000,v])
            if savescenarios:
                fname="takeover"+str(numac)
                cmd.saveic(fname,sim,traf)

    # create a conflict with several aircraft flying in a wall formation    
    elif command == "WALL":
        scr.isoalt=0
        traf.deleteall()
        mperdeg=111319.
        distance=0.6 # in degrees lat/lon, for now
        hsep=traf.dbconf.R # [m] horizontal separation minimum
        hseplat=hsep/mperdeg
        wallsep=1.1 #factor of extra space in the wall
        traf.create(["OWNSHIP","WALL",0,-distance,90, 20000, 200])
        for i in range(20):
            acid="OTHER"+str(i)
            traf.create([acid,"WALL",(i-10)*hseplat*wallsep,distance,270,20000,200])
        if savescenarios:
            fname="wall"
            cmd.saveic(fname,sim,traf)
    
    # create a conflict with several aircraft flying in two rows angled towards each other
    elif command == "ROW":
        commandhelp = "SYN_ROW n angle [-r=radius in NM] [-a=alt in ft] [-s=speed EAS in kts] [-t=actype]"
        if numargs == 0:
            scr.echo(commandhelp)
        else:
            try:
                traf.deleteall() # start fresh
                synerror,acalt,acspd,actype,startdistance,ang = angledtraffic.arguments(numargs,cmdargs[1:]) # process arguments
                if synerror:
                    raise Exception()
                    
                mperdeg=111319.
                hsep=traf.dbconf.R # [m] horizontal separation minimum
                hseplat=hsep/mperdeg
                matsep=1.1 #factor of extra space in the formation
                hseplat=hseplat*matsep
                       
                aclat = startdistance * np.cos(np.deg2rad(ang)) #[deg]
                aclon = startdistance * np.sin(np.deg2rad(ang))
                latsep = abs(hseplat*np.cos(np.deg2rad(90-ang))) #[deg]
                lonsep = abs(hseplat*np.sin(np.deg2rad(90-ang)))                
                
                alternate = 1
                for i in range(int(cmdargs[1])): # Create a/c
                    aclat = aclat+i*latsep*alternate
                    aclon = aclon-i*lonsep*alternate
                    traf.create(["ANG"+str(i*2), actype, aclat, aclon, 180+ang, acalt, acspd])
                    traf.create(["ANG"+str(i*2+1), actype, aclat, -aclon, 180-ang, acalt, acspd])
                    alternate = alternate * -1   
                    
                scr.pan([0,0],True)
            except Exception:
                scr.echo('Syntax error: unknown argument flag')
            except:
                scr.echo('Syntax error')
                scr.echo(commandhelp) 
                

    # create a conflict with several aircraft flying in two columns angled towards each other
    elif command == "COLUMN":
        commandhelp = "SYN_COLUMN n angle [-r=radius in NM] [-a=alt in ft] [-s=speed EAS in kts] [-t=actype]"
        if numargs == 0:
            scr.echo(commandhelp)
        else:
            try:
                traf.deleteall() # start fresh
                synerror,acalt,acspd,actype,startdistance,ang = angledtraffic.arguments(numargs,cmdargs[1:]) # process arguments
                if synerror:
                    raise Exception() 

                mperdeg=111319.
                hsep=traf.dbconf.R # [m] horizontal separation minimum
                hseplat=hsep/mperdeg
                matsep=1.1 #factor of extra space in the formation
                hseplat=hseplat*matsep
                       
                aclat = startdistance * np.cos(np.deg2rad(ang)) #[deg]
                aclon = startdistance * np.sin(np.deg2rad(ang))
                latsep = abs(hseplat*np.cos(np.deg2rad(ang))) #[deg]
                lonsep = abs(hseplat*np.sin(np.deg2rad(ang)))

                traf.create(["ANG0", actype, aclat, aclon, 180+ang, acalt, acspd])
                traf.create(["ANG1", actype, aclat, -aclon, 180-ang, acalt, acspd])  
                
                for i in range(1,int(cmdargs[1])): # Create a/c
                    aclat = aclat+latsep
                    aclon = aclon+lonsep
                    traf.create(["ANG"+str(i*2), actype, aclat, aclon, 180+ang, acalt, acspd])
                    traf.create(["ANG"+str(i*2+1), actype, aclat, -aclon, 180-ang, acalt, acspd])  

                scr.pan([0,0],True)
            except Exception:
                scr.echo('Syntax error: unknown argument flag')
            except:
                scr.echo('Syntax error')
                scr.echo(commandhelp)      
                

#    elif command == "TESTCIRCLE":
#        scr.swtestarea = True   #show circles in testing area
#        scr.redrawradbg=True    #draw the background again
#        traf.dbconf=CASAScircle.Dbconf(traf,300., 5.*nm, 1000.*ft)
#                                #change the ASAS system with one that incorporates
#                                #the circular testing area

    # Toggle the display of certain elements in screen
    elif command == "DISP":
        if numargs == 0:
            scr.echo(callsign+"DISP <SEP/SPD/TEST>")
        else:
            sw = cmdargs[1]
            
            #show separation circles between aircraft of 2.5 nm radius
            if sw == "SEP":
                scr.swsep = not scr.swsep
            elif sw == "SPD":
                scr.swspd = not scr.swspd
            elif sw == "TEST":
                scr.swtestarea = not scr.swtestarea
            
            #unknown command
            else:
                scr.echo(callsign+"DISP <SEP/SPD/TEST>")

    
    #give up
    else:
        scr.echo("Unknown command: " + callsign + command)
#    pass

class angledtraffic():
    
    @staticmethod        
    def arguments(numargs, cmdargs): 
        syntaxerror = False        
        # tunables:
        acalt = float(10000) # default
        acspd = float(300) # default
        actype = "B747" # default
        startdistance = 1 # default
        
        ang = float(cmdargs[1])/2    
        
        if numargs>2:   #process optional arguments
            for i in range(2 ,numargs): # loop over arguments (TODO: put arguments in np array)
                if cmdargs[i].upper().startswith("-R"): #radius
                    startdistance = qdrpos(0,0,90,float(cmdargs[i][3:]))[2] #input in nm
                elif cmdargs[i].upper().startswith("-A"): #altitude
                    acalt = txt2alt(cmdargs[i][3:])*ft
                elif cmdargs[i].upper().startswith("-S"): #speed
                    acspd = txt2spd(cmdargs[i][3:],acalt)
                elif cmdargs[i].upper().startswith("-T"): #ac type
                    actype = cmdargs[i][3:].upper()
                else:
                    syntaxerror = True
        return syntaxerror,acalt,acspd,actype,startdistance,ang