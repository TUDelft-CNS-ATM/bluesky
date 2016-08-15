# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:55:16 2014

@author: Jerom Maas
"""
import random
import numpy as np
from ..tools.aero import ft, nm, eas2tas

savescenarios=False #whether to save a scenario as .scn file after generation via commands

def process(command, numargs, commandargs, sim, traf, scr, cmd):
    #First, find by which callsign the CFreeFlight module is called in Cstack.py
    for sign,filename in cmd.extracmdmodules.iteritems():
        if filename==__name__:
            callsign = sign
    
    #change display settings and delete AC to generate own FF scenarios
    if command == "START":
        scr.swgeo=False         #don't draw coastlines and borders
        scr.swsat=False         #don't draw the satellite image
        scr.apsw=0              #don't draw airports
        scr.swlabel=0           #don't draw aircraft labels
        scr.wpsw=0              #don't draw waypoints
        scr.swfir=False         #don't show FIRs
        scr.swgrid=True         #do show a grid
        scr.pan(0,0)            #focus the map at the prime meridian and equator
        scr.redrawradbg=True    #draw the background again
        scr.swsep = True        #show circles of seperation between ac
        scr.swspd = True        #show speed vectors of aircraft
        scr.zoom(0.7,True) # zoom in one level
        cmd.scenlines=[]        #skip the rest of the scenario
        cmd.scenlines.append("00:00:00.00>"+callsign+"TESTCIRCLE")
        
        sim.mode=sim.init
    
    #create a perpendicular conflict between two aircraft
    elif command == "SIMPLE":
        scr.isoalt=0
        traf.deleteall()
        traf.create("OWNSHIP", "GENERIC", -.5, 0, 0, 20000, 200)
        traf.create("INTRUDER", "GENERIC", 0, .5, 270, 20000, 200)
        
    #create a perpendicular conflict with slight deviations to aircraft speeds and places
    elif command == "SIMPLED":
        scr.isoalt=0
        traf.deleteall()
        ds=random.uniform(0.92,1.08)
        dd=random.uniform(0.92,1.08)
        traf.create("OWNSHIP", "GENERIC", -.5*dd, 0, 0, 20000, 200*ds)
        traf.create("INTRUDER", "GENERIC", 0, .5/dd, 270, 20000, 200/ds)        
    
    #create a superconflict of x aircraft in a circle towards the center
    elif command == "SUPER":
        if numargs ==0:
            scr.echo(callsign+"SUPER <NUMBER OF A/C>")
        else:
            scr.isoalt=0
            traf.deleteall()
            numac=int(float(commandargs[1]))
            distance=0.50 #this is in degrees lat/lon, for now
            alt=20000 #meters
            spd=200 #kts
            for i in range(numac):
                angle=2*np.pi/numac*i
                acid="SUP"+str(i)
                traf.create(acid,"SUPER",distance*-np.cos(angle),distance*np.sin(angle),360-360/numac*i,alt,spd)
            if savescenarios:
                fname="super"+str(numac)
                cmd.saveic(fname,sim,traf)

    
    #create a sphereconflict of 3 layers of superconflicts
    elif command == "SPHERE":
        if numargs ==0:
            scr.echo(callsign+"SPHERE <NUMBER OF A/C PER LAYER>")
        else:
            scr.isoalt=1./200
            traf.deleteall()
            numac=int(float(commandargs[1]))
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
                traf.create(acidl,"SUPER",lat,lon,track,lowalt,lospd)    
                acidm="SPH"+str(i)+"MID"
                traf.create(acidm,"SUPER",lat,lon,track,midalt,mispd)    
                acidh="SPH"+str(i)+"HIG"
                traf.create(acidh,"SUPER",lat,lon,track,highalt,hispd)    
                
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
             
    #create a superconflict of x aircraft in a circle towards the center
    elif command == "FUNNEL":
        if numargs ==0:
            scr.echo(callsign+"FUNNEL <FUNNELSIZE IN NUMBER OF A/C>")
        else:
            scr.isoalt=0
            traf.deleteall()
            traf.asas=CASASfunnel.Dbconf(traf,300., 5.*nm, 1000.*ft)
            size=float(commandargs[1])
            mperdeg=111319.
            distance=0.90 #this is in degrees lat/lon, for now
            alt=20000 #meters
            spd=200 #kts
            numac=8 #number of aircraft
            for i in range(numac):
                angle=np.pi/2/numac*i+np.pi/4
                acid="SUP"+str(i)
                traf.create(acid,"SUPER",distance*-np.cos(angle),distance*-np.sin(angle),90,alt,spd)             
                
            separation=traf.asas.R*1.01 #[m] the factor 1.01 is so that the funnel doesn't collide with itself
            sepdeg=separation/np.sqrt(2.)/mperdeg #[deg]
            
            for row in range(1):
                for col in range(15):
                    opening=(size+1)/2.*separation/mperdeg
                    Coldeg=sepdeg*col  #[deg]
                    Rowdeg=sepdeg*row  #[deg]
                    acid1="FUNN"+str(row)+"-"+str(col)
                    acid2="FUNL"+str(row)+"-"+str(col)
                    traf.create(acid1,"FUNNEL", Coldeg+Rowdeg+opening,-Coldeg+Rowdeg+0.5,0,alt,0)             
                    traf.create(acid2,"FUNNEL",-Coldeg-Rowdeg-opening,-Coldeg+Rowdeg+0.5,0,alt,0)
                    
            if savescenarios:
                fname="funnel"+str(size)
                cmd.saveic(fname,sim,traf)
                
             
    #create a conflict with several aircraft flying in a wall formation    
    elif command == "MATRIX":
        if numargs ==0:
            scr.echo(callsign+"MATRIX <SIZE>")
        else:
            size=int(float(commandargs[1]))
            scr.isoalt=0
            traf.deleteall()
            mperdeg=111319.
            hsep=traf.asas.R # [m] horizontal separation minimum
            hseplat=hsep/mperdeg
            matsep=1.1 #factor of extra space in the matrix
            hseplat=hseplat*matsep
            vel=200 #m/s
            extradist=(vel*1.1)*5*60/mperdeg #degrees latlon flown in 5 minutes
            for i in range(size):
                acidn="NORTH"+str(i)
                traf.create(acidn,"MATRIX",hseplat*(size-1.)/2+extradist,(i-(size-1.)/2)*hseplat,180,20000,vel)
                acids="SOUTH"+str(i)
                traf.create(acids,"MATRIX",-hseplat*(size-1.)/2-extradist,(i-(size-1.)/2)*hseplat,0,20000,vel)
                acide="EAST"+str(i)
                traf.create(acide,"MATRIX",(i-(size-1.)/2)*hseplat,hseplat*(size-1.)/2+extradist,270,20000,vel)
                acidw="WEST"+str(i)
                traf.create(acidw,"MATRIX",(i-(size-1.)/2)*hseplat,-hseplat*(size-1.)/2-extradist,90,20000,vel)
                
            if savescenarios:
                fname="matrix"+str(size)
                cmd.saveic(fname,sim,traf)

    #create a conflict with several aircraft flying in a floor formation    
    elif command == "FLOOR":
        scr.isoalt=1./50
        traf.deleteall()
        mperdeg=111319.
        altdif=3000 # meters
        hsep=traf.asas.R # [m] horizontal separation minimum
        floorsep=1.1 #factor of extra spacing in the floor
        hseplat=hsep/mperdeg*floorsep
        traf.create("OWNSHIP","FLOOR",-1,0,90, 20000+altdif, 200)
        idx = traf.id.index("OWNSHIP")
        traf.avs[idx]=-10
        traf.aalt[idx]=20000-altdif
        for i in range(20):
            acid="OTH"+str(i)
            traf.create(acid,"FLOOR",-1,(i-10)*hseplat,90,20000,200)            
        if savescenarios:
            fname="floor"
            cmd.saveic(fname,sim,traf)            

    #create a conflict with several aircraft overtaking eachother    
    elif command == "TAKEOVER":
        if numargs ==0:
            scr.echo(callsign+"TAKEOVER <NUMBER OF A/C>")
        else:
            numac=int(float(commandargs[1]))
            scr.isoalt=0
            traf.deleteall()
            mperdeg=111319.
            vsteps=50 #[m/s]
            for v in range(vsteps,vsteps*(numac+1),vsteps): #m/s
                acid="OT"+str(v)
                distancetofly=v*5*60 #m
                degtofly=distancetofly/mperdeg
                traf.create(acid,"OT",0,-degtofly,90,20000,v)
            if savescenarios:
                fname="takeover"+str(numac)
                cmd.saveic(fname,sim,traf)

    #create a conflict with several aircraft flying in a wall formation    
    elif command == "WALL":
        scr.isoalt=0
        traf.deleteall()
        mperdeg=111319.
        distance=0.6 # in degrees lat/lon, for now
        hsep=traf.asas.R # [m] horizontal separation minimum
        hseplat=hsep/mperdeg
        wallsep=1.1 #factor of extra space in the wall
        traf.create("OWNSHIP","WALL",0,-distance,90, 20000, 200)
        for i in range(20):
            acid="OTHER"+str(i)
            traf.create(acid,"WALL",(i-10)*hseplat*wallsep,distance,270,20000,200)
        if savescenarios:
            fname="wall"
            cmd.saveic(fname,sim,traf)            

            
    elif command == "TESTCIRCLE":
        scr.swtestarea = True   #show circles in testing area
        scr.redrawradbg=True    #draw the background again
        traf.asas=CASAScircle.Dbconf(traf,300., 5.*nm, 1000.*ft)
                                #change the ASAS system with one that incorporates
                                #the circular testing area
        traf.asas.Rtest=50*nm #Testing area radius
        traf.asas.Rinit=65*nm #Initialization area radius        

    # Toggle the display of certain elements in screen
    elif command == "DISP":
        if numargs == 0:
            scr.echo(callsign+"DISP <SEP/SPD/TEST>")
        else:
            sw = commandargs[1]
            
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
                
    #change isometric altitude
    elif command == "ISOALT":
        if numargs == 0:
            scr.echo(callsign+"ISOALT <pixels/meter>")
        else:
            scr.isoalt = float(commandargs[1])
        
    #save a screenshot
    elif command == "SNAP":
        scr.savescreen()

    elif command == "TEST":
        for i in range(6):
            acid="TEST"+str(i)
            traf.create(acid,"OT",0,i/6.,0,8000+4000*i,200)
    
    
    #give up
    else:
        scr.echo("Unknown command: " + callsign + command)
    pass

