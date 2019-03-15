""" Stack functions to generate synthetic traffic scenarios."""
import random
import numpy as np
import bluesky as bs
from bluesky import stack
from bluesky.tools.aero import ft, eas2tas,kts
from bluesky.tools import geo
from bluesky.tools.misc import txt2alt, txt2spd

def wrapcreate(acid=None, actype=None,aclat=None, aclon=None,
                                   achdg=None, acalt=None, acspd=None):
    stack.stack("CRE "+",".join([acid,actype,str(aclat),str(aclon),str(achdg),str(acalt/ft),
                                 str(acspd/kts)]))
    return

savescenarios=False #whether to save a scenario as .scn file after generation via commands

def process(*cmdargs):
    command = cmdargs[0]
    numargs = len(cmdargs) - 1
    callsign = 'SYN_'

    # change display settings and delete AC to generate own FF scenarios
    if command == "START":
        bs.scr.swgeo=False         #don't draw coastlines and borders
        bs.scr.swsat=False         #don't draw the satellite image
        bs.scr.apsw=0              #don't draw airports
        bs.scr.swlabel=0           #don't draw aircraft labels
        bs.scr.wpsw=0              #don't draw waypoints
        bs.scr.swfir=False         #don't show FIRs
        bs.scr.swgrid=True         #do show a grid
        bs.scr.pan(0, 0)           #focus the map at the prime meridian and equator
        bs.scr.redrawradbg=True    #draw the background again
        bs.scr.swsep = True        #show circles of seperation between ac
        bs.scr.swspd = True        #show speed vectors of aircraft
        bs.scr.zoom(0.4, True)     #set zoom level to the standard distance
        # cmd.scenlines=[]        #skip the rest of the scenario
        # cmd.scencmd=[]          #skip the rest of the scenario
        # cmd.scentime=[]         #skip the rest of the scenario
        # cmd.scenlines.append("00:00:00.00>"+callsign+"TESTCIRCLE")
        # cmd.scenlines.append("00:00:00.00>DT 1")
        # cmd.scenlines.append("00:00:00.00>FIXDT ON")
        bs.stack("RESET")

    # display help
    elif command == "HELP":
        return True, ("This is the synthetic traffic scenario module\n"
            "Possible subcommands: HELP, SIMPLE, SIMPLED, DIFG, SUPER, SPHERE, "
            "MATRIX, FLOOR, TAKEOVER, WALL, ROW, COLUMN, DISP")

    #create a perpendicular conflict between two aircraft
    elif command == "SIMPLE":
        bs.scr.isoalt = 0
        bs.traf.reset()
        bs.traf.create(acid="OWNSHIP", actype="GENERIC", aclat=-.5, aclon=0,
                       achdg=0, acalt=5000 * ft, acspd=200)
        bs.traf.create(acid="INTRUDER", actype="GENERIC", aclat=0, aclon=.5,
                       achdg=270, acalt=5000 * ft, acspd=200)
        return True

    #create a perpendicular conflict with slight deviations to aircraft speeds and places
    elif command == "SIMPLED":
        bs.scr.isoalt = 0
        bs.traf.reset()
        ds = random.uniform(0.92, 1.08)
        dd = random.uniform(0.92, 1.08)
        bs.traf.create(acid="OWNSHIP", actype="GENERIC", aclat=-.5 * dd, aclon=0,
                       achdg=0, acalt=20000 * ft, acspd=200 * ds)
        bs.traf.create(acid="INTRUDER", actype="GENERIC", aclat=0, aclon=.5 / dd,
                       achdg=270, acalt=20000 * ft, acspd=200 / ds)
        return True

    # used for testing the differential game resolution method
    elif command == "DIFG":
        if numargs < 5:
            return False, "5 ARGUMENTS REQUIRED"
        else:
            bs.scr.isoalt=0
            bs.traf.reset()
            x=  bs.traf.asas.xw[int(float(cmdargs[1]))]/111319.
            y=  bs.traf.asas.yw[int(float(cmdargs[2]))]/111319.
            v_o=bs.traf.asas.v_o[int(float(cmdargs[3]))]
            v_w=bs.traf.asas.v_w[int(float(cmdargs[4]))]
            phi=np.degrees(bs.traf.asas.phi[int(float(cmdargs[5]))])
            wrapcreate(acid="OWN", actype="GENERIC", aclat=0, aclon=0,
                           achdg=0, acalt=5000*ft, acspd=v_o)
            wrapcreate(acid="WRN", actype="GENERIC", aclat=y, aclon=x,
                           achdg=phi, acalt=5000*ft, acspd=v_w)
            return True

    # create a superconflict of x aircraft in a circle towards the center
    elif command == "SUPER":
        if numargs ==0:
            return True, callsign+"SUPER <NUMBER OF A/C>"
        else:
            bs.scr.isoalt=0
            bs.traf.reset()
            numac=int(float(cmdargs[1]))
            distance=0.50 #this is in degrees lat/lon, for now
            alt=20000*ft #ft
            spd=200 #kts
            for i in range(numac):
                angle=2*np.pi/numac*i
                acid = "SUP" + str(i)
                wrapcreate(acid=acid, actype="SUPER",
                               aclat=distance * -np.cos(angle),
                               aclon=distance * np.sin(angle),
                               achdg=360.0 - 360.0 / numac * i,
                               acalt=alt, acspd=spd)
            if savescenarios:
                fname="super"+str(numac)
                # cmd.saveic(fname)
            return True

    # create a sphereconflict of 3 layers of superconflicts
    elif command == "SPHERE":
        if numargs ==0:
            return True, callsign+"SPHERE <NUMBER OF A/C PER LAYER>"
        else:
            bs.scr.isoalt=1./200
            bs.traf.reset()
            numac=int(float(cmdargs[1]))
            distance=0.5 #this is in degrees lat/lon, for now
            distancenm=distance*111319./1852
            alt=20000 #ft
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
                wrapcreate(acid=acidl, actype="SUPER",aclat=lat, aclon=lon,
                               achdg=track, acalt=lowalt*ft, acspd=lospd)
                acidm="SPH"+str(i)+"MID"
                wrapcreate(acid=acidm, actype="SUPER",aclat=lat, aclon=lon,
                               achdg=track, acalt=midalt*ft, acspd=mispd)
                acidh="SPH"+str(i)+"HIG"
                wrapcreate(acid=acidh, actype="SUPER",aclat=lat, aclon=lon,
                               achdg=track, acalt=highalt*ft, acspd=hispd)

                idxl = bs.traf.id.index(acidl)
                idxh = bs.traf.id.index(acidh)

                bs.traf.vs[idxl]     =  vs
                bs.traf.vs[idxh]     = -vs

                bs.traf.selvs[idxl]  =  vs
                bs.traf.selvs[idxh]  = -vs

                bs.traf.selalt[idxl] = highalt
                bs.traf.selalt[idxh] = lowalt

            if savescenarios:
                fname="sphere"+str(numac)
                # cmd.saveic(fname)
            return True

    elif command == "FUNNEL":
        if numargs ==0:
            bs.scr.echo(callsign+"FUNNEL <FUNNELSIZE IN NUMBER OF A/C>")
        else:
            bs.scr.isoalt=0
            bs.traf.deleteall()
            bs.traf.asas=CASASfunnel.Dbconf(300., 5.*nm, 1000.*ft)
            size=float(commandargs[1])
            mperdeg=111319.
            distance=0.90 #this is in degrees lat/lon, for now
            alt=20000 #meters
            spd=200 #kts
            numac=8 #number of aircraft
            for i in range(numac):
                angle=np.pi/2/numac*i+np.pi/4
                acid="SUP"+str(i)
                wrapcreate(acid=acid, actype="SUPER",
                               aclat=distance*-np.cos(angle),
                               aclon=distance*-np.sin(angle),
                               achdg=90, acalt=alt, acspd=spd)

            separation=bs.traf.asas.R*1.01 #[m] the factor 1.01 is so that the funnel doesn't collide with itself
            sepdeg=separation/np.sqrt(2.)/mperdeg #[deg]

            for row in range(1):
                for col in range(15):
                    opening=(size+1)/2.*separation/mperdeg
                    Coldeg=sepdeg*col  #[deg]
                    Rowdeg=sepdeg*row  #[deg]
                    acid1="FUNN"+str(row)+"-"+str(col)
                    acid2="FUNL"+str(row)+"-"+str(col)
                    wrapcreate(acid=acid1, actype="FUNNEL",
                                   aclat=Coldeg+Rowdeg+opening,
                                   aclon=-Coldeg+Rowdeg+0.5,
                                   achdg=0, acalt=alt, acspd=0)
                    wrapcreate(acid=acid2, actype="FUNNEL",
                                   aclat=-Coldeg-Rowdeg-opening,
                                   aclon=-Coldeg+Rowdeg+0.5,
                                   achdg=0, acalt=alt, acspd=0)

            if savescenarios:
                fname="funnel"+str(size)
                cmd.saveic(fname)

    # create a conflict with several aircraft flying in a matrix formation
    elif command == "MATRIX":
        if numargs ==0:
            return True, callsign+"MATRIX <SIZE>"
        else:
            size=int(float(cmdargs[1]))
            bs.scr.isoalt=0
            bs.traf.reset()
            mperdeg=111319.
            hsep=bs.traf.asas.R # [m] horizontal separation minimum
            hseplat=hsep/mperdeg
            matsep=1.1 #factor of extra space in the matrix
            hseplat=hseplat*matsep
            vel=200 #m/s
            extradist=(vel*1.1)*5*60/mperdeg #degrees latlon flown in 5 minutes
            for i in range(size):
                acidn="NORTH"+str(i)
                wrapcreate(acid=acidn, actype="MATRIX",
                               aclat=hseplat*(size-1.)/2+extradist,
                               aclon=(i-(size-1.)/2)*hseplat,
                               achdg=180, acalt=20000*ft, acspd=vel)
                acids="SOUTH"+str(i)
                wrapcreate(acid=acids, actype="MATRIX",
                               aclat=-hseplat*(size-1.)/2-extradist,
                               aclon=(i-(size-1.)/2)*hseplat,
                               achdg=0, acalt=20000*ft, acspd=vel)
                acide="EAST"+str(i)
                wrapcreate(acid=acide, actype="MATRIX",
                               aclat=(i-(size-1.)/2)*hseplat,
                               aclon=hseplat*(size-1.)/2+extradist,
                               achdg=270, acalt=20000*ft, acspd=vel)
                acidw="WEST"+str(i)
                wrapcreate(acid=acidw, actype="MATRIX",
                               aclat=(i-(size-1.)/2)*hseplat,
                               aclon=-hseplat*(size-1.)/2-extradist,
                               achdg=90, acalt=20000*ft, acspd=vel)

            if savescenarios:
                fname="matrix"+str(size)
                # cmd.saveic(fname)
            return True

    # create a conflict with several aircraft flying in a floor formation
    elif command == "FLOOR":
        bs.scr.isoalt=1./50
        bs.traf.reset()
        mperdeg=111319.
        altdif=3000 # ft
        hsep=bs.traf.asas.R # [m] horizontal separation minimum
        floorsep=1.1 #factor of extra spacing in the floor
        hseplat=hsep/mperdeg*floorsep
        wrapcreate(acid="OWNSHIP", actype="FLOOR",
                       aclat=-1, aclon=0,
                       achdg=90, acalt=(20000+altdif)*ft, acspd=200)
        idx = bs.traf.id.index("OWNSHIP")
        bs.traf.selvs[idx]=-10
        bs.traf.selalt[idx]=20000-altdif
        for i in range(20):
            acid="OTH"+str(i)
            wrapcreate(acid=acid, actype="FLOOR",
                           aclat=-1, aclon=(i-10)*hseplat,
                           achdg=90, acalt=20000*ft, acspd=200)
        if savescenarios:
            fname="floor"
            # cmd.saveic(fname)
        return True

    # create a conflict with several aircraft overtaking eachother
    elif command == "TAKEOVER":
        if numargs ==0:
            return True, callsign+"TAKEOVER <NUMBER OF A/C>"
        else:
            numac=int(float(cmdargs[1]))
            bs.scr.isoalt=0
            bs.traf.reset()
            mperdeg=111319.
            vsteps=50 #[m/s]
            for v in range(vsteps,vsteps*(numac+1),vsteps): #m/s
                acid="OT"+str(v)
                distancetofly=v*5*60 #m
                degtofly=distancetofly/mperdeg
                wrapcreate(acid=acid, actype="OT", aclat=0, aclon=-degtofly,
                               achdg=90, acalt=20000*ft, acspd=v)
            if savescenarios:
                fname="takeover"+str(numac)
                # cmd.saveic(fname)
            return True

    # create a conflict with several aircraft flying in a wall formation
    elif command == "WALL":
        bs.scr.isoalt=0
        bs.traf.reset()
        mperdeg=111319.
        distance=0.6 # in degrees lat/lon, for now
        hsep=bs.traf.asas.R # [m] horizontal separation minimum
        hseplat=hsep/mperdeg
        wallsep=1.1 #factor of extra space in the wall
        wrapcreate(acid="OWNSHIP", actype="WALL",
                       aclat=0, aclon=-distance,
                       achdg=90, acalt=20000*ft, acspd=200)
        for i in range(20):
            acid="OTHER"+str(i)
            wrapcreate(acid=acid, actype="WALL",
                           aclat=(i-10)*hseplat*wallsep, aclon=distance,
                           achdg=270, acalt=20000*ft, acspd=200)
        if savescenarios:
            fname="wall"
            # cmd.saveic(fname)
        return True

    # create a conflict with several aircraft flying in two rows angled towards each other
    elif command == "ROW":
        commandhelp = "SYN_ROW n angle [-r=radius in NM] [-a=alt in ft] [-s=speed EAS in kts] [-t=actype]"
        if numargs == 0:
            return True, commandhelp
        else:
            try:
                bs.traf.reset() # start fresh
                synerror,acalt,acspd,actype,startdistance,ang = angledtraffic.arguments(numargs,cmdargs[1:]) # process arguments
                if synerror:
                    raise Exception()

                mperdeg=111319.
                hsep=bs.traf.asas.R # [m] horizontal separation minimum
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
                    wrapcreate(acid="ANG"+str(i*2), actype=actype,
                                   aclat=aclat, aclon=aclon,
                                   achdg=180+ang, acalt=acalt*ft, acspd=acspd)
                    wrapcreate(acid="ANG"+str(i*2+1), actype=actype,
                                   aclat=aclat, aclon=-aclon,
                                   achdg=180-ang, acalt=acalt*ft, acspd=acspd)
                    alternate = alternate * -1

                bs.scr.pan([0,0],True)
                return True
            except Exception:
                return False, 'unknown argument flag'
            except:
                return False, commandhelp

    # create a conflict with several aircraft flying in two columns angled towards each other
    elif command == "COLUMN":
        commandhelp = "SYN_COLUMN n angle [-r=radius in NM] [-a=alt in ft] [-s=speed EAS in kts] [-t=actype]"
        if numargs == 0:
            return True, commandhelp
        else:
            try:
                bs.traf.reset() # start fresh
                synerror,acalt,acspd,actype,startdistance,ang = angledtraffic.arguments(numargs,cmdargs[1:]) # process arguments
                if synerror:
                    raise Exception()

                mperdeg=111319.
                hsep=bs.traf.asas.R # [m] horizontal separation minimum
                hseplat=hsep/mperdeg
                matsep=1.1 #factor of extra space in the formation
                hseplat=hseplat*matsep

                aclat = startdistance * np.cos(np.deg2rad(ang)) #[deg]
                aclon = startdistance * np.sin(np.deg2rad(ang))
                latsep = abs(hseplat*np.cos(np.deg2rad(ang))) #[deg]
                lonsep = abs(hseplat*np.sin(np.deg2rad(ang)))

                wrapcreate(acid="ANG0", actype=actype,
                               aclat=aclat, aclon=aclon,
                               achdg=180+ang, acalt=acalt*ft, acspd=acspd)
                wrapcreate(acid="ANG1", actype=actype,
                               aclat=aclat, aclon=-aclon,
                               achdg=180-ang, acalt=acalt*ft, acspd=acspd)

                for i in range(1,int(cmdargs[1])): # Create a/c
                    aclat = aclat+latsep
                    aclon = aclon+lonsep
                    wrapcreate(acid="ANG"+str(i*2), actype=actype,
                                   aclat=aclat, aclon=aclon,
                                   achdg=180+ang, acalt=acalt*ft, acspd=acspd)
                    wrapcreate(acid="ANG"+str(i*2+1), actype=actype,
                                   aclat=aclat, aclon=-aclon,
                                   achdg=180-ang, acalt=acalt*ft, acspd=acspd)

                bs.scr.pan([0,0],True)
                return True
            except Exception:
                return False, 'unknown argument flag'
            except:
                return False, commandhelp

    #give up
    else:
        return False, "Unknown command: " + callsign + command


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
                    startdistance = geo.qdrpos(0,0,90,float(cmdargs[i][3:]))[2] #input in nm
                elif cmdargs[i].upper().startswith("-A"): #altitude
                    acalt = txt2alt(cmdargs[i][3:])*ft
                elif cmdargs[i].upper().startswith("-S"): #speed
                    acspd = txt2spd(cmdargs[i][3:],acalt)
                elif cmdargs[i].upper().startswith("-T"): #ac type
                    actype = cmdargs[i][3:].upper()
                else:
                    syntaxerror = True
        return syntaxerror,acalt,acspd,actype,startdistance,ang
