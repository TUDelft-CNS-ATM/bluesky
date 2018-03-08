""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """


# Developed by Julia and Jerom, superteam!

# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools.aero import kts, ft, fpm, vcas2tas
from bluesky.tools import geo
import numpy as np

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    global fnumber, commandnames, teststart

    # Run the time
    stack.stack("OP")
    stack.stack("FF")
    # Reset the traffic simulation
    traf.reset()
    commandnames = list(stack.cmddict.keys())
    # Make a list of testing functions
    fnumber = 0
    teststart = True


    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'STACKCHECK',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        'update':          update,
        'preupdate':       preupdate
        }

    stackfunctions = {
        # The command name for your function
        'MYFUN': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'MYFUN ON/OFF',

            # A list of the argument types your function accepts. For a description of this, see ...
            '[onoff]',

            # The name of your function in this plugin
            myfun,

            # a longer help text of your function.
            'Print something to the bluesky console based on the flag passed to MYFUN.']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
    global commandnames, fnumber, teststart, timer
    if fnumber < len(commandnames):
        # form function name
        func = "test_"+ commandnames[fnumber].lower()
        timer = sim.simt

        # if function exists
        if func in globals().keys():
            # run the test and get result
            result = globals()[func](teststart)
            teststart = False
            if result != None:
                stack.stack("ECHO "+commandnames[fnumber]+" DOES %sWORK" % ("" if result else "NOT ") )
                # Prepare for next test function
                fnumber += 1
                teststart = True
                # reset traffic to get rid of unnecessary crap
                traf.reset()

        else:
            # if function does not exist - keep looping
            fnumber += 1
            #stack.stack("ECHO "+func+" DOES NOT EXIST")
    else:
        stack.stack("RESET")
        stack.stack("PAUSE")
        stack.stack("ECHO FINISHED PERFORMING STACKCHECK")
        stack.stack("PLUGIN REMOVE STACKCHECK")

def test_addwpt(start):
    global starttime, timer
    timelimit = 120
    if start:
        # Create a waypoint which does not lie on the aircraft path
        stack.stack("CRE KLM10 B747 52 4 000 FL99 150")
        stack.stack("KLM10 ADDWPT 52 4.1")
        starttime = timer
    else:
        # When the aircraft reaches the waypoint, return success
        _, d = geo.qdrdist(52,4.1,traf.lat[0],traf.lon[0])
        closeenough = d < 0.2 # d is measured in nm
        if timer-starttime>timelimit or closeenough:
            return closeenough

def test_after(start):
    global starttime, timer
    timelimit = 240
    lowertimelimit = 120
    if start:
        # Create a waypoint which does not lie on the aircraft path
        # And create a next waypoint to reach after that
        stack.stack("CRE KLM10 B747 52 4 000 FL99 150")
        stack.stack("DEFWPT TESTWP 52 4.1")
        stack.stack("KLM10 ADDWPT TESTWP")
        stack.stack("KLM10 AFTER TESTWP ADDWPT 51.9 4.1")
        starttime = timer
    else:
        # When the aircraft reaches the waypoint, return success
        _, d = geo.qdrdist(51.9,4.1,traf.lat[0],traf.lon[0])
        closeenough = d < 0.2 # d is measured in nm
        if timer-starttime>timelimit or closeenough:
            return closeenough and timer-starttime>lowertimelimit

def test_alt(start):
    global starttime, timer
    timelimit = 60.
    fl = 100
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 FL99 250")
        stack.stack("KLM10 ALT FL%s" %fl)
        starttime = timer
    else:
        closeenough = (fl*100 - 10) * ft <= traf.alt[0] < (fl*100 + 10) * ft
        if (timer - starttime) > timelimit or closeenough:
            return closeenough

def test_before(start):
    global starttime, timer
    timelimit = 240
    if start:
        # Create a waypoint which does not lie on the aircraft path
        # And create a next waypoint to reach after that
        stack.stack("CRE KLM10 B747 52 4 000 FL99 150")
        stack.stack("DEFWPT WP1 52.0 4.1")
        stack.stack("DEFWPT WP2 52.1 4.2")
        stack.stack("DEFWPT WP3 52.0 4.3")
        stack.stack("KLM10 ADDWPT WP1")
        stack.stack("KLM10 AFTER WP1 ADDWPT WP3")
        stack.stack("KLM10 BEFORE WP3 ADDWPT WP2")
        starttime = timer
    else:
        # When the aircraft reaches waypoint 3 within the time limit return success
        _, d = geo.qdrdist(52.1 ,4.2 ,traf.lat[0],traf.lon[0])
        closeenough = d < 4 # d is measured in nm
        if timer-starttime>timelimit or closeenough:
            return closeenough

def test_cre(start):
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 59 250")
    else:
        return traf.ntraf==1

def test_defwpt(start):
    global starttime, timer
    timelimit = 120
    if start:
        # Create a waypoint which does not lie on the aircraft path using defwpt
        stack.stack("CRE KLM10 B747 52 4 000 FL99 150")
        stack.stack("DEFWPT TESTWP 52 4.1")
        stack.stack("KLM10 ADDWPT TESTWP")
        starttime = timer
    else:
        # When the aircraft reaches the waypoint, return success
        _, d = geo.qdrdist(52,4.1,traf.lat[0],traf.lon[0])
        closeenough = d < 0.2 # d is measured in nm
        if timer-starttime>timelimit or closeenough:
            return closeenough

def test_del(start):
    global starttime, timer
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 59 250")
        stack.stack("CRE KLM11 B747 53 4 000 59 250")
        stack.stack("CRE KLM12 B747 54 4 000 59 250")
        starttime = timer
    elif traf.ntraf==3 and timer-starttime<5:
        stack.stack("DEL KLM10")
    else:
        return traf.ntraf==2

def test_delwpt(start):
    global starttime, timer, wp2reached
    timelimit = 240
    if start:
        # Create a waypoint which does not lie on the aircraft path
        # And create a next waypoint to reach after that
        stack.stack("CRE KLM10 B747 52 4 000 FL99 150")
        stack.stack("DEFWPT WPDEL1 52.0 4.1")
        stack.stack("DEFWPT WPDEL2 52.1 4.2")
        stack.stack("DEFWPT WPDEL3 52.0 4.3")
        stack.stack("KLM10 ADDWPT WPDEL1")
        stack.stack("KLM10 ADDWPT WPDEL2")
        stack.stack("KLM10 ADDWPT WPDEL3")
        stack.stack("KLM10 DELWPT WPDEL2")
        starttime = timer
        wp2reached = False
    else:
        _, d2 = geo.qdrdist(52.1 ,4.2 ,traf.lat[0],traf.lon[0])
        if d2 < 1:
            wp2reached = True
        # When the aircraft reaches waypoint 3 within the time limit return success
        _, d = geo.qdrdist(52.0 ,4.3 ,traf.lat[0],traf.lon[0])
        wp3reached = d < 1 # d is measured in nm
        if timer-starttime>timelimit or wp3reached:
            return wp3reached and not wp2reached

def test_dest(start):
    dest = 'EHAM'
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 FL100 250")
        stack.stack("KLM10 DEST %s"%dest)
    else:
        return traf.ap.dest[0] == dest

def test_direct(start):
    global starttime, timer, wp1reached
    timelimit = 350
    if start:
        # Create a waypoint which does not lie on the aircraft path
        # And create the waypoint to direct to
        stack.stack("CRE KLM10 B747 52 4 000 FL99 150")
        stack.stack("DEFWPT WPDIR1 52.2 4.1")
        stack.stack("DEFWPT WPDIR2 52.2 3.9")
        stack.stack("KLM10 ADDWPT WPDIR1")
        stack.stack("KLM10 ADDWPT WPDIR2")
        stack.stack("KLM10 DIRECT WPDIR2")
        starttime = timer
        wp1reached = False
    else:
        # If close to WP1, save that WP1 is reached
        _, d1 = geo.qdrdist(52.2 , 4.1 ,traf.lat[0],traf.lon[0])
        if d1 < 1:
            wp1reached = True

        # When the aircraft reaches waypoint 2 within the time limit, return success
        _, d2 = geo.qdrdist(52.2 , 3.9 ,traf.lat[0],traf.lon[0])
        closeenough = d2 < 1 # d is measured in nm
        if timer-starttime>timelimit or closeenough:
            return closeenough and not wp1reached

def test_eng(start):
    """ at a moment the engine change DOES result in performance coefficient change
    in accordance with a new engine, but the engine id/type is not changed
    """
    global starttime, timer
    timelimit = 5.  # seconds
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 FL100 250")
        starttime = timer
    # elif timer - starttime < timelimit:
    #     print(traf.perf.engines)
    #     print("eng type before change", traf.perf.etype[0])
    #     stack.stack("ENG KLM10 RB211-22B")
    #     print("eng type after change", traf.perf.etype[0])
    else:
        return False

def test_hdg(start):
    global starttime, timer
    timelimit = 70.  # seconds
    new_heading = 170
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 FL100 250")
        stack.stack("KLM10 HDG %s"%new_heading)
        starttime = timer
    else:
        closeenough = (new_heading - 1) <= traf.hdg[0] < (new_heading + 1)
        if (timer - starttime) > timelimit or closeenough:
            return closeenough

def test_lnav(start):
    global starttime, timer, detourmade
    timelimit = 150
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 59 250")
        stack.stack("CRE KLM11 B747 52 5.1 000 59 250")
        stack.stack("KLM10 ADDWPT 52.2 4")
        stack.stack("KLM11 ADDWPT 52.2 5.1")
        stack.stack("KLM10 ADDWPT 52.2 3.5")
        stack.stack("KLM11 ADDWPT 52.2 5.6")
        stack.stack("KLM10 LNAV OFF")
        stack.stack("KLM10 LNAV ON")
        stack.stack("KLM11 LNAV OFF")
        starttime = timer
    elif timer-starttime > timelimit:
        lnavon = traf.hdg[0]>=265
        lnavoff = traf.hdg[1]<= 5 or traf.hdg[1] >=355
        return lnavon and lnavoff

def test_mcre(start):
    ac_type = 'B747'
    alt = 100  # FL
    spd = 250  # CAS in knots
    ades = 'EHAM'  # airport of destination is not checked yet
    if start:
        stack.stack("MCRE 11")
    elif traf.ntraf == 11:
        traf.reset()
        stack.stack("MCRE 5 %s FL%s %s %s" % (ac_type, alt, spd, ades))
    else:

        close_enough_alt = np.logical_and(traf.alt <= (alt * 100 * ft + 10),
                                          traf.alt >= (alt * 100 * ft - 10))

        close_enough_spd = np.logical_and(traf.cas <= (spd * kts + 10),
                                          traf.cas >= (spd * kts - 10))

        return (traf.ntraf == 5 and len(traf.type)==5 and
                np.all(close_enough_alt) and np.all(close_enough_spd))

def test_move(start):
    # set new values for a MOVE command 
    new_lat, new_lon = 52, 8
    new_alt = 200  # FL
    new_hdg = 180  # degrees
    new_spd = 300  # CAS in knots
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 FL100 250")
        stack.stack("MOVE KLM10 52 8 FL200 180 300")
    else:
        # check whether variables are within the reasonable ranges from the set values
        close_enough_lat = new_lat - 0.1 <= traf.lat[0] <= new_lat + 0.1
        close_enough_lon = new_lon - 0.1 <= traf.lon[0] <= new_lon + 0.1
        close_enough_hdg = new_hdg - 1 <= traf.hdg[0] <= new_hdg + 1
        close_enough_alt = new_alt * 100 * ft - 10 <= traf.alt[0] <= new_alt * 100 * ft + 10
        close_enough_spd = new_spd * kts - 5 <= traf.cas[0] <= new_spd * kts + 5
        return close_enough_lat and close_enough_lon and close_enough_hdg and close_enough_alt and close_enough_spd

def test_orig(start):
    orig = 'EHAM'
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 FL100 250")
        stack.stack("KLM10 ORIG %s"%orig)
    else:
        return traf.ap.orig[0] == orig

def test_spd(start):
    global starttime, timer
    timelimit = 60.
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 59 250")
        stack.stack("KLM10 SPD 300")
        starttime = timer
    else:
        closeenough = 295 *kts <= traf.cas[0] < 305*kts
        if timer-starttime > timelimit or closeenough:
            return closeenough

def test_vnav(start):
    global starttime, timer, detourmade
    timelimit = 60
    if start:
        stack.stack("CRE KLM10 B747 52 4 000 FL59 250")
        stack.stack("CRE KLM11 B747 52 5.1 000 FL59 250")
        stack.stack("KLM10 ADDWPT 52.1 4 FL65")
        stack.stack("KLM11 ADDWPT 52.1 5.1 FL65")
        stack.stack("KLM10 VNAV OFF")
        stack.stack("KLM10 VNAV ON")
        stack.stack("KLM11 VNAV OFF")
        starttime = timer
    elif timer-starttime > timelimit:
        vnavon  = traf.alt[0] >= 6200 *ft
        vnavoff = 5800*ft < traf.alt[1] < 6000 *ft
        return vnavon and vnavoff

def test_vs(start):
    global starttime, timer
    timelimit = 10.  # seconds
    new_vs = 2000  # fpm
    init_alt, new_alt = 100, 200
    if start:
        # Create an aircraft that should perform a vertical manoever
        stack.stack("CRE KLM10 B747 52 4 000 FL%s 250"%init_alt)
        stack.stack("KLM10 ALT FL%s"%new_alt)
        stack.stack("KLM10 VS %s" % new_vs)
        starttime = timer
    else:
        # Check if the aircraft adopted the new vs
        closeenough = (new_vs * fpm - 10) <= traf.vs[0] < (new_vs * fpm + 10)
        if (timer - starttime) > timelimit or closeenough:
            return closeenough

def test_wind(start):
    global starttime, timer
    timelimit = 20
    if start:
        # Create a very strong headwind
        cas = 250
        tas = vcas2tas(cas*kts,10000*ft)/kts
        stack.stack("CRE KLM10 B747 52 4 000 FL100 "+str(int(cas)))
        # Mind that wind is defined in the direction that it is coming from
        stack.stack("WIND 52 4 FL100 000 "+str(int(tas)))
        starttime = timer
    elif timer-starttime>timelimit:
        # If the aircraft did not move 
        _, d = geo.qdrdist(52,4,traf.lat[0],traf.lon[0])
        return d < 0.1 # d is measured in nm

def preupdate():
    pass

### Other functions of your plugin
def myfun(flag=True):
    return True, 'My plugin received an o%s flag.' % ('n' if flag else 'ff')
