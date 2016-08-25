""" BlueSky Datalogger """
from collections import OrderedDict
from .. import settings

# Lists to contain the definitions of event and periodic logs
eventlogs = dict()
periodicvars = OrderedDict()
selectedpvars = list()
periodiclogs = {
    'SNAPLOG': ['this is the header', None],
    'INSTLOG': ['this is the header', None],
    'SKYLOG': ['this is the header', None],
    'SELECTIVESNAP': ['this is the header', None]
}


def setPeriodicVars(pvars):
    global periodicvars, selectedpvars
    periodicvars = pvars
    selectedpvars = pvars.values()


def createEventLog(name, header, variables):
    """ This function creates the event log dictionary. Call it in the __init__
    function of the class where the desired event logger is going to be used """
    # Eventlogs is a dict with the separate event logs, that each have a header,
    # a set of variables with their format, and a reference to the current file
    # (which starts as None)
    global eventlogs
    varnames, formats = zip(*variables)
    header = header + '\n' + str.join(', ', varnames)
    eventlogs[name] = [header, tuple(formats), None]


def start(name, fileext=''):
    """ This function opens a text file and writes header text to it. It is to
    be called when a particular log is activated, or started, with a stack command"""
    global eventlogs, periodiclogs
    if name in periodiclogs:
        if periodiclogs[name][1] is not None:
            periodiclogs[name][1].close()
        # TO DO: NAME OF SCENARIO AND GMT TIME
        # QUESTION: Should we close the files after wrting, and open them again when necessary?
        periodiclogs[name][1] = open(name + '.' + fileext, 'w')
        periodiclogs[name][1].write(periodiclogs[name][0] + '\n')

    elif name in eventlogs:
        if eventlogs[name][2] is not None:
            eventlogs[name][2].close()
        # TO DO: NAME OF SCENARIO AND GMT TIME
        eventlogs[name][2] = open(name + '.' + fileext, 'w')
        eventlogs[name][2].write(eventlogs[name][0] + '\n')

    else:
        return False, name + ' not found.'


def stop(name=None):
    """ This function stops logging of the desired log, or stops all logs. It is
    to be called when a log is switched off using a stack command, or when BlueSky
    exits"""
    global eventlogs, periodiclogs
    if name is None:
        # Close all open file objects
        for elog in eventlogs:
            elog[2].close()
            elog[2] = None
        for plog in periodiclogs:
            plog[1].close()
            plog[1] = None
    elif name in periodiclogs:
        periodiclogs[name][1].close()
        periodiclogs[name][1] = None
    elif name in eventlogs:
        eventlogs[name][2].close()
        eventlogs[name][2] = None


def reset():
    """ This function resets all logs. It is to be called when simulation is 
    reset and in reset functions of the all the classes where periodic (traffic) 
    and event logs are used """
    global eventlogs, periodiclogs
    global snapdt, instdt, skydt, selsnapdt, snapt, instt, skyt, selsnapt

    snapt = 0.0
    instt = 0.0
    skyt = 0.0
    selsnapt = 0.0

    snapdt = settings.snapdt
    instdt = settings.instdt
    skydt = settings.skydt
    selsnapdt = settings.selsnapdt

    # Close all periodic logs and remove reference to its file object
    for name in periodiclogs:
        if periodiclogs[name][1] is not None:
            periodiclogs[name].close()
            periodiclogs[name] = None
    # Close all event logs and remove reference to its file object
    for name in eventlogs:
        if eventlogs[name][2] is not None:
            eventlogs[name].close()
            eventlogs[name] = None


def logEvent(name, *data):
    """ This function is used to write to file the details of a desrired event
    when it occurs. It is to be called at the location where such event could occur"""
    log = eventlogs[name]
    formats = log[1]
    eventfile = log[2]
    if eventfile is None:
        return
    eventfile.write(str.join(', ', formats) % data)


def logPeriodic(simt, traf):
    """ This function writes to files of all periodic logs by calling the appropriate 
    functions for each type of periodic log, at the approriate update time. """
    global snapt, instt, skyt, selsnapt

    fsnap = periodiclogs['SNAPLOG'][1]
    finst = periodiclogs['INSTLOG'][1]
    fsky = periodiclogs['SKYLOG'][1]
    fselsnap = periodiclogs['SELECTIVESNAP'][1]

    # update each of the periodic logs if it is their update time and if activated
    if fsnap and simt >= snapt:
        snapt = simt + snapdt
        # Write to fsnap
        logSnap(simt, traf, fsnap)
    elif finst and simt >= instt:
        instt = simt + instdt
        # Write to finst
        logInst(simt, traf, finst)
    elif fsky and simt >= skyt:
        skyt = simt + skydt
        # Write to fsky
        logSky(simt, traf, fsky)
    elif fselsnap and simt >= selsnapt:
        selsnapt = simt + selsnapdt
        # Write to fselsnap
        logSelSnap(simt, traf, fselsnap)


def logSnap(simt, traf, fsnap):
    """ This function writes the snap data to the snap file """
    fsnap.write()


def logInst(simt, traf, finst):
    """ This function writes the intantaneous conflict data to the inst file """
    finst.write()


def logSky(simt, traf, fsky):
    """ This function writes the simulation summary data to the sky file """
    fsky.write()


def logSelSnap(simt, traf, fselsnap):
    """ This function writes the snap data of the desired aircraft/s to the selective snap file """


# TO DO: Functions to switch on logging based on the stack commands. One function
# for each of the 6 deafult logs. These functions should also set the logdt of each
# log type as an optional argument.

''' ALGORITHM TO CONVERT ARRAY TO STRING AND WRITE ARRAY TO FILE

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

# Create a random array with a large size
largearray = np.random.rand(10000,50)
stacker = np.random.rand(10000,1)
largearray = np.hstack((stacker,largearray))

tic()
largearray = largearray.astype('|S10')
toc()

# write array to file 
with open('largeArray.txt','w') as f_handle:
    np.savetxt(f_handle,largearray, delimiter=',   ', newline='\n', fmt='%.10s')

# create another random array with a large size
largearray2 = np.random.rand(10000,50)
largearray2 = largearray.astype('|S10')

tic()
# append array to file 
with open('largeArray.txt','a') as f_handle:
    np.savetxt(f_handle,largearray2,delimiter=',   ', newline='\n', fmt='%.10s')

toc()
'''
