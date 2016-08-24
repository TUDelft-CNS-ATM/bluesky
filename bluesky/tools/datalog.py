""" BlueSky Datalogger """
from .. import settings

eventlogs  = dict()
periodiclogs = {
    'SNAPLOG' : ['this is the header', None],
    'INSTLOG' : ['this is the header', None],
    'SKYLOG'  : ['this is the header', None],
    'SELECTIVESNAP' : ['this is the header', None]
}


def createEventLog(name, variables, header):
    # Eventlogs is a dict with the separate event logs, that each have a header,
    # a set of variables, and a reference to the current file (which starts
    # as None)
    global eventlogs
    varnames, formats  = zip(*variables)
    header             = header + '\n' + str.join(', ', varnames)
    eventlogs[name]    = [header, tuple(formats), None]


def logEvent(name, *data):
    log     = eventlogs[name]
    formats = log[1]
    file    = log[2]
    if file is None:
        return
    file.write(str.join(', ', formats) % data)


def logPeriodic(simt, traf):
    global snapt, instt, skyt, selsnapt
    fsnap    = periodiclogs['SNAPLOG'][1]
    finst    = periodiclogs['INSTLOG'][1]
    fsky     = periodiclogs['SKYLOG'][1]
    fselsnap = periodiclogs['SELECTIVESNAP'][1]
    if fsnap and simt > snapt:
        snapt = snapt + snapdt
        # Write to fsnap
        logSnap(simt, traf)
    elif finst and simt > instt:
        instt = instt + instdt
        # Write to finst
        logInst(simt, traf)
    elif fsky and simt > skyt:
        skyt = skyt + skydt
        # Write to fsky
        logSky(simt, traf)
    elif fselsnap and simt > selsnapt:
        selsnapt = selsnapt + selsnapdt
        # Write to fselsnap
        logSelSnap(simt, traf)

def logSnap(simt, traf):
    file = periodiclogs['SNAPLOG'][1]
    file.write()


def save(name=None):
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


def start(name, suffix=''):
    global eventlogs, periodiclogs
    if name in periodiclogs:
        if periodiclogs[name][1] is not None:
            periodiclogs[name][1].close()
        periodiclogs[name][1] = open(name + '.' + suffix, 'w')
        periodiclogs[name][1].write(periodiclogs[name][0] + '\n')

    elif name in eventlogs:
        if eventlogs[name][2] is not None:
            eventlogs[name][2].close()
        eventlogs[name][2] = open(name + '.' + suffix, 'w')
        eventlogs[name][2].write(eventlogs[name][0] +  '\n')

    else:
        return False, name + ' not found.'


def reset():
    global eventlogs, periodiclogs, snapdt, instdt, skydt, selsnapdt, snapt, instt, skyt, selsnapt
    for name in periodiclogs:
        if periodiclogs[name][1] is not None:
            periodiclogs[name].close()
            periodiclogs[name] = None

    snapt    = 0.0
    instt    = 0.0
    skyt     = 0.0
    selsnapt = 0.0

    snapdt    = settings.snapdt
    instdt    = settings.instdt
    skydt     = settings.skydt
    selsnapdt = settings.selsnapdt

