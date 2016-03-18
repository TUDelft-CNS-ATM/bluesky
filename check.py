#!/usr/bin/python
import traceback
print "checking for pyqt,",
try:
    import PyQt4
except ImportError:
    try:
        import PyQt5
    except ImportError:
        print "pyqt version 4 or 5 missing. You might still be ok if you want to run the pygame version of BlueSky."
    else:
        print "OK: PyQt5 found."
else:
    print "OK: PyQt4 found."

print "checking for pyopengl,",
try:
    import OpenGL.GL
except:
    print "pyopengl is missing."
else:
    print "OK."

print "checking for pygame,",
try:
    import pygame
except ImportError:
    print "pygame is missing.."
else:
    print 'OK.'

print "checking for scipy,",
try:
    import scipy
except ImportError:
    print "scipy is missing.."
else:
    print 'OK.'

print "checking for numpy,",
try:
    import numpy
except ImportError:
    print "numpy is missing.."
else:
    print 'OK.'

print "checking for matplotlib,",
try:
    import matplotlib
except ImportError:
    print "matplotlib is missing.."
else:
    print 'OK.'


print "checking bluesky modules"
try:
    from bluesky import *
    from bluesky.ui import *
    from bluesky.stack import *
    from bluesky.sim import *
    from bluesky.traf import *
    from bluesky.traf.asas import *
    from bluesky.traf.metric import *
    from bluesky.traf.navdb import *
    from bluesky.traf.params import *
    from bluesky.traf.perf import *
except Exception, err:
    print traceback.format_exc()
    print "One or more bluesky modules is not working properly, check error for more detail."
else:
    print "Cool! all modules good to go!"
