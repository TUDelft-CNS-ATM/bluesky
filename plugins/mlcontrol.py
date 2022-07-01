""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, net, sim, traf  #, settings, navdb, traf, sim, scr, tools


myclientrte = None

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'MLCONTROL',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 1.0,

        'update':          update,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         reset
        }

    stackfunctions = {
        # The command name for your function
        'MLSTEP': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'MLSTEP',

            # A list of the argument types your function accepts. For a description of this, see ...
            '',

            # The name of your function in this plugin
            mlstep,

            # a longer help text of your function.
            'Simulate one MLCONTROL time interval.']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
    data = dict(
        lat=traf.lat,
        lon=traf.lon,
        alt=traf.alt
    )
    net.send_event(b'MLSTATEREPLY', data, myclientrte)
    sim.hold()

def preupdate():
    pass

def reset():
    pass

def mlstep():
    global myclientrte
    myclientrte = stack.routetosender()
    sim.op()