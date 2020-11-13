''' This module provides PerfBase, the base class for aircraft
    performance implementations.
'''
import numpy as np
from bluesky import settings
from bluesky.core import Entity, timed_function
from bluesky.stack import command


settings.set_variable_defaults(performance_dt=1.0)


class PerfBase(Entity, replaceable=True):
    """ Base class for BlueSky aircraft performance implementations. """
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            # --- fixed parameters ---
            self.actype = np.array([], dtype=str)  # aircraft type
            self.Sref = np.array([])  # wing reference surface area [m^2]
            self.engtype = np.array([])  # integer, aircraft.ENG_TF...

            # --- dynamic parameters ---
            self.mass = np.array([])  # effective mass [kg]
            self.phase = np.array([])
            self.cd0 = np.array([])
            self.k = np.array([])
            self.bank = np.array([])
            self.thrust = np.array([])  # thrust
            self.drag = np.array([])  # drag
            self.fuelflow = np.array([])  # fuel flow

            # Performance limits per aircraft
            self.vmin = np.array([])
            self.vmax = np.array([])


    @timed_function(name="performance", dt=settings.performance_dt, manual=True)
    def update(self, dt=settings.performance_dt):
        """implement this method """
        pass

    def limits(self, intent_v, intent_vs, intent_h, ax):
        """implement this method """
        return intent_v, intent_vs, intent_h

    def currentlimits(self):
        """implement this method """
        # Get current kinematic performance envelop of all aircraft
        pass

    def acceleration(self):
        ''' Default aircraft acceleration is 2 m/s2. '''
        return 2.0

    @command(name='ENG')
    def engchange(self, acid : 'acid', engine_id : 'txt' = ''):
        """ Specify a different engine type for aircraft 'acid' """
        return False, 'The currently selected performance model doesn\'t support engine changes.'

    @command(name='PERFSTATS', aliases=('PERFINFO', 'PERFDATA'))
    def show_performance(self, acid : 'acid'):
        """ Show aircraft perfromance parameters for aircraft 'acid' """
        return False, 'The currently selected performance model doesn\'t provide this function.'


    @staticmethod
    @command(name='PERF')
    def setmethod(name: 'txt' = ''):
        ''' Select a Performance implementation. '''
        # Get a dict of all registered Performance models
        methods = PerfBase.derived()
        names = ['OFF' if n == 'PERFBASE' else n for n in methods]

        if not name:
            curname = 'OFF' if PerfBase.selected() is PerfBase else PerfBase.selected().__name__
            return True, f'Current Performance model: {curname}' + \
                         f'\nAvailable performance models: {", ".join(names)}'
        # Check if the requested method exists
        if name == 'OFF':
            PerfBase.select()
            return True, 'Performance model turned off.'
        method = methods.get(name, None)
        if method is None:
            return False, f'{name} doesn\'t exist.\n' + \
                          f'Available performance models: {", ".join(names)}'

        # Select the requested method
        method.select()
        return True, f'Selected {method.__name__} as performance model.'
