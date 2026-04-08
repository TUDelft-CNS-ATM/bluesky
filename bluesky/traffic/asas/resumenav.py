''' Resume navigation base class. '''
import bluesky as bs
from bluesky.core import Entity
from bluesky.stack import command


class ResumeNavigation(Entity, replaceable=True):
    ''' Base class for Resume Navigation implementations. '''
    def __init__(self):
        super().__init__()
        self.resopairs = set()  # Resolved conflicts that are still before CPA

    def reset(self):
        super().reset()
        self.resopairs.clear()

    def update(self, conf, ownship, intruder):
        ''' Perform an update step of the Resume Navigation implementation. '''
        if ResumeNavigation.selected() is not ResumeNavigation:
            self.resumenav(conf, ownship, intruder)

    def resumenav(self, conf, ownship, intruder):
        '''
            Decide for each aircraft in the conflict list whether the ASAS
            should be followed or not, based on if the aircraft pairs passed
            their CPA.

            This function should be reimplemented in a subclass for actual
            resume navigation logic. See for instance
            bluesky.traffic.asas.pastcpa.
        '''
        pass

    @staticmethod
    @command(name='RESNAV')
    def setmethod(name : 'txt' = ''):
        ''' Select a Resume Navigation method. '''
        # Get a dict of all registered resume navigation methods
        methods = ResumeNavigation.derived()
        names = ['OFF' if n == 'RESUMENAVIGATION' else n for n in methods]

        if not name:
            curname = 'OFF' if ResumeNavigation.selected() is ResumeNavigation \
                else ResumeNavigation.selected().__name__
            return True, f'Current resume navigation method: {curname}' + \
                         f'\nAvailable resume navigation methods: {", ".join(names)}'
        # Check if the requested method exists
        if name == 'OFF':
            ResumeNavigation.select()
            return True, 'Resume Navigation turned off.'
        method = methods.get(name, None)
        if method is None:
            return False, f'{name} doesn\'t exist.\n' + \
                          f'Available resume navigation methods: {", ".join(names)}'

        # Select the requested method
        method.select()
        return True, f'Selected {method.__name__} as resume navigation method.'
