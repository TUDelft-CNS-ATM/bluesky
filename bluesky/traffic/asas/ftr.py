''' Resume navigation based on the Free-To-Revert (FTR) criterion. '''
import numpy as np

import bluesky as bs
from bluesky.stack import command
from bluesky.traffic.asas import ResumeNavigation


class FTR(ResumeNavigation):
    ''' Resume navigation using the Free-To-Revert method.

        This is from Wouter Schaberg's Thesis
        https://repository.tudelft.nl/record/uuid:529b6868-f0b4-49e3-94c3-82e52ebe0c7d

        FTR resumes navigation by calculating whether the 'target' distance
        at CPA is bigger than the protected zone. The word target there is used
        because it is calculated from the desired velocity (from autopilot)
        of ownship, against:

        1. intruder's current velocity, criterion 1
        2. intruder's desired velocity (from autopilot), criterion 2

        However, typically aircraft don't share intent these days so the default
        mode of criterion 2 is ASSUMED -> the desired velocity is equal to the 
        velocit when the two aircraft first detected in conflict.

        FTRINTENT OFF sets the criterion 2 to always TRUE, making it inactive
        FTRINTENT DECLARED enables the criterion 2 to use the autopilot data of intruders
    '''

    def __init__(self):
        super().__init__()
        self.assumed = dict()

        self.intent = 'ASSUMED'

    def selectedmsg(self):
        '''
            Line RESNAV appends to its own confirmation, naming the FTRINTENT
            method criterion 2 will use.
        '''
        return f'Selected {self.intent} as FTRINTENT method.'

    def reset(self):
        super().reset()
        self.assumed.clear()
        self.intent = 'ASSUMED'

    def resumenav(self, conf, ownship, intruder):
        '''
            Decide for each aircraft in the conflict list whether the ASAS
            should be followed or not, based on whether reverting to the
            autopilot velocity would still clear the intruder.
        '''
        # Record the assumed intent on the tick a pair becomes active, before
        # the pair is merged into resopairs, so it is the velocity the intruder
        # was flying when this conflict was first detected.
        for pair in conf.confpairs:
            if pair not in self.resopairs:
                idx2 = bs.traf.id2idx(pair[1])
                if idx2 >= 0:
                    self.assumed[pair] = (intruder.gseast[idx2], intruder.gsnorth[idx2])

        # Add new conflicts to resopairs
        self.resopairs.update(conf.confpairs)

        # Conflict pairs to be deleted
        delpairs = set()
        changeactive = dict()

        # Look at all conflicts, also the ones that are solved but for which
        # reverting would not yet be safe
        for conflict in self.resopairs:
            idx1, idx2 = bs.traf.id2idx(conflict)
            # If the ownship aircraft is deleted remove its conflict from the list
            if idx1 < 0:
                delpairs.add(conflict)
                continue

            if idx2 >= 0:
                # Distance vector using flat earth approximation
                re = 6371000.
                dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) *
                                      np.cos(0.5 * np.radians(intruder.lat[idx2] +
                                                              ownship.lat[idx1])),
                                      np.radians(intruder.lat[idx2] - ownship.lat[idx1])])

                rpz = np.max(conf.rpz[[idx1, idx2]])

                # The ownship's own side of both criteria: its desired velocity,
                # i.e. what the autopilot would fly the moment ASAS lets go.
                vown = self.desired_spd_trk(ownship, idx1)

                # Criterion 1: the intruder holds its current, observed velocity
                vintr = np.array([intruder.gseast[idx2], intruder.gsnorth[idx2]])
                ftr = self.clears(dist, vintr - vown, rpz)

                # Criterion 2: the intruder reverts to its desired velocity too
                if ftr and self.intent != 'OFF':
                    if self.intent == 'ASSUMED':
                        vrevert = self.assumed.get(conflict)
                        vrevert = None if vrevert is None else np.array(vrevert)
                    else:
                        vrevert = self.desired_spd_trk(intruder, idx2)
                    if vrevert is not None:
                        ftr = self.clears(dist, vrevert - vown, rpz)

            # Keep resolving for ownship while reverting would not clear the
            # intruder. A pair still in horizontal LOS never clears: with the
            # pair diverging the criterion falls back to the current distance,
            # which is inside rpz by definition.
            if idx2 >= 0 and not ftr:
                # Enable ASAS for this aircraft
                changeactive[idx1] = True
            else:
                # Switch ASAS off for ownship if there are no other conflicts
                # that this aircraft is involved in.
                changeactive[idx1] = changeactive.get(idx1, False)
                # If reverting is safe, remove the pair from the resopairs list
                delpairs.add(conflict)

        for idx, active in changeactive.items():
            # Loop a second time: this is to avoid that ASAS resolution is
            # turned off for an aircraft that is involved simultaneously in
            # multiple conflicts, where the first, but not all conflicts are
            # resolved.
            bs.traf.cr.active[idx] = active
            if not active:
                # Waypoint recovery after conflict: Find the next active waypoint
                # and send the aircraft to that waypoint.
                iwpid = bs.traf.ap.route[idx].findact(idx)
                if iwpid != -1:  # To avoid problems if there are no waypoints
                    bs.traf.ap.route[idx].direct(
                        idx, bs.traf.ap.route[idx].wpname[iwpid])

        # Remove pairs that are free to revert or have deleted aircraft
        self.resopairs -= delpairs
        for conflict in delpairs:
            self.assumed.pop(conflict, None)

    @staticmethod
    def clears(dist, vrel, rpz):
        '''
            Whether relative motion (position dist, velocity vrel) keeps the
            closest point of approach beyond rpz.

            Uses the *forward* closest approach: a pair that is already
            diverging (tcpa <= 0), or that has no relative motion, is judged by
            where it is now, not by a hypothetical closest approach in the past.
        '''
        vrel2 = np.dot(vrel, vrel)
        dist2 = np.dot(dist, dist)
        if vrel2 < 1e-9:
            return dist2 > rpz * rpz
        tcpa = -np.dot(dist, vrel) / vrel2
        if tcpa <= 0.0:
            return dist2 > rpz * rpz
        return max(0.0, dist2 - tcpa * tcpa * vrel2) > rpz * rpz

    @staticmethod
    def desired_spd_trk(traf, idx):
        '''
            Ground velocity [east, north] the autopilot of aircraft idx would
            fly right now: its desired track at its desired airspeed, corrected
            for wind.

            ap.trk is a desired *ground track*, so the wind triangle is solved
            for the ground speed that makes that track good at ap.tas. Without
            wind this reduces exactly to ap.tas along ap.trk.
        '''
        trk = np.radians(traf.ap.trk[idx])
        along = np.array([np.sin(trk), np.cos(trk)])
        wind = np.array([traf.windeast[idx], traf.windnorth[idx]])
        # Split the wind into a component along the desired track and one across
        # it; the aircraft crabs to cancel the crosswind, what is left over
        # plus the tailwind component is the ground speed.
        wpar = np.dot(wind, along)
        wperp = wind[0] * along[1] - wind[1] * along[0]
        tas = traf.ap.tas[idx]
        if abs(wperp) >= tas:
            # Crosswind exceeds the airspeed: the track cannot be made good.
            # Fall back to the current ground velocity.
            return np.array([traf.gseast[idx], traf.gsnorth[idx]])
        return (wpar + np.sqrt(tas * tas - wperp * wperp)) * along

    @command(name='FTRINTENT')
    def setintent(self, mode: 'txt' = ''):
        '''
            Select how FTR obtains the intruder's desired velocity for its
            second (intent-based) criterion.

            OFF:      skip the second criterion.
            ASSUMED:  the velocity the intruder was flying when the conflict
                      was first detected (inferred, no intent sharing needed).
                      This is the default.
            DECLARED: read the intruder's autopilot (perfect intent sharing).
        '''
        modes = ('OFF', 'ASSUMED', 'DECLARED')
        if not mode:
            return True, f'Current FTRINTENT method: {self.intent}' + \
                         f'\nAvailable FTRINTENT methods: {", ".join(modes)}'
        if mode.upper() not in modes:
            return False, f'{mode} doesn\'t exist.\n' + \
                          f'Available FTRINTENT methods: {", ".join(modes)}'
        self.intent = mode.upper()
        return True, self.selectedmsg()
