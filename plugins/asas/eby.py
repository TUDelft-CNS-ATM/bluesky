''' Eby ConflictResolution implementation plugin. '''
import numpy as np
from bluesky.tools.aero import vtas2eas
from bluesky.traffic.asas import ConflictResolution


def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EBY',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim'
    }

    return config


class Eby(ConflictResolution):
    def resolve(self, conf, ownship, intruder):
        ''' Resolve all current conflicts '''
        # required change in velocity
        dv = np.zeros((ownship.ntraf, 3))

        for ((ac1, ac2), qdr, dist, tcpa, tLOS) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)
            if idx1 > -1 and idx2 > -1:
                dv_eby = self.Eby_straight(
                    ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2)
                dv[idx1] -= dv_eby
                dv[idx2] += dv_eby

        # now we have the change in speed vector for each aircraft.
        dv=np.transpose(dv)
        # the old speed vector, cartesian coordinates
        trkrad = np.radians(ownship.trk)
        v = np.array([np.sin(trkrad) * ownship.tas,\
                      np.cos(trkrad)*ownship.tas,\
                      ownship.vs])
        # the new speed vector
        newv = dv + v

        # the new speed vector in polar coordinates
        newtrack = (np.arctan2(newv[0,:], newv[1,:]) * 180 / np.pi) % 360
        newgs = np.sqrt(newv[0,:] ** 2 + newv[1,:] ** 2)
        neweas = vtas2eas(newgs, ownship.alt)

        # Cap the velocity
        neweascapped = np.maximum(ownship.perf.vmin, np.minimum(ownship.perf.vmax, neweas))

        return newtrack, neweascapped, newv[2, :], np.sign(newv[2, :]) * 1e5

    def Eby_straight(self, ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2):
        ''' 
            Resolution: Eby method assuming aircraft move straight forward,
            solving algebraically, only horizontally.
        '''
        # from degrees to radians
        qdr  = np.radians(qdr)
        # relative position vector
        d    = np.array([np.sin(qdr) * dist, \
                         np.cos(qdr) * dist, \
                         intruder.alt[idx2] - ownship.alt[idx1]])

        # find track in radians
        t1 = np.radians(ownship.trk[idx1])
        t2 = np.radians(intruder.trk[idx2])

        # write velocities as vectors and find relative velocity vector
        v1 = np.array([np.sin(t1) * ownship.tas[idx1], np.cos(t1) * ownship.tas[idx1], ownship.vs[idx1]])
        v2 = np.array([np.sin(t2) * intruder.tas[idx2], np.cos(t2) * intruder.tas[idx2], intruder.vs[idx2]])
        v = np.array(v2 - v1)
        # bear in mind: the definition of vr (relative velocity) is opposite to
        # the velocity vector in the LOS_nominal method, this just has consequences
        # for the derivation of tstar following Eby method, not more
        """
        intrusion vector:
        i(t)=self.hsep-d(t)
        d(t)=sqrt((d[0]+v[0]*t)**2+(d[1]+v[1]*t)**2)
        find max(i(t)/t)
        -write the equation out
        -take derivative, set to zero
        -simplify, take square of everything so the sqrt disappears (creates two solutions)
        -write to the form a*t**2 + b*t + c = 0
        -Solve using the quadratic formula
        """
        # These terms are used to construct a,b,c of the quadratic formula
        R2 = (conf.rpz * self.resofach) ** 2 # in meters
        d2 = np.dot(d, d) # distance vector length squared
        v2 = np.dot(v, v) # velocity vector length squared
        dv = np.dot(d, v) # dot product of distance and velocity

        # Solving the quadratic formula
        a = R2 * v2 - dv **2
        b = 2 * dv * (R2 - d2)
        c = R2 * d2 - d2 ** 2
        discrim = b ** 2 - 4 * a * c

        if discrim < 0: # if the discriminant is negative, we're done as taking the square root will result in an error
            discrim = 0
        time1 = (-b + np.sqrt(discrim)) / (2 * a)
        time2 = (-b - np.sqrt(discrim)) / (2 * a)

        #time when the size of the conflict is largest relative to time to solve
        tstar = min(abs(time1), abs(time2))

        #find drel and absolute distance at tstar
        drelstar = d + v * tstar
        dstarabs = np.linalg.norm(drelstar)
        #exception: if the two aircraft are on exact collision course
        #(passing eachother within 10 meter), change drelstar
        exactcourse = 10 #10 meter
        dif = exactcourse - dstarabs
        if dif > 0:
            vperp = np.array([-v[1], v[0], 0]) #rotate velocity 90 degrees in horizontal plane
            drelstar += dif * vperp / np.linalg.norm(vperp) #normalize to 10 m and add to drelstar
            dstarabs = np.linalg.norm(drelstar)

        #intrusion at tstar
        i = (conf.rpz * self.resofach) - dstarabs

        #desired change in the plane's speed vector:
        dv = i * drelstar / (dstarabs * tstar)
        return dv
