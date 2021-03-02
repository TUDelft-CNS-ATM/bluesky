""" ORCA algorithm implementation by Andrei Badea"""

from bluesky.traffic.asas import ConflictResolution
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import bluesky as bs
import numpy as np
import itertools
det = np.linalg.det

def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'ORCA',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim'
    }

    return config

class ORCA(ConflictResolution): 
    # Define some variables
    def __init__(self):
        super().__init__()
        # The directions of the velocity obstacle boundaries in unit vector form
        # useful for projecting relative velocity on them
        self.left_cutoff_leg_dir = np.array([])
        self.right_cutoff_leg_dir =np.array([])
        
    def resolve(self, conf, ownship, intruder):
        # Total number of aircraft
        ntraf = ownship.ntraf
        
        # Make a copy of traffic data, track and ground speed
        newtrack    = np.copy(ownship.trk)
        newgscapped = np.copy(ownship.gs)
        
        # Iterate over all aircraft
        for idx in np.arange(ntraf):
            # Calculate ORCA only for aircraft in conflict
            if conf.inconf[idx]:
                # Find the indices for the conflict pairs
                idx_pairs = self.pairs(conf, ownship, intruder, idx)
                
                # Find ORCA solution for aircraft 'idx'
                trk_new, gs_new = self.ORCA(conf, ownship, intruder, idx, idx_pairs)
                
                # Write the new velocity and track of aircraft 'idx' to traffic data
                newtrack[idx]    = trk_new
                newgscapped[idx] = gs_new           
        
        # For now the implementation is 2D, let the autopilot do its thing
        vscapped       = ownship.ap.vs
        alt            = ownship.ap.alt 
        
        return newtrack, newgscapped, vscapped, alt


    def ORCA(self, conf, ownship, intruder, idx, idx_pairs):
        
        # Extract ownship data
        gs_own = np.array([ownship.gseast[idx], ownship.gsnorth[idx]])# [m/s]
        
        lines = []
        # Go through all conflict pairs for aircraft "idx", basically take
        # intruders one by one
        for i, idx_pair in enumerate(idx_pairs):
            # Extract conflict bearing and distance information
            qdr = conf.qdr[idx_pair]
            dist= conf.dist[idx_pair]
            
            # Determine the index of the intruder
            idx_intruder = intruder.id.index(conf.confpairs[idx_pair][1])
            
            # Determine the optimal change in velocity for this intruder. ORCA
            # uses information such as lookahead time and resolution update
            # time to make the solution better. The lookahead time is used to
            # cut off the VO cone. The asas update time (dt) is used to find the
            # best solution for the next time step in case of LOS.
            dv, n = self.get_avoidance_velocity(conf, ownship, intruder, qdr, dist, idx
                    , idx_intruder, bs.settings.asas_dtlookahead, bs.settings.asas_dt)
            
            # Solutions are cooperative
            factor = 0.5
            
            # Create a line corresponding to where the new velocity should be
            # in order to evade this intruder.
            line = Line(gs_own + dv * factor, n)
            lines.append(line)
            
        # Take all lines for all intruders, and find an optimal point such that
        # all intruders are evaded, and velocity change is minimised
        fail_index, vel = self.halfplane_optimize(lines, gs_own)     
        
        # Check if optimisation failed
        if fail_index < len(lines)-1:
            # Calculate velocity using the safest solution algorithm
            vel = self.safest_solution(lines, fail_index, vel)
        
        # Convert optimal velocity vector to heading and ground speed change
        trk_new = (np.arctan2(vel[0],vel[1])*180/np.pi) % 360
        gs_new = np.sqrt(vel[0]**2 + vel[1]**2)
        
        # print('HDG command is %s for %s, changed from %s.'
        #       % (str(trk_new), str(ownship.id[idx]), str(ownship.trk[idx])))
        # print('GS  command is %s for %s, changed from %s.' 
        #       % (str(gs_new), str(ownship.id[idx]), str(ownship.gs[idx])))
        
        return trk_new, gs_new
    
    def pairs(self, conf, ownship, intruder, idx):
        '''Returns the indices of conflict pairs that involve aircraft idx
        '''
        idx_pairs = np.array([], dtype = int)
        for idx_pair, pair in enumerate(conf.confpairs):
            if (ownship.id[idx] == pair[0]):
                idx_pairs = np.append(idx_pairs, idx_pair)
        return idx_pairs
    
    # Modification of implementation of ORCA for robots by Mak Nazecic-Andrlon
    # Copyright (c) 2013 Mak Nazecic-Andrlon
    # pyorca   
    def get_avoidance_velocity(self, conf, ownship, intruder, qdr, dist, idx, idx_intruder, t, dt):
        """Get the smallest relative change in velocity between agent and collider
        that will get them onto the boundary of each other's velocity obstacle
        (VO), and thus avert collision."""
    
        # Convert qdr from degrees to radians
        qdr = np.radians(qdr)

        # Relative position vector between ownship and intruder
        x = np.array([np.sin(qdr)*dist, np.cos(qdr)*dist])
        
        # Relative velocity vector between ownship and intruder
        v = (np.array([ownship.gseast[idx], ownship.gsnorth[idx]]) - \
             np.array([intruder.gseast[idx_intruder], intruder.gsnorth[idx_intruder]]))
        
        # Take minimum separation and multiply it with safely factor
        r = conf.rpz * self.resofach
    
        x_len_sq = self.norm_sq(x)
    
        if x_len_sq >= r * r:
            # The centre of the x/t circle with radius r/t does not give the
            # corrent cut-off line for deciding whether to find the solution
            # on the circle or the leg, we need to build the geometry that allows
            # us to determine which is the shortest change in velocity
    
            if self.on_circle(v, x, r, t):
                # v is closer to the cutoff circle than to a VO leg
                # Project V on cutoff circle
                w = v - x/t
                # Find change in velocity
                u = self.normalized(w) * r/t - w
                n = self.normalized(w)
            else: 
                # v is closer to a VO leg than to the cutoff circle
                # Find on which side w.r.t x does v lie to know on which leg
                # to project. Also, ORCA has a bias to the left, so in case
                # the cross product is exactly 0, then we'll project on the left.
                if np.cross(v,x) > 0:
                    # Retrieve unit leg direction vector
                    leg_dir= self.right_cutoff_leg_dir
                    # Retrieve the normal unit vector towards the exterior 
                    # of the VO
                    n = self.perp_right(leg_dir)
                else:
                    # Retrieve unit leg direction vector
                    leg_dir = self.left_cutoff_leg_dir
                    # Retrieve the normal unit vector towards the exterior 
                    # of the VO
                    n = self.perp_left(leg_dir)
                
                # Project v on leg line
                protv = leg_dir * np.dot(v, leg_dir)
                # Find velocity difference
                u = protv - v
                
        else:
            # We're already intersecting. Pick the closest velocity to our
            # velocity that will get us out of the collision within the next
            # timestep.
            w = v - x/dt
            u = self.normalized(w) * r/dt - w
            n = self.normalized(w)
        return u, n
    
    
    def halfplane_optimize(self, lines, optimal_point):
        """Find the point closest to optimal_point in the intersection of the
        closed half-planes defined by lines which are in Hessian normal form
        (point-normal form)."""
        # We implement the quadratic time (though linear expected given randomly
        # permuted input) incremental half-plane intersection algorithm as laid
        # out in http://www.mpi-inf.mpg.de/~kavitha/lecture3.ps
        point = optimal_point
        fail_index = len(lines)-1
        for i, line in enumerate(lines):
            # If this half-plane already contains the current point, all is well.
            if np.dot(point - line.point, line.direction) >= 0:
                # assert False, point
                continue
    
            # Otherwise, the new optimum must lie on the newly added line. Compute
            # the feasible interval of the intersection of all the lines added so
            # far with the current one.
            prev_lines = itertools.islice(lines, i)
            feasible, left_dist, right_dist = self.line_halfplane_intersect(line, prev_lines)
            
            if feasible == False:
                fail_index = i
                return fail_index, point
            
            # Avoid program stopping because of the infesible error
            # try:
            #     left_dist, right_dist = self.line_halfplane_intersect(line, prev_lines)
            # except:
            #     print('Stopped at %s out of %s.' % (str(i), str(len(lines))))
            #     break
    
            # Now project the optimal point onto the line segment defined by the
            # the above bounds. This gives us our new best point.
            point = self.point_line_project(line, optimal_point, left_dist, right_dist)
        return fail_index, point

    def point_line_project(self, line, point, left_bound, right_bound):
        """Project point onto the line segment defined by line, which is in
        point-normal form, and the left and right bounds with respect to line's
        anchor point."""
        # print("left_bound=%s, right_bound=%s" % (left_bound, right_bound))
        new_dir = self.perp(line.direction)
        # print("new_dir=%s" % new_dir)
        proj_len = np.dot(point - line.point, new_dir)
        # print("proj_len=%s" % proj_len)
        clamped_len = np.clip(proj_len, left_bound, right_bound)
        # print("clamped_len=%s" % clamped_len)
        return line.point + new_dir * clamped_len
    
    def line_halfplane_intersect(self, line, other_lines):
        """Compute the signed offsets of the interval on the edge of the
        half-plane defined by line that is included in the half-planes defined by
        other_lines.
        The offsets are relative to line's anchor point, in units of line's
        direction.
        """
        # We use the line intersection algorithm presented in
        # http://stackoverflow.com/a/565282/126977 to determine the intersection
        # point. "Left" is the negative of the canonical direction of the line.
        # "Right" is positive.
        left_dist = float("-inf")
        right_dist = float("inf")
        feasible = True
        for prev_line in other_lines:
            num1 = np.dot(prev_line.direction, line.point - prev_line.point)
            den1 = det((line.direction, prev_line.direction))
            # num2 = det((perp(prev_line.direction), line.point - prev_line.point))
            # den2 = det((perp(line.direction), perp(prev_line.direction)))
    
            # assert abs(den1 - den2) < 1e-6, (den1, den2)
            # assert abs(num1 - num2) < 1e-6, (num1, num2)
    
            num = num1
            den = den1
    
            # Check for zero denominator, since ZeroDivisionError (or rather
            # FloatingPointError) won't necessarily be raised if using numpy.
            if den == 0:
                # The half-planes are parallel.
                if num < 0:
                    # The intersection of the half-planes is empty; there is no
                    # solution.
                    # raise InfeasibleError
                    feasible = False
                    return feasible, left_dist, right_dist 
                else:
                    # The *half-planes* intersect, but their lines don't cross, so
                    # ignore.
                    continue
    
            # Signed offset of the point of intersection, relative to the line's
            # anchor point, in units of the line's direction.
            offset = num / den
            if den > 0:
                # Point of intersection is to the right.
                right_dist = min((right_dist, offset))
            else:
                # Point of intersection is to the left.
                left_dist = max((left_dist, offset))
    
            if left_dist > right_dist:
                # The interval is inconsistent, so the feasible region is empty.
                # raise InfeasibleError
                feasible = False
                return feasible, left_dist, right_dist
        return feasible, left_dist, right_dist
    
    def perp_left(self, a):
        ''' Gives perpendicular unit vector pointing to the "left" (+90 deg)
        for vector "a" '''
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b/np.linalg.norm(b)

    def perp_right(self, a):
        ''' Gives perpendicular unit vector pointing to the "right" (-90 deg)
        for vector "a" '''
        b = np.empty_like(a)
        b[0] = a[1]
        b[1] = -a[0]
        return b/np.linalg.norm(b)
    
    def left_cutoff_leg(self, x, r, t):
        '''Gives the cutoff point of the left leg.'''
        # Find vector that describes radius
        r_vec = self.perp_left(x) * r
        # Find the big left leg vector
        left_leg = x + r_vec
        # Find the left leg direction
        left_cutoff_leg_dir = self.normalized(left_leg)
        # Save this for later
        self.left_cutoff_leg_dir = left_cutoff_leg_dir
        # Find the length of the left cutoff leg
        left_cutoff_leg = np.sqrt(self.norm_sq(x/t) - (r/t)*(r/t))
        # Return left cutoff vector
        return left_cutoff_leg * left_cutoff_leg_dir
        
    def right_cutoff_leg(self, x, r, t):
        '''Gives the cutoff point of the right leg.'''
        # Find vector that describes radius
        r_vec = self.perp_right(x) * r
        # Find the big right leg vector
        right_leg = x + r_vec
        # Find the right leg direction
        right_cutoff_leg_dir = self.normalized(right_leg)
        # Save this for later
        self.right_cutoff_leg_dir = right_cutoff_leg_dir
        # Find the length of the right cutoff leg
        right_cutoff_leg = np.sqrt(self.norm_sq(x/t) - (r/t)*(r/t))
        # Return right cutoff vector
        return right_cutoff_leg * right_cutoff_leg_dir
    
    def on_circle(self, v, x, r, t):
        '''Outputs if point is closer to cutoff circle boundary than it is to
        the VO leg, which means we need to project velocity on circle'''
        # First we need the 4 points of the polygon in which we're looking for
        # a point in
        
        # First and third points are the origin and the cutoff circle 
        # centre (x/t)
        point1 = Point(0,0)
        point3 = Point(x/t)
        
        # Point 2 and Point 4 are the connection points between VO legs and the
        # circle, which can be found by using the normalized direction vectors
        # multiplied by the length of the leg
        
        # Find cutoff vectors
        left_cutoff = self.left_cutoff_leg(x,r,t)
        right_cutoff = self.right_cutoff_leg(x,r,t)
        
        # Create points
        point2 = Point(left_cutoff)
        point4 = Point(right_cutoff)
        
        # Create polygon
        poly = Polygon([point1, point2, point3, point4])
        
        return poly.contains(Point(v))
    
    def safest_solution(self, lines, fail_index, fail_result):
        ''' This runs if the optimisation failed, and tries to implement
        the safest solution that violates the limits imposed by VOs the least'''
        distance = 0
        result = fail_result
        # Start adjusting solution starting with the first unfeasible one
        for i in np.arange(fail_index, len(lines)):
            
            # Check if solution is truly infesible. By default first one is as
            # distance is 0.
            if (det((lines[i].direction, lines[i].point - result)) > distance):
                proj_lines = []
                
                # Go through all previous limits in order to minimize distance
                # with all
                for j in np.arange(0, i):
                    determinant = det((lines[i].direction, lines[j].direction));
                    newpoint = np.array([0,0])
                    newdir = np.array([0,0])
                    
                    # Bascially, the lines are *almost* parallel, 1e-5 is taken
                    # as a criteria in case of floating point error. Value taken
                    # from the original ORCA cpp implementation
                    if determinant < 1e-5:
                        if lines[i].direction @ lines[j].direction > 0:
                            # Lines point in the same direction, line j is redundant
                            continue
                        else:
                            # Lines are opposite, closest point to both sits
                            # in the middle
                            newpoint = 0.5 * (lines[i].point + lines[j].point)
                            
                    else:
                        # Compute closest point to both lines
                        newpoint = lines[i].point + (det((lines[j].direction, \
                                  lines[i].point - lines[j].point)) \
                                 / determinant) * lines[i].direction
                        
                    newdir = self.normalized(lines[j].direction - lines[i].direction)
                    newline = Line(newpoint, newdir)
                    proj_lines.append(newline)
                
                # Compute the new velocity using the original optimisation
                # algorithm
                feasible, vel = self.halfplane_optimize(proj_lines, result)
                
                if feasible:
                    # In theory this should always happen
                    result = vel
                    
                # Update distance
                distance = det((lines[i].direction, lines[i].point - result))
                
        return result
                
    def perp(self, a):
        return np.array((a[1], -a[0]))
    
    def norm_sq(self, x):
        return np.dot(x, x)
    
    def norm(self,x):
        return np.sqrt(self.norm_sq(x))
    
    def normalized(self, x):
        l = self.norm_sq(x)
        assert l > 0, (x, l)
        return x / np.sqrt(l)
    
    def angle(self, a, b):
        ''' Find non-directional angle between vector a and b'''
        return np.arccos(np.dot(a,b)/(self.norm(a) * self.norm(b)))
    
    def dist_sq(self, a, b):
        return self.norm_sq(b - a)
    
        
class InfeasibleError(RuntimeError):
    """Raised if an LP problem has no solution."""
    pass


class Line(object):
    """A line in space."""
    def __init__(self, point, direction):
        super(Line, self).__init__()
        self.point = np.array(point)
        self.direction = self.normalized(np.array(direction))

    def __repr__(self):
        return "Line(%s, %s)" % (self.point, self.direction)
    
    def normalized(self, x):
        l = self.norm_sq(x)
        assert l > 0, (x, l)
        return x / np.sqrt(l)
    
    def norm_sq(self, x):
        return np.dot(x, x)