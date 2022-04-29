""" This plugin load the graph for the USEPE project. """
# Import the global bluesky objects. Uncomment the ones you need
import configparser
import datetime
import os
import pickle

from bluesky import core, traf, stack, sim  # , settings, navdb,  scr, tools
from usepe.city_model.dynamic_segments import dynamicSegments
from usepe.city_model.scenario_definition import createFlightPlan
from usepe.city_model.strategic_deconfliction import initialPopulation, deconflictedPathPlanning
from usepe.city_model.utils import read_my_graphml, layersDict
import pandas as pd


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2022'


usepeconfig = None
usepegraph = None
usepesegments = None
usepestrategic = None
usepeflightplans = None
usepedronecommands = None


# ## Initialisation function of your plugin. Do not change the name of this
# ## function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate the UsepeLogger entity
    global usepeconfig

    global usepegraph
    global usepesegments
    global usepeflightplans
    global usepestrategic
    global usepedronecommands

    config_path = r"C:\workspace3\scenarios-USEPE\scenario\USEPE\exercise_1\settings_exercise_1_reference.cfg"
    graph_path = r"C:\workspace3\scenarios-USEPE\scenario\USEPE\exercise_1\data\testing_graph.graphml"
    segment_path = r"C:\workspace3\scenarios-USEPE\scenario\USEPE\exercise_1\data\offline_segments.pkl"
    fligh_plan_csv_path = r"C:\workspace3\scenarios-USEPE\scenario\USEPE\exercise_1\data\delivery_testing.csv"

    initial_time = 0  # seconds
    final_time = 7200  # seconds

    usepeconfig = configparser.ConfigParser()
    usepeconfig.read( config_path )

    usepegraph = UsepeGraph( graph_path )
    usepesegments = UsepeSegments( segment_path )
    usepestrategic = UsepeStrategicDeconfliction( initial_time, final_time )
    usepeflightplans = UsepeFlightPlan( fligh_plan_csv_path )
    usepedronecommands = UsepeDroneCommands()

    # Configuration parameters
    config = {
        'plugin_name': 'USEPE',
        'plugin_type': 'sim',
        'update_interval': 1.0,

        # The update function is called after traffic is updated.
        'update': update,

        # The preupdate function is called before traffic is updated.
        'preupdate': preupdate,

        # Reset
        'reset': reset }

    # init_plugin() should always return a configuration dict.
    return config


def update():
    usepegraph.update()
    usepesegments.update()
    usepestrategic.update()
    usepeflightplans.update()
    usepedronecommands.update()
    return


def preupdate():
    usepegraph.preupdate()
    usepesegments.preupdate()
    usepestrategic.preupdate()
    usepeflightplans.preupdate()
    usepedronecommands.preupdate()
    return


def reset():
    usepegraph.reset()
    usepesegments.reset()
    usepestrategic.reset()
    usepeflightplans.reset()
    usepedronecommands.reset()
    return


class UsepeGraph( core.Entity ):
    ''' UsepeGraph new entity for BlueSky '''

    def __init__( self, graph_path ):
        super().__init__()

        self.graph = graph_path
        self.graph = read_my_graphml( graph_path )
        self.test = 1
        self.layers_dict = layersDict( usepeconfig )

    def update( self ):  # Not used
        # stack.stack( 'ECHO Example update: creating a graph' )
        return

    def preupdate( self ):  # Not used
        # print( self.graph.nodes() )
        return

    def reset( self ):  # Not used
        return


class UsepeSegments( core.Entity ):
    ''' UsepeSegments new entity for BlueSky '''

    def __init__( self, segment_path ):
        super().__init__()

        with open( segment_path, 'rb' ) as f:
            self.segments = pickle.load( f )

        #### Remove
        for key in self.segments:
            if self.segments[key]['speed'] == 0:
                self.segments[key]['speed'] = 5
                self.segments[key]['capacity'] = 1
                self.segments[key]['updated'] = True
        usepegraph.graph, self.segments = dynamicSegments( usepegraph.graph, usepeconfig, self.segments, deleted_segments=None )
        #####

    def dynamicSegments( self ):
        """
        Here include the function that updated the segmetns
        """
        updated = False
        segments = self.segments
        return updated, segments

    def update( self ):  # Not used
        # stack.stack( 'ECHO Example update: import segments' )
        return

    def preupdate( self ):  # Not used
        updated, self.segments = self.dynamicSegments()

        if updated:
            # Perform all the activities associated to the segmetns update

            # TODO
            pass

        return

    def reset( self ):  # Not used
        return


class UsepeStrategicDeconfliction( core.Entity ):
    ''' UsepeStrategicDeconfliction new entity for BlueSky '''

    def __init__( self, initial_time, final_time ):
        super().__init__()
        self.initial_time = initial_time
        self.final_time = final_time
        self.users = initialPopulation( usepesegments.segments, self.initial_time, self.final_time )

        self.delivery_drones = 0
        self.background_drones = 0

    def strategicDeconflictionDrone( self, df_row ):
        row = df_row.iloc[0]
        orig = [row['origin_lon'], row['origin_lat'], row['origin_alt'] ]
        dest = [row['destination_lon'], row['destination_lat'], row['destination_alt'] ]
        departure_time = row['departure_s']

        if row['purpose'] == 'delivery':
            name = row['purpose'] + str( self.delivery_drones )
            self.delivery_drones += 1
        elif row['purpose'] == 'background':
            name = row['purpose'] + str( self.background_drones )
            self.background_drones += 1

        if row['drone'] == 'M600':
            v_max = 18
            vs_max = 5
            safety_volume_size = 1

        ac = {'id': name, 'type': row['drone'], 'accel': 3.5, 'v_max': v_max, 'vs_max': vs_max,
              'safety_volume_size': safety_volume_size }
        users, route, delayed_time = deconflictedPathPlanning( orig, dest, departure_time,
                                                               usepegraph.graph, self.users,
                                                               self.initial_time, self.final_time,
                                                               usepesegments.segments, usepeconfig, ac )

        df_row['delayed_time'] = delayed_time
        usepeflightplans.route_dict[name] = route
        usepeflightplans.ac_dict[name] = ac
        df_row['ac'] = name
        usepeflightplans.flight_plan_df_processed = pd.concat( 
            [usepeflightplans.flight_plan_df_processed, df_row] ).sort_values( by='delayed_time' )

        self.users = users

    def update( self ):  # Not used
        return

    def preupdate( self ):  # Not used
        return

    def reset( self ):  # Not used
        return


class UsepePathPlanning( core.Entity ):
    ''' UsepePathPlanning new entity for BlueSky '''

    def __init__( self ):
        super().__init__()

    def update( self ):  # Not used
        return

    def preupdate( self ):  # Not used
        return

    def reset( self ):  # Not used
        return


class UsepeDroneCommands( core.Entity ):
    ''' UsepeDroneCommands new entity for BlueSky '''

    def __init__( self ):
        super().__init__()

    def createDroneCommands( self, row ):
        route = usepeflightplans.route_dict[row['ac']]
        ac = usepeflightplans.ac_dict[row['ac']]
        # departure_time = str( datetime.timedelta( seconds=row['delayed_time'] ) )

        departure_time = str( datetime.timedelta( seconds=0 ) )  # Relative time is considered
        G = usepegraph.graph
        layers_dict = usepegraph.layers_dict

        scenario_path = r'.\scenario\usepe\temp\scenario_traffic_drone_{}.scn'.format( ac['id'] )
        print( scenario_path )
        scenario_file = open( scenario_path, 'w' )

        createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )

        scenario_file.close()

        stack.stack( 'PCALL {} REL'.format( '.' + scenario_path[10:] ) )

    def droneTakeOff( self ):
        while not usepeflightplans.flight_plan_df_processed[usepeflightplans.flight_plan_df_processed['delayed_time'] <= sim.simt].empty:
            row = usepeflightplans.flight_plan_df_processed.iloc[0]
            self.createDroneCommands( row )
            usepeflightplans.flight_plan_df_processed = usepeflightplans.flight_plan_df_processed.drop( usepeflightplans.flight_plan_df_processed.index[0] )

    def update( self ):  # Not used
        return

    def preupdate( self ):
        self.droneTakeOff()
        return

    def reset( self ):  # Not used
        return


class UsepeFlightPlan( core.Entity ):
    ''' UsepeFlightPlan new entity for BlueSky '''

    def __init__( self, fligh_plan_csv_path ):
        super().__init__()

        self.flight_plan_df = pd.read_csv( fligh_plan_csv_path, index_col=0 ).sort_values( by=['planned_time_s'] )
        self.flight_plan_df_buffer = self.flight_plan_df[self.flight_plan_df['planned_time_s'] >= sim.simt]
        self.flight_plan_df_processed = pd.DataFrame( columns=list( self.flight_plan_df.columns ) +
                                                      ['delayed_time'] + ['ac'] )


        self.route_dict = {}
        self.ac_dict = {}
        # self.processFlightPlans()

    def processFlightPlans( self ):
        while not self.flight_plan_df_buffer[self.flight_plan_df_buffer['planned_time_s'] <= sim.simt].empty:
            df_row = self.flight_plan_df_buffer.iloc[[0]]
            print( df_row )
            usepestrategic.strategicDeconflictionDrone( df_row )

            self.flight_plan_df_buffer = self.flight_plan_df_buffer.drop( self.flight_plan_df_buffer.index[0] )

    def update( self ):  # Not used
        return

    def preupdate( self ):
        self.processFlightPlans()
        return

    def reset( self ):  # Not used
        return
