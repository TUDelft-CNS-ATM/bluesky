import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta

from bluesky import core, sim

def init_plugin():
    ''' Plugin initialisation function. '''
    example = DF_arrays()

    config = {
        'plugin_name':     'DFEXAMPLE',
        'plugin_type':     'sim',
        }

    return config


class DF_arrays(core.Entity):

    def __init__(self):
        super().__init__()

        with self.settrafarrays():
            self.flight_data = pd.DataFrame({
                                            'npassengers': pd.Series(dtype=int),
                                            'mean_fare': pd.Series(dtype=float),
                                            'departure_time': pd.Series(dtype=datetime),
                                            'arrival_time': pd.Series(dtype=datetime),
                                            'predicted_arrival_time': pd.Series(dtype=datetime),
                                            'flight_distance': pd.Series(dtype=float),
                                            'predicted_flight_distance': pd.Series(dtype=float),
                                            })

    def create(self, n=1):
        super().create(n)

        # Get some data at start of simulation
        self.flight_data.loc[-n:, 'npassengers'] = 10
        self.flight_data.loc[-n:, 'mean_fare'] = 100
        self.flight_data.loc[-n:, 'departure_time'] = datetime(2022, 1, 1) + timedelta(seconds = sim.simt)
        self.flight_data.loc[-n:, 'predicted_arrival_time'] = datetime(2022, 1, 1) + timedelta(seconds = sim.simt) + timedelta(seconds=randint(0, 100))
        self.flight_data.loc[-n:, 'predicted_flight_distance'] = randint(0, 100)

    
    def delete(self, idx):

        # Get arrival time and flight distance
        self.flight_data.loc[idx, 'arrival_time'] = self.flight_data.loc[idx, 'departure_time'] + timedelta(seconds = sim.simt)
        self.flight_data.loc[idx, 'flight_distance'] = traf.distflown

        # Call the actual delete function
        super().delete(idx)
