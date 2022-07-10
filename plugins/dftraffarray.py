import numpy as np
import pandas as pd
import geopandas as gpd

from bluesky import core 

def init_plugin():
    ''' Plugin initialisation function. '''
    example = DF_arrays()

    config = {
        'plugin_name':     'DFFUN',
        'plugin_type':     'sim',
        }

    return config


class DF_arrays(core.Entity):

    def __init__(self):
        super().__init__()

        with self.settrafarrays():
            self.passenger_data = pd.DataFrame({
                                            'fare': pd.Series(dtype=float),
                                            'age':  pd.Series(dtype=int),
                                            'connecting': pd.Series(dtype=bool)
                                            })

            self.geom_fun = gpd.GeoDataFrame({
                                            'a': pd.Series(dtype='int'),
                                            'geometry': gpd.GeoSeries(dtype=object),
                                            'c': pd.Series(dtype='float')
                                            })

    def create(self, n=1):
        super().create(n)

        self.passenger_data.loc[-n:, 'fare'] = 100
        self.passenger_data.loc[-n:, 'age'] = 20
        self.passenger_data.loc[-n:, 'connecting'] = True