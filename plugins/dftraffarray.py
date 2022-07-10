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
            self.npassengers = pd.DataFrame({
                                            'a': pd.Series(dtype='int'),
                                            'b': pd.Series(dtype='str'),
                                            'c': pd.Series(dtype='float')
                                            })

            self.geom_fun = gpd.GeoDataFrame({
                                            'a': pd.Series(dtype='int'),
                                            'geometry': gpd.GeoSeries(dtype=object),
                                            'c': pd.Series(dtype='float')
                                            })

    def create(self, n=1):
        ''' This function gets called automatically when new aircraft are created. '''
        # Don't forget to call the base class create when you reimplement this function!
        super().create(n)
        # After base creation we can change the values in our own states for the new aircraft

        self.npassengers.loc[-n:, 'a'] = 5
        self.npassengers.loc[-n:, 'b'] = 'test'
        self.npassengers.loc[-n:, 'c'] = 10.0