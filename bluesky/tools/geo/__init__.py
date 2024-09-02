''' BlueSky functions for geographical calculations. '''
from bluesky import settings
from bluesky.tools.geo._geo import nm, magdec, initdecl_data, magdeccmd, kwikpos


# Register settings defaults
settings.set_variable_defaults(prefer_compiled=False)
if getattr(settings, 'prefer_compiled'):
    try:
        from bluesky.tools.geo._cgeo import (
            rwgs84,
            rwgs84_matrix,
            qdrdist,
            qdrdist_matrix,
            latlondist,
            latlondist_matrix,
            wgsg,
            qdrpos,
            kwikdist,
            kwikdist_matrix,
            kwikqdrdist,
            kwikqdrdist_matrix
        )
        print('Using compiled geo functions')
    except ImportError:
        from bluesky.tools.geo._geo import (rwgs84, rwgs84_matrix, qdrdist, 
                                        qdrdist_matrix, latlondist, latlondist_matrix,
                                        wgsg, qdrpos, kwikdist, kwikdist_matrix, 
                                        kwikqdrdist, kwikqdrdist_matrix)
        print('Could not load compiled geo functions, Using Python-based geo functions instead')
else:
    from bluesky.tools.geo._geo import (rwgs84, rwgs84_matrix, qdrdist, 
                                        qdrdist_matrix, latlondist, latlondist_matrix,
                                        wgsg, qdrpos, kwikdist, kwikdist_matrix, 
                                        kwikqdrdist, kwikqdrdist_matrix)
    print('Using Python-based geo functions')
