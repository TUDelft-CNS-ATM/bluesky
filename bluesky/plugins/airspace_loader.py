""" OpenAIP Airspace Loader plugin for BlueSky.

    Reads airspace polygons from the OpenAIP airspaces.json file.

    Stack commands
    --------------
    LOADAIRSPACES <country> [type type ...]
        Draw airspaces for a country.  Types are given by name (e.g. CTR,
        TMA, FIR) or by number.  If no types are given, the list in
        airspace_default_types (settings.cfg) is used.

    DELAIRSPACES <country|ALL> [type type ...]
        Remove drawn airspaces.  Types are given by name or number.
        Use ALL to clear everything at once.

    LOADATC <country>
        Shorthand to draw only FIR for a country.

    LOADATCALL
        Draw FIR for every country in the database.

    LISTAIRSPACES
        Print a summary of every airspace currently drawn by this plugin.

    Airspace type names (and numbers)
    ---------------------------------
    Restricted(1)  Danger(2)   Prohibited(3)  CTR(4)   TMZ(5)   RMZ(6)
    TMA(7)         TRA(8)      TSA(9)         FIR(10)  ATZ(13)  CTA(26)
"""

import gzip
import json
import os
from collections import defaultdict

### Import the global bluesky objects.
from bluesky import core, stack, settings
from bluesky.tools import areafilter
from bluesky.tools.aero import ft


_DEFAULT_AIRSPACE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', 'resources', 'navdata', 'airspaces.json.gz')
)

### Register settings that can be overridden in settings.cfg.
settings.set_variable_defaults(
    # Path to airspaces data file. Defaults to resources/navdata/ where
    # openaip.py saves it. Supports both .json.gz and plain .json.
    # Override with an absolute path in settings.cfg if needed.
    airspace_data_path=_DEFAULT_AIRSPACE_PATH,

    # Types drawn when no type numbers are given to LOADAIRSPACES.
    airspace_default_types=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 26],

    # Polygons with more vertices than this are downsampled before drawing.
    airspace_max_polygon_vertices=150,

    # Total vertex budget across all drawn airspaces combined.
    # BlueSky hard limit is 200 000; we default to 195 000 for a small margin.
    airspace_vertex_budget=195000,
)

TYPE_NAMES = {
    0:  'Other',         1:  'Restricted',    2:  'Danger',
    3:  'Prohibited',    4:  'CTR',           5:  'TMZ',
    6:  'RMZ',           7:  'TMA',           8:  'TRA',
    9:  'TSA',           10: 'FIR',           11: 'UIR',
    12: 'ADIZ',          13: 'ATZ',           14: 'MATZ',
    15: 'Airway',        16: 'MTR',           17: 'Alert',
    18: 'Warning',       19: 'Protected',     20: 'HTZ',
    21: 'Gliding',       22: 'TRP',           23: 'TIZ',
    24: 'TIA',           25: 'MTA',           26: 'CTA',
    27: 'ACC',           28: 'Aerial/Rec',    29: 'LowAlt',
    30: 'MRT',           31: 'TFR',           32: 'VFR Sector',
    33: 'FIS Sector',    34: 'LTA',           35: 'UTA',
    36: 'MCTR',
}

# Reverse mapping: uppercase name → type number.
# Names with slashes or spaces get an additional cleaned-up alias.
NAME_TO_TYPE = {}
for _num, _name in TYPE_NAMES.items():
    _key = _name.upper().replace(' ', '').replace('/', '')
    NAME_TO_TYPE[_key] = _num
    NAME_TO_TYPE[_name.upper()] = _num

### Display colour (R, G, B) 0-255, keyed by airspace type.
TYPE_COLORS = {
    3:  (255,   0,   0),   # Prohibited  → red
    1:  (255, 140,   0),   # Restricted  → orange
    2:  (255, 140,   0),   # Danger      → orange
    4:  (100, 150, 255),   # CTR         → light blue
    7:  (100, 150, 255),   # TMA         → light blue
    26: (100, 150, 255),   # CTA         → light blue
    5:  (  0, 200, 200),   # TMZ         → cyan
    6:  (  0, 200, 200),   # RMZ         → cyan
    8:  (255, 255,   0),   # TRA         → yellow
    9:  (255, 255,   0),   # TSA         → yellow
    10: (255, 255, 255),   # FIR         → white
}

DEFAULT_COLOR = (180, 180, 180)   # light grey for any unlisted type


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    airspaceloader = AirspaceLoader()

    config = {
        'plugin_name': 'AIRSPACELOADER',
        'plugin_type': 'sim',
    }

    return config


class AirspaceLoader(core.Entity):
    ''' Loads and draws OpenAIP airspace polygons in BlueSky on demand. '''

    def __init__(self):
        super().__init__()

        # Index built once from the JSON file at startup.
        # Layout: _index[COUNTRY][type_int] = [airspace_dict, ...]
        self._index = defaultdict(lambda: defaultdict(list))

        # Tracks which shape names were drawn so we can delete them later.
        # Layout: _drawn[COUNTRY][type_int] = [shape_name, ...]
        self._drawn = defaultdict(lambda: defaultdict(list))

        # Maps each shape name → vertex count; used to subtract the right
        # amount from _vertex_count when a shape is deleted.
        self._shape_verts = {}

        # Running total of vertices currently drawn by this plugin.
        self._vertex_count = 0

        self._build_index()

    def reset(self):
        ''' Clear all drawn airspaces when the simulation resets. '''
        super().reset()
        # areafilter.reset() has already cleared the shapes from the screen.
        # We only need to reset our own bookkeeping.
        self._drawn = defaultdict(lambda: defaultdict(list))
        self._shape_verts = {}
        self._vertex_count = 0

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_index(self):
        ''' Read airspaces data and store every airspace in a fast lookup dict.

            Supports both gzipped (.json.gz) and plain (.json) files.
            If the configured path is not found, the other variant is tried
            automatically as a fallback.
        '''
        data_path = os.path.abspath(settings.airspace_data_path)

        # If the configured path doesn't exist, try the other extension.
        if not os.path.isfile(data_path):
            if data_path.endswith('.json.gz'):
                alt = data_path[:-3]          # strip .gz
            elif data_path.endswith('.json'):
                alt = data_path + '.gz'       # add .gz
            else:
                alt = None

            if alt and os.path.isfile(alt):
                print(f'[AIRSPACELOADER] {data_path} not found, using {alt}')
                data_path = alt
            else:
                print(f'[AIRSPACELOADER] WARNING: file not found at {data_path}')
                print( '[AIRSPACELOADER] Run extract_open_aip_data.py first, then restart BlueSky.')
                return

        # Open with gzip or plain text depending on extension.
        if data_path.endswith('.gz'):
            with gzip.open(data_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        for airspace in data:
            country = airspace.get('country', '').upper()
            atype   = airspace.get('type')
            geom    = airspace.get('geometry', {})

            # Only Polygon geometry is supported.
            if geom.get('type') != 'Polygon':
                continue

            self._index[country][atype].append(airspace)

        total       = sum(len(v) for c in self._index.values() for v in c.values())
        n_countries = len(self._index)
        print(f'[AIRSPACELOADER] Indexed {total} airspaces across {n_countries} countries.')
        print(f'[AIRSPACELOADER] Use LOADAIRSPACES <country> to draw airspaces.')

    def _alt_to_meters(self, limit):
        ''' Convert an OpenAIP altitude limit dict to metres.

            OpenAIP unit codes:   0=feet  1=metres  6=flight level (FL)
        '''
        value = limit.get('value', 0)
        unit  = limit.get('unit',  0)

        if unit == 6:
            return value * 100.0 * ft   # FL → feet → metres
        elif unit == 1:
            return float(value)          # already metres
        else:
            return value * ft            # feet (unit 0 or unknown)

    def _draw_one(self, airspace, country):
        ''' Draw a single airspace polygon. Returns (name, n_vertices) on
            success, or (None, 0) if the airspace was skipped. '''

        # GeoJSON outer ring: list of [lon, lat] pairs.
        ring = airspace['geometry']['coordinates'][0]

        # BlueSky expects a flat (lat, lon, lat, lon, ...) tuple.
        # GeoJSON stores coordinates as [lon, lat], so we swap the order.
        coords  = tuple(val for lon, lat in ring for val in (lat, lon))
        n_verts = len(ring)

        if n_verts < 3:
            return None, 0

        # Stop if adding this polygon would exceed the total vertex budget.
        if self._vertex_count + n_verts > settings.airspace_vertex_budget:
            return None, 0

        # Build a unique, BlueSky-safe shape name: uppercase, no spaces/hyphens.
        raw  = airspace.get('name', 'UNKNOWN')
        name = ''.join(c for c in raw.upper().replace(' ', '_').replace('-', '_')
                       if c.isalnum() or c == '_')
        name = f'{country}_{name}'

        top    = self._alt_to_meters(airspace.get('upperLimit', {}))
        bottom = self._alt_to_meters(airspace.get('lowerLimit', {}))

        # Create the polygon and set its colour.
        areafilter.defineArea(name, 'POLYALT', coords, top, bottom)
        r, g, b = TYPE_COLORS.get(airspace.get('type', -1), DEFAULT_COLOR)
        areafilter.colour(name, r, g, b)

        self._shape_verts[name]  = n_verts
        self._vertex_count      += n_verts
        return name, n_verts

    # -----------------------------------------------------------------------
    # Stack commands
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_types(types):
        ''' Convert a sequence of type tokens (names or numbers) to int codes.

            Accepts: "CTR", "FIR", "4", 4, etc.
            Returns: (list_of_ints, error_message_or_None)
        '''
        resolved = []
        for t in types:
            # Already an int (e.g. from internal call)
            if isinstance(t, int):
                resolved.append(t)
                continue
            key = str(t).strip().upper()
            # Try name lookup first, then raw number.
            if key in NAME_TO_TYPE:
                resolved.append(NAME_TO_TYPE[key])
            elif key.isdigit():
                resolved.append(int(key))
            else:
                return None, (f'[AIRSPACELOADER] Unknown airspace type "{t}". '
                              f'Use a name (e.g. CTR, TMA, FIR) or a number (e.g. 4, 7, 10).')
        return resolved, None

    @stack.command
    def loadairspaces(self, country: str, *types: str):
        ''' Draw airspaces for a country.

            Arguments:
                country   2-letter ISO country code, e.g. NL, BE, DE, US
                types     optional type names or numbers (see module docstring).
                          If omitted, airspace_default_types from settings.cfg is used.

            Examples:
                LOADAIRSPACES NL
                LOADAIRSPACES NL CTR TMA FIR
                LOADAIRSPACES NL 4 7 10
                LOADAIRSPACES US Restricted Danger Prohibited
        '''
        country = country.upper()

        if country not in self._index:
            return False, (f'[AIRSPACELOADER] No data for "{country}". '
                           f'Check that airspaces.json was downloaded and '
                           f'that the country code is a valid 2-letter ISO code.')

        if types:
            type_filter, err = self._resolve_types(types)
            if err:
                return False, err
        else:
            type_filter = list(settings.airspace_default_types)
        drawn = skipped = 0

        for atype in type_filter:
            for airspace in self._index[country].get(atype, []):
                name, n = self._draw_one(airspace, country)
                if name:
                    self._drawn[country][atype].append(name)
                    drawn += 1
                else:
                    skipped += 1

        type_labels = ', '.join(TYPE_NAMES.get(t, str(t)) for t in type_filter)
        msg = (f'[AIRSPACELOADER] {country}: drew {drawn} airspace(s) '
               f'({type_labels}). '
               f'Vertices in use: {self._vertex_count}/{settings.airspace_vertex_budget}.')

        if skipped:
            msg += (f' {skipped} skipped — vertex budget reached or polygon too small. '
                    f'Run DELAIRSPACES first to free space, or increase '
                    f'airspace_vertex_budget in settings.cfg.')

        return True, msg

    @stack.command
    def loadatc(self, country: str):
        ''' Draw FIR (10) airspaces for a country.

            Shorthand for: LOADAIRSPACES <country> 10

            Examples:
                LOADATC NL
                LOADATC US
        '''
        return self.loadairspaces(country,10)

    @stack.command
    def loadatcall(self):
        ''' Draw FIR (10) airspaces for every country in the database.

            Example:
                LOADATCALL
        '''
        drawn = skipped = 0

        for country in self._index:
            for airspace in self._index[country].get(10, []):
                name, n = self._draw_one(airspace, country)
                if name:
                    self._drawn[country][10].append(name)
                    drawn += 1
                else:
                    skipped += 1

        msg = (f'[AIRSPACELOADER] Drew {drawn} FIR airspace(s) across all countries. '
               f'Vertices in use: {self._vertex_count}/{settings.airspace_vertex_budget}.')
        if skipped:
            msg += (f' {skipped} skipped — vertex budget reached. '
                    f'Reduce airspace_max_polygon_vertices in settings.cfg to fit more.')
        return True, msg

    @stack.command
    def delairspaces(self, country: str, *types: str):
        ''' Remove airspaces drawn by LOADAIRSPACES.

            Arguments:
                country   2-letter ISO code, or ALL to remove everything.
                types     optional type names or numbers to remove.
                          If omitted, all types for that country are removed.

            Examples:
                DELAIRSPACES NL
                DELAIRSPACES NL CTR TMA
                DELAIRSPACES NL 4 7
                DELAIRSPACES ALL
        '''
        country = country.upper()

        if country == 'ALL':
            total = sum(len(names) for td in self._drawn.values() for names in td.values())
            for td in self._drawn.values():
                for names in td.values():
                    for name in names:
                        areafilter.deleteArea(name)
            self._drawn = defaultdict(lambda: defaultdict(list))
            self._shape_verts = {}
            self._vertex_count = 0
            return True, f'[AIRSPACELOADER] Removed {total} airspace(s). Vertex count reset to 0.'

        if country not in self._drawn:
            return False, f'[AIRSPACELOADER] No airspaces currently drawn for "{country}".'

        if types:
            type_filter, err = self._resolve_types(types)
            if err:
                return False, err
        else:
            type_filter = list(self._drawn[country].keys())

        removed = 0
        for atype in type_filter:
            for name in self._drawn[country].pop(atype, []):
                self._vertex_count -= self._shape_verts.pop(name, 0)
                areafilter.deleteArea(name)
                removed += 1

        if not self._drawn[country]:
            del self._drawn[country]

        return True, (f'[AIRSPACELOADER] Removed {removed} airspace(s) for {country}. '
                      f'Vertices in use: {self._vertex_count}/{settings.airspace_vertex_budget}.')

    @stack.command
    def listairspaces(self):
        ''' Show all airspaces currently drawn by the AIRSPACELOADER plugin. '''
        if not self._drawn:
            return True, '[AIRSPACELOADER] No airspaces are currently drawn.'

        lines = ['[AIRSPACELOADER] Currently drawn airspaces:']
        total = 0

        for country in sorted(self._drawn):
            for atype in sorted(self._drawn[country]):
                names = self._drawn[country][atype]
                label = TYPE_NAMES.get(atype, f'type {atype}')
                lines.append(f'  {country:4s}  {label:<20s}  {len(names)} shape(s)')
                for name in names:
                    lines.append(f'    → {name}')
                total += len(names)

        lines.append(f'  Total: {total} shape(s),  '
                     f'{self._vertex_count}/{settings.airspace_vertex_budget} vertices used.')
        return True, '\n'.join(lines)