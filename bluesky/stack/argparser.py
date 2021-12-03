''' Stack argument parsers. '''
import inspect
import re
from types import SimpleNamespace
from matplotlib import colors
from bluesky.tools.misc import txt2bool, txt2lat, txt2lon, txt2alt, txt2tim, \
    txt2hdg, txt2vs, txt2spd
from bluesky.tools.position import Position, islat
import bluesky as bs


# Regular expression for argument parser
# Reading the regular expression:
# [\'"]?             : skip potential opening quote
# (?<=[\'"])[^\'"]+  : look behind for a leading quote, and if so, parse everything until closing quote
# (?<![\'"])[^\s,]+  : look behind for not a leading quote, then parse until first whitespace or comma
# [\'"]?\s*,?\s*     : skip potential closing quote, whitespace, and a potential single comma
re_getarg = re.compile(
    r'\s*[\'"]?((?<=[\'"])[^\'"]*|(?<![\'"])[^\s,]*)[\'"]?\s*,?\s*(.*)')
# re_getarg = re.compile(r'[\'"]?((?<=[\'"])[^\'"]+|(?<![\'"])[^\s,]+)[\'"]?\s*,?\s*')

# Stack reference data namespace
refdata = SimpleNamespace(lat=None, lon=None, alt=None, acidx=-1, hdg=None, cas=None)


def getnextarg(cmdstring):
    ''' Return first argument and remainder of command string from cmdstring. '''
    return re_getarg.match(cmdstring).groups()


def reset():
    ''' Reset reference data. '''
    refdata.lat = None
    refdata.lon = None
    refdata.alt = None
    refdata.acidx = -1
    refdata.hdg = None
    refdata.cas = None


class Parameter:
    ''' Wrapper class for stack function parameters. '''
    def __init__(self, param, annotation='', isopt=None):
        self.name = param.name
        self.default = param.default
        self.optional = (self.hasdefault() or param.kind == param.VAR_POSITIONAL) if isopt is None else isopt
        self.gobble = param.kind == param.VAR_POSITIONAL and not annotation
        self.annotation = annotation or param.annotation

        # Make list of parsers
        if self.annotation is inspect._empty:
            # Without annotation the argument is passed on unchanged as string
            # (i.e., the 'word' argument type)
            self.parsers = [Parser(str)]
            self.annotation = 'word'
        elif isinstance(self.annotation, str):
            # If the annotation is a string we get our parsers from the argparsers dict
            pfuns = [argparsers.get(a) for a in self.annotation.split('/')]
            self.parsers = [p for p in pfuns if p is not None]
        elif isinstance(param.annotation, type) and issubclass(param.annotation, Parser):
            # If the paramter annotation is a class derived from Parser
            self.parsers = [self.annotation()]
        else:
            # All other annotation types are expected to have default behaviour
            # and are wrapped in Parser
            self.parsers = [Parser(self.annotation)]

        # This parameter is not valid if it has no parsers, or is keyword-only.
        # In those cases it can be skipped from the list of parameters when
        # processing a stack command line.
        self.valid = bool(self.parsers) and self.canwrap(param)

    def __call__(self, argstring):
        # First check if argument is omitted and default value is needed
        if not argstring or argstring[0] == ',':
            _, argstring = re_getarg.match(argstring).groups()
            if self.hasdefault():
                return self.default, argstring
            if self.optional:
                return (None, argstring) if argstring else ('',)
            raise ArgumentError(f'Missing argument {self.name}')
        # Try available parsers
        error = ''
        for parser in self.parsers:
            try:
                return parser.parse(argstring)
            except (ValueError, ArgumentError) as e:
                error += ('\n' + e.args[0])

        # If all fail, raise error
        raise ArgumentError(error)

    def __str__(self):
        return f'{self.name}:{self.annotation}'

    def __bool__(self):
        return self.valid

    def size(self):
        ''' Returns the (maximum) number of return variables when parsing this
            parameter. '''
        return max((p.size for p in self.parsers))

    def hasdefault(self):
        ''' Returns True if this parameter has a default value. '''
        return self.default is not inspect._empty

    @staticmethod
    def canwrap(param):
        ''' Returns True if Parameter can be used to wrap given function parameter.
            Returns False if param is keyword-only. '''
        return param.kind not in (param.VAR_KEYWORD, param.KEYWORD_ONLY)


class ArgumentError(Exception):
    ''' This error is raised when stack argument parsing fails. '''
    pass
class Parser:
    ''' Base implementation of argument parsers
        that are used to parse arguments to stack commands.
    '''

    # Output size of this parser
    size = 1

    def __init__(self, parsefun=None):
        self.parsefun = parsefun

    def parse(self, argstring):
        ''' Parse the next argument from argstring. '''
        curarg, argstring = re_getarg.match(argstring).groups()
        return self.parsefun(curarg), argstring


class StringArg(Parser):
    ''' Argument parser that simply consumes the entire remaining text string. '''
    def parse(self, argstring):
        return argstring, ''


class AcidArg(Parser):
    ''' Argument parser for aircraft callsigns and group ids. '''
    def parse(self, argstring):
        arg, argstring = re_getarg.match(argstring).groups()
        acid = arg.upper()
        if acid in bs.traf.groups:
            idx = bs.traf.groups.listgroup(acid)
        else:
            idx = bs.traf.id2idx(acid)
            if idx < 0:
                raise ArgumentError(f'Aircraft with callsign {acid} not found')

            # Update ref position for navdb lookup
            refdata.lat = bs.traf.lat[idx]
            refdata.lon = bs.traf.lon[idx]
            refdata.acidx = idx
        return idx, argstring


class WpinrouteArg(Parser):
    ''' Argument parser for waypoints in an aircraft route. '''
    def parse(self, argstring):
        arg, argstring = re_getarg.match(argstring).groups()
        wpname = arg.upper()
        if refdata.acidx >= 0 and wpname in bs.traf.ap.route[refdata.acidx].wpname or wpname == '*':
            return wpname, argstring
        raise ArgumentError(f'{wpname} not found in the route of {bs.traf.id[refdata.acidx]}')

class WptArg(Parser):
    ''' Argument parser for waypoints.
        Makes 1 or 2 argument(s) into 1 position text to be used as waypoint

        Examples valid position texts:
        lat/lon : "N52.12,E004.23","N52'14'12',E004'23'10"
        navaid/fix: "SPY","OA","SUGOL"
        airport:   "EHAM"
        runway:    "EHAM/RW06" "LFPG/RWY23"
        Default values
    '''
    def parse(self, argstring):
        arg, argstring = re_getarg.match(argstring).groups()
        name = arg.upper()

        # Try aircraft first: translate a/c id into a valid position text with a lat,lon
        idx = bs.traf.id2idx(name)
        if idx >= 0:
            name = f'{bs.traf.lat[idx]},{bs.traf.lon[idx]}'

        # Check if lat/lon combination
        elif islat(name):
            # lat,lon ? Combine into one string with a comma
            arg, argstring = re_getarg.match(argstring).groups()
            name = name + "," + arg

        # apt,runway ? Combine into one string with a slash as separator
        elif argstring[:2].upper() == "RW" and name in bs.navdb.aptid:
            arg, argstring = re_getarg.match(argstring).groups()
            name = name + "/" + arg.upper()

        return name, argstring


class PosArg(Parser):
    ''' Argument parser for lat/lon positions.
        Makes 1 or 2 argument(s) into a lat/lon coordinate

        Examples valid position texts:
        lat/lon : "N52.12,E004.23","N52'14'12',E004'23'10"
        navaid/fix: "SPY","OA","SUGOL"
        airport:   "EHAM"
        runway:    "EHAM/RW06" "LFPG/RWY23"
        Default values
    '''
    # This parser's output size is 2 (lat, lon)
    size = 2

    def parse(self, argstring):
        arg, argstring = re_getarg.match(argstring).groups()
        argu = arg.upper()

        # Try aircraft first: translate a/c id into a valid position text with a lat,lon
        idx = bs.traf.id2idx(argu)
        if idx >= 0:
            return bs.traf.lat[idx], bs.traf.lon[idx], argstring

        # Check if lat/lon combination
        if islat(argu):
            nextarg, argstring = re_getarg.match(argstring).groups()
            refdata.lat = txt2lat(argu)
            refdata.lon = txt2lon(nextarg)
            return txt2lat(argu), txt2lon(nextarg), argstring

        # apt,runway ? Combine into one string with a slash as separator
        if argstring[:2].upper() == "RW" and argu in bs.navdb.aptid:
            arg, argstring = re_getarg.match(argstring).groups()
            argu = argu + "/" + arg.upper()

        if refdata.lat is None:
            refdata.lat, refdata.lon = bs.scr.getviewctr()

        posobj = Position(argu, refdata.lat, refdata.lon)
        if posobj.error:
            raise ArgumentError(f'{argu} is not a valid waypoint, airport, runway, or aircraft id.')

        # Update reference lat/lon
        refdata.lat = posobj.lat
        refdata.lon = posobj.lon
        refdata.hdg = posobj.refhdg

        return posobj.lat, posobj.lon, argstring


class PandirArg(Parser):
    ''' Parse pan direction commands. '''
    def parse(self, argstring):
        arg, argstring = re_getarg.match(argstring).groups()
        pandir = arg.upper()
        if pandir not in ('LEFT', 'RIGHT', 'UP', 'ABOVE', 'RIGHT', 'DOWN'):
            raise ArgumentError(f'{arg} is not a valid pan direction')
        return pandir, argstring


class ColorArg(Parser):
    ''' Parse color commands. '''
    def parse(self, argstring):
        arg, argstring = re_getarg.match(argstring).groups()
        try:
            if arg.isnumeric():
                g, argstring = re_getarg.match(argstring).groups()
                b, argstring = re_getarg.match(argstring).groups()
                return int(arg), int(g), int(b), argstring
            r, g, b = [int(255 * i) for i in colors.to_rgb(arg.upper())]
            return r, g, b, argstring
        except ValueError:
            raise ArgumentError(f'Could not parse "{arg}" as color')


argparsers = {
    '*': None,
    'txt': Parser(str.upper),
    'word': Parser(str),
    'string': StringArg(),
    'float': Parser(float),
    'int': Parser(int),
    'onoff': Parser(txt2bool),
    'bool': Parser(txt2bool),
    'acid': AcidArg(),
    'wpinroute': WpinrouteArg(),
    'wpt': WptArg(),
    'latlon': PosArg(),
    'lat': PosArg(),
    'lon': None,
    'pandir': PandirArg(),
    'spd': Parser(txt2spd),
    'vspd': Parser(txt2vs),
    'alt': Parser(txt2alt),
    'hdg': Parser(lambda txt: txt2hdg(txt, refdata.lat, refdata.lon)),
    'time': Parser(txt2tim),
    'color': ColorArg()}
