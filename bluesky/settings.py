'''BlueSky global configuration module'''
import re
import sys
import shutil
import inspect
from pathlib import Path


# Default config file path
_cfgfile = Path('settings.cfg')


def init(cfgfile=''):
    '''Initialize configuration.
       Import config settings from settings.cfg if this exists, if it doesn't
       create an initial config file'''
       # When run from source (e.g., directly downloaded from git), both rundir and srcdir are the CWD
    rundir = Path('')
    srcdir = Path('')

    # If BlueSky is run from a compiled bundle instead of from source, or installed as a package
    # adjust the startup path and change the path of configurable files to $home/bluesky

    if Path(__file__) != Path.cwd() / 'bluesky/settings.py':
        # In this case, the run dir is a (to be created) bluesky folder in the user directory
        rundir = Path.home() / 'bluesky'
        if not rundir.is_dir():
            rundir.mkdir()
        # And the source dir resides in site-packages/bluesky/resources
        srcdir = Path(__file__).parent / 'resources'


    datadir = rundir / 'data'
    cachedir = rundir / 'data/cache'
    badadir = rundir / 'data/performance/BADA'
    badasrc = srcdir / 'data/performance/BADA'
    perfdir = srcdir / 'data/performance'
    gfxdir = srcdir / 'data/graphics'
    navdir = srcdir / 'data/navdata'
    scnsrc = srcdir / 'scenario'
    scndir = rundir / 'scenario'
    outdir = rundir / 'output'
    plgsrc = srcdir / 'plugins'
    plgdir = rundir / 'plugins'
    configsrc = srcdir / 'data/default.cfg'

    if cfgfile:
        print(f'Reading config from {cfgfile}')
    else:
        cfgfile = rundir / 'settings.cfg'
        # Create config file if it doesn't exist yet. Ask for gui settings if bluesky
        # was started with BlueSky.py
        if not cfgfile.is_file():
            print()
            print('No config file settings.cfg found in your BlueSky starting directory!')
            print()
            print('This config file contains several default settings related to the simulation loop and the graphics.')
            print('A default version will be generated, which you can change if necessary before the next time you run BlueSky.')
            print()

            with open(configsrc, 'r') as fin, open(cfgfile, 'w') as file_out:
                for line in fin:
                    if line[:9] == 'data_path':
                        line = f"data_path = '{datadir}'\n"
                    if line[:10] == 'cache_path':
                        line = f"cache_path = '{cachedir}'\n"
                    elif line[:8] == 'log_path':
                        line = f"log_path = '{outdir}'\n"
                    elif line[:13] == 'scenario_path':
                        line = f"scenario_path = '{scndir}'\n"
                    elif line[:11] == 'plugin_path':
                        line = f"plugin_path = '{plgdir}'\n"
                    elif line[:14] == 'perf_path_bada':
                        line = f"perf_path_bada = '{badadir}'\n"
                    elif line[:9] == 'perf_path':
                        line = f"perf_path = '{perfdir}'\n"
                    elif line[:8] == 'gfx_path':
                        line = f"gfx_path = '{gfxdir}'\n"
                    elif line[:12] == 'navdata_path':
                        line = f"navdata_path = '{navdir}'\n"

                    file_out.write(line)

        else:
            print(f'Reading config from {cfgfile}')

    exec(compile(open(cfgfile).read(), cfgfile, 'exec'), globals())

    # Use the path specified in cfgfile if available
    if 'cache_path' in globals():
        cachedir = Path(globals()['cache_path'])
    if 'log_path' in globals():
        outdir = Path(globals()['log_path'])
    if 'perf_path_bada' in globals():
        badadir = Path(globals()['perf_path_bada'])
    if 'scenario_path' in globals():
        scndir = Path(globals()['scenario_path'])
    if 'plugin_path' in globals():
        plgdir = Path(globals()['plugin_path'])

    # Update cachedir with python version-specific subfolder
    cachedir = cachedir / f'py{sys.version_info[0]}'
    globals()['cache_path'] = cachedir

    # Store name of config file
    globals()['_cfgfile'] = cfgfile

    # Create default directories if they don't exist yet
    for d in (outdir, cachedir):
        if not d.is_dir():
            print('Creating directory "%s"' % d)
            d.mkdir()
    for d in [(badasrc, badadir), (scnsrc, scndir), (plgsrc, plgdir)]:
        if not d[1].is_dir():
            print('Creating directory "%s", and copying default files' % d[1])
            try:
                shutil.copytree(*d)
            except FileNotFoundError:
                print('Unable to copy "%s" files to "%s"' %(d[0], d[1]), file=sys.stderr)

    return True

_settings_hierarchy = dict()
_settings = list()
def set_variable_defaults(**kwargs):
    ''' Register a default value for a configuration variable. Use this functionality
        in plugins to make sure that configuration variables are available upon usage.

        Example:
            from bluesky import settings
            settings.set_variable_defaults(var1=1.0, var2=[1, 2, 3])

            This will make settings.var1 and settings.var2 available, with the
            provided default values.'''
    for key, value in kwargs.items():
        if key not in globals():
            globals()[key] = value
        else:
            kwargs[key] = globals()[key]
        if key not in _settings:
            _settings.append(key)

    # Keep track of who this variable belongs to
    callertree = inspect.currentframe().f_back.f_globals['__name__'].split('.')
    tree = _settings_hierarchy
    visited = set()
    for loc in callertree:
        if loc in visited:
            continue
        if loc not in tree:
            tree[loc] = dict()
        tree = tree[loc]
        visited.add(loc)
    tree.update(kwargs)

def save(fname=None, changes=None):
    # Apply any changes that are passed for saving
    if changes:
        globals().update(changes)
    # Make a copy of the list of settings
    allsettings = list(_settings)
    # Write to a specified file if passed, else current config file
    fname = Path(fname or _cfgfile)
    # Get config file formatting from file to be updated if it exists, else get
    # it from the config file template
    srcdir = Path('')
    if Path(__file__) != Path.cwd() / 'bluesky/settings.py':
        # Source dir resides in site-packages/bluesky/resources
        srcdir = Path(__file__).parent / 'resources'
        if not fname.is_absolute():
            fname = Path.home() / 'bluesky' / fname

    configsrc = fname if fname.is_file() else srcdir / 'data/default.cfg'
    with open(configsrc) as f:
        lines = f.readlines()

    with open(fname, 'w') as file_out:
        # first write all lines following the format of the source file
        for line in lines:
            key = (re.findall(r'^(\w+)\s*=?.*', line.strip()) + [''])[0]
            if key in allsettings:
                allsettings.pop(allsettings.index(key))
                value = globals()[key]
                if isinstance(value, str):
                    file_out.write(f'{key} = \'{value}\'\n')
                else:
                    file_out.write(f'{key} = {value}\n')
            else:
                file_out.write(line)
        # Then write any remaining additional settings
        file_out.write('\n')
        for key in allsettings:
            value = globals()[key]
            if isinstance(value, str):
                file_out.write(f'{key} = \'{value}\'\n')
            else:
                file_out.write(f'{key} = {value}\n')

    return True, f'Saved settings to {fname}'
