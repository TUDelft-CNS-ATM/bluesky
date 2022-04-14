'''BlueSky global configuration module'''
import re
import os
import sys
import shutil
import site
import inspect
from pathlib import Path


def init(cfgfile=''):
    '''Initialize configuration.
       Import config settings from settings.cfg if this exists, if it doesn't
       create an initial config file'''
    rundir = ''
    srcdir = ''

    # If BlueSky is run from a compiled bundle instead of from source,
    # adjust the startup path and change the path of
    # configurable files to $home/bluesky
    if getattr(sys, 'frozen', False):
        srcdir = os.path.dirname(sys.executable)
        rundir = os.path.join(os.path.expanduser('~'), 'bluesky')

    # If BlueSky is installed as a python package the location of the data files need to
    # be adjusted so that importing will not fail when copying config file below
    if not os.path.isfile(os.path.join(rundir, 'data/default.cfg')):
        # collate list of possible data install roots
        root_dirs = site.getusersitepackages()
        root_dirs = [root_dirs] if isinstance(root_dirs, str) else root_dirs
        root_dirs += site.getsitepackages()

        # search for bluesky shared data directory
        found_dir = False
        for root_dir in root_dirs:
            dirpath = os.path.join(root_dir, 'share', 'bluesky')
            if os.path.exists(dirpath):
                srcdir = dirpath
                found_dir = True
                break
        
        # if the path does not exist, it's worth trying the project root. This
        # would work if the package was cloned from the git and is installed
        # with "pip install -e ."
        if not found_dir:
            srcdir = get_project_root()

    datadir = os.path.join(rundir, 'data')
    cachedir = os.path.join(rundir, 'data/cache')
    badadir = os.path.join(rundir, 'data/performance/BADA')
    badasrc = os.path.join(srcdir, 'data/performance/BADA')
    perfdir = os.path.join(srcdir, 'data/performance')
    gfxdir = os.path.join(srcdir, 'data/graphics')
    navdir = os.path.join(srcdir, 'data/navdata')
    scnsrc = os.path.join(srcdir, 'scenario')
    scndir = os.path.join(rundir, 'scenario')
    outdir = os.path.join(rundir, 'output')
    plgsrc = os.path.join(srcdir, 'plugins')
    plgdir = os.path.join(rundir, 'plugins')
    configsrc = os.path.join(srcdir, 'data/default.cfg')

    if not cfgfile:
        cfgfile = os.path.join(rundir, 'settings.cfg')

    # Create config file if it doesn't exist yet. Ask for gui settings if bluesky
    # was started with BlueSky.py
    if not os.path.isfile(cfgfile):
        print()
        print('No config file settings.cfg found in your BlueSky starting directory!')
        print()
        print('This config file contains several default settings related to the simulation loop and the graphics.')
        print('A default version will be generated, which you can change if necessary before the next time you run BlueSky.')
        print()

        with open(configsrc, 'r') as fin, open(cfgfile, 'w') as file_out:
            for line in fin:
                if line[:9] == 'data_path':
                    line = "data_path = '" + datadir.replace('\\', '/') + "'\n"
                if line[:10] == 'cache_path':
                    line = "cache_path = '" + cachedir.replace('\\', '/') + "'\n"
                elif line[:8] == 'log_path':
                    line = "log_path = '" + outdir.replace('\\', '/') + "'\n"
                elif line[:13] == 'scenario_path':
                    line = "scenario_path = '" + scndir.replace('\\', '/') + "'\n"
                elif line[:11] == 'plugin_path':
                    line = "plugin_path = '" + plgdir.replace('\\', '/') + "'\n"
                elif line[:14] == 'perf_path_bada':
                    line = "perf_path_bada = '" + badadir.replace('\\', '/') + "'\n"
                elif line[:9] == 'perf_path':
                    line = "perf_path = '" + perfdir.replace('\\', '/') + "'\n"
                elif line[:8] == 'gfx_path':
                    line = "gfx_path = '" + gfxdir.replace('\\', '/') + "'\n"
                elif line[:12] == 'navdata_path':
                    line = "navdata_path = '" + navdir.replace('\\', '/') + "'\n"

                file_out.write(line)

    else:
        print(f'Reading config from {cfgfile}')

    exec(compile(open(cfgfile).read(), cfgfile, 'exec'), globals())

    # Use the path specified in cfgfile if available
    if 'cache_path' in globals():
        cachedir = globals()['cache_path']
    if 'log_path' in globals():
        outdir = globals()['log_path']
    if 'perf_path_bada' in globals():
        badadir = globals()['perf_path_bada']
    if 'scenario_path' in globals():
        scndir = globals()['scenario_path']
    if 'plugin_path' in globals():
        plgdir = globals()['plugin_path']

    # Update cachedir with python version-specific subfolder
    cachedir = os.path.join(cachedir, 'py%d' % sys.version_info[0])
    globals()['cache_path'] = cachedir

    # Store name of config file
    globals()['_cfgfile'] = cfgfile

    # Create default directories if they don't exist yet
    for d in (outdir, cachedir):
        if not os.path.isdir(d):
            print('Creating directory "%s"' % d)
            os.makedirs(d)
    for d in [(badasrc, badadir), (scnsrc, scndir), (plgsrc, plgdir)]:
        if not os.path.isdir(d[1]):
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
    fname = fname or _cfgfile
    # Get config file formatting from file to be updated if it exists, else get
    # it from the config file template
    srcdir = ''
    if getattr(sys, 'frozen', False):
        srcdir = os.path.dirname(sys.executable)
        if not os.path.isabs(fname):
            fname = os.path.join(os.path.expanduser('~'), 'bluesky', fname)

    configsrc = fname if os.path.isfile(fname) else os.path.join(srcdir, 'data/default.cfg')
    lines = [line for line in open(configsrc, 'r')]

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

def get_project_root() -> str:
    ''' Return the absolute path of the project root. '''

    # return root dir relative to this file, make sure you update it if this
    # file is moved in the project directory
    return str(Path(__file__).absolute().parent.parent)
