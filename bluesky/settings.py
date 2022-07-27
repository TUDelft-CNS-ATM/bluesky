'''BlueSky global configuration module'''
import re
import sys
import shutil
import inspect
from pathlib import Path


# Default config file path
_cfgfile = Path('settings.cfg')
# Base path and source path
_basepath = Path('')
_srcpath = Path('')


def resolve_path(path):
    ''' Resolve a path to BlueSky-related (data) files. Adds base path to relative paths.

        Arguments:
        - path: The path to resolve

        Returns:
        - resolved path
    '''
    path = Path(path)
    return path if path.is_absolute() else (_basepath / path).resolve()


def init(cfgfile=''):
    '''Initialize configuration.
       Import config settings from settings.cfg if this exists, if it doesn't
       create an initial config file'''
       # When run from source (e.g., directly downloaded from git), both rundir and srcdir are the CWD
    global _basepath, _srcpath

    # If BlueSky is run from a compiled bundle instead of from source, or installed as a package
    # adjust the startup path and change the path of configurable files to $home/bluesky

    if Path(__file__) != Path.cwd() / 'bluesky/settings.py':
        # In this case, the run dir is a (to be created) bluesky folder in the user directory
        _basepath = Path.home() / 'bluesky'
        # And the source dir resides in site-packages/bluesky/resources
        _srcpath = Path(__file__).parent / 'resources'
        # Check if basedir already exists. If not create it
        if not _basepath.is_dir():
            populate_basedir()

    if not cfgfile:
        cfgfile = _basepath / 'settings.cfg'

        # check if config file exists
        if not cfgfile.is_file():
            # If not, create a default config file
            print(f'Creating default config file "{cfgfile}"')
            shutil.copyfile(_srcpath / 'data/default.cfg', cfgfile)
  
    print(f'Reading config from {cfgfile}')
    exec(compile(open(cfgfile).read().replace('\\', '/'), cfgfile, 'exec'), globals())

    # Store name of config file
    globals()['_cfgfile'] = cfgfile

    # populate some directories in case they don't exist if using from source 
    for d in ('output', 'data/cache'):
        if not (_basepath / d).is_dir():
            print(f'Creating directory "{_basepath / d}"')
            (_basepath / d).mkdir(parents=True, exist_ok=True)

    return True


def populate_basedir():
    ''' Populate bluesky folder in home directory when running bluesky as a package. '''
    # Create base path and copy default config
    print(f'Creating BlueSky base directory "{_basepath.absolute()}"')
    _basepath.mkdir()
    print(f'Copying default configfile to {_basepath / "settings.cfg"}')
    shutil.copyfile(_srcpath / 'data/default.cfg', _basepath / 'settings.cfg')

    # Paths to create
    for d in ('output', 'data/cache', 'data/performance'):
        print(f'Creating directory "{_basepath / d}"')
        (_basepath / d).mkdir(parents=True, exist_ok=True)

    # Paths to create and populate
    for d in ('scenario', 'plugins', 'data/performance/BADA'):
        print(f'Creating directory "{_basepath / d}", and copying default files')
        try:
            shutil.copytree(_srcpath / d, _basepath / d)
        except FileNotFoundError:
                print('Unable to copy "%s" files to "%s"' %(d[0], d[1]), file=sys.stderr)

    # Performance paths to create symbolic links for
    for d in (_srcpath / 'data/performance').iterdir():
        # Skip BADA dir (which is copied), link others
        if d.name.upper() != 'BADA':
            symlink = _basepath / 'data/performance' / d.name
            try:
                symlink.symlink_to(d, target_is_directory=True)
            except FileNotFoundError:
                    print(f'Unable to create symbolic link "{symlink}"', file=sys.stderr)

    # Other data paths to create symbolic links for
    for d in ('data/graphics', 'data/navdata'):
        symlink = _basepath / d
        try:
            symlink.symlink_to(_srcpath / d, target_is_directory=True)
        except FileNotFoundError:
                print(f'Unable to create symbolic link "{symlink}"', file=sys.stderr)


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
