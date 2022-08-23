'''BlueSky global configuration module'''
import re
import sys
import shutil
import inspect
from pathlib import Path
from turtle import up


# Flag indicating if BlueSky is running from source or packaged
_packaged = (Path(__file__) != Path.cwd() / 'bluesky/settings.py')
# Base path and source path depend on whether BlueSky is running from source
# In this case, the run dir is a (to be created) bluesky folder in the user directory
_basepath = (Path.home() / 'bluesky') if _packaged else Path('')
# And the source dir resides in site-packages/bluesky/resources
_srcpath = (Path(__file__).parent / 'resources') if _packaged else Path('')
# Default config file path
_cfgfile = _basepath / 'settings.cfg'


def resolve_path(path):
    ''' Resolve a path to BlueSky-related (data) files. Adds base path to relative paths.

        Arguments:
        - path: The path to resolve

        Returns:
        - resolved path
    '''
    path = Path(path)
    return path if path.is_absolute() else (_basepath / path).resolve()


def init(cfgfile='', updatedirs=False):
    '''Initialize configuration.
       Import config settings from settings.cfg if this exists, if it doesn't
       create an initial config file'''


    # If BlueSky is run from a compiled bundle instead of from source, or installed as a package
    # adjust the startup path and change the path of configurable files to $home/bluesky

    # Check if basedir already exists. If not create it.
    # Also (re)populate directories if updatedirs is set
    if _packaged and (updatedirs or not _basepath.is_dir()):
        populate_basedir(updatedirs)

    # If no alternative is given we'll use the default config file
    if not cfgfile:
        cfgfile = _cfgfile
        # check if config file exists
        if not cfgfile.is_file():
            # If not, create a default config file
            print(f'Creating default config file "{cfgfile}"')
            shutil.copyfile(_srcpath / 'data/default.cfg', cfgfile)
    else:
        # Store name of config file
        globals()['_cfgfile'] = Path(cfgfile)

    # Read the configuration file
    print(f'Reading config from {cfgfile}')
    exec(compile(open(cfgfile).read().replace('\\', '/'), cfgfile, 'exec'), globals())

    # populate some directories in case they don't exist if using from source 
    for d in ('output', 'data/cache'):
        if not (_basepath / d).is_dir():
            print(f'Creating directory "{_basepath / d}"')
            (_basepath / d).mkdir(parents=True, exist_ok=True)

    return True


def populate_basedir(updatedirs=False):
    ''' Populate bluesky folder in home directory when running bluesky as a package. '''
    if _basepath.exists() and not updatedirs:
        print(f'Error: directory {_basepath} already exists!')
        return
    def copyfn(src, dst, *args, follow_symlinks=True):
        ''' Custom copy function that checks if file exists and is newer '''
        dpath = Path(dst)
        spath = Path(src)
        if dpath.exists() and spath.stat().st_mtime > dpath.stat().st_mtime:
            if not copyfn.replaceall:
                res = input(f'Replace {dst} with newer {src}? [y/N/a]: ') or 'n'
                copyfn.replaceall = (res.lower()[0] == 'a')
                if res.lower()[0] not in 'ya':
                    return
        return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
    # Replace all prompt only once per call of populate_basedir
    copyfn.replaceall = False

    # Create base path and copy default config
    if not _basepath.exists():
        print(f'Creating BlueSky base directory "{_basepath.absolute()}"')
        _basepath.mkdir()
    print(f'Copying default configfile to {_basepath / "settings.cfg"}')
    copyfn(_srcpath / 'data/default.cfg', _basepath / 'settings.cfg')

    # Paths to create
    for d in ('output', 'data/cache', 'data/performance'):
        print(f'Creating directory "{_basepath / d}"')
        (_basepath / d).mkdir(parents=True, exist_ok=True)

    # Paths to create and populate
    for d in ('scenario', 'plugins', 'data/performance/BADA'):
        print(f'Creating directory "{_basepath / d}", and copying default files')
        try:
            shutil.copytree(_srcpath / d, _basepath / d, dirs_exist_ok=updatedirs, copy_function=copyfn)
        except FileNotFoundError:
                print('Unable to copy "%s" files to "%s"' %(d[0], d[1]), file=sys.stderr)

    def mklink(symlink):
        if symlink.exists() and not symlink.is_symlink():
            print(f'{symlink} exists as a file/folder, but should be a symbolic link to a package resource folder.')
            ans = input('Do you want me to replace the file/folder with the symbolic link? [y/N]') or 'n'
            if ans.lower()[0] == 'y':
                shutil.rmtree(symlink, ignore_errors=True)
        try:
            if not symlink.exists():
                symlink.symlink_to(d, target_is_directory=True)
        except FileNotFoundError:
                print(f'Unable to create symbolic link "{symlink}"', file=sys.stderr)

    # Performance paths to create symbolic links for
    for d in (_srcpath / 'data/performance').iterdir():
        # Skip BADA dir (which is copied), link others
        if d.is_dir() and d.name.upper() != 'BADA':
            mklink(_basepath / 'data/performance' / d.name)

    # Other data paths to create symbolic links for
    for d in ('data/graphics', 'data/navdata'):
        mklink(_basepath / d)

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
