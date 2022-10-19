'''BlueSky global configuration module'''
import re
import inspect
from bluesky.pathfinder import resource

# Default config file path
_cfgfile = 'settings.cfg'


def init(cfgfile=''):
    '''Initialize configuration.
       Import config settings from settings.cfg if this exists, if it doesn't
       create an initial config file'''

    # If no alternative is given we'll use the default config file
    if not cfgfile:
        cfgfile = _cfgfile

    # Store name of config file
    globals()['_cfgfile'] = resource(cfgfile)

    # Read the configuration file
    print(f'Reading config from {_cfgfile}')
    with open(_cfgfile) as fin:
        config = fin.read()
    # BlueSky resources have been moved since v2022.9.19. Update config file if necessary
    matches = re.findall(r'.+_path.+=.*data[/\\].*', config)
    if matches:
        print(f'Old-style resource paths found in {_cfgfile}. Do you want to update it')
        print('with the following changes?')
        for line in matches:
            var, p = line.split('=')
            np = p.strip().replace('data/', '').replace('data\\', '')
            print(f'{var.strip()}: from {p.strip()} to {np}')
        resp = input('Make these changes? [Y/n]: ')
        if (not resp) or 'y' in resp.lower():
            # Update settings and store to file
            print(f'Updating {_cfgfile}')
            config = config.replace('data/', '').replace('data\\', '')
            with open(_cfgfile, 'w') as fout:
                fout.write(config)
        else:
            print('If bluesky doesn\'t load correctly, please update your config file manually')

    exec(compile(config.replace('\\', '/'), _cfgfile, 'exec'), globals())

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
    ''' Save BlueSky configuration to file. '''
    # Apply any changes that are passed for saving
    if changes:
        globals().update(changes)
    # Make a copy of the list of settings
    allsettings = list(_settings)
    # Write to a specified file if passed, else current config file
    fname = resource(fname or _cfgfile)
    # Get config file formatting from file to be updated if it exists, else get
    # it from the config file template
    configsrc = fname if fname.is_file() else resource('default.cfg')
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
