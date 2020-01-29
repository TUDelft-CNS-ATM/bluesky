""" Implementation of BlueSky's plugin system. """
import ast
from os import path
from pathlib import Path
import sys
import imp
import bluesky as bs
from bluesky import settings
from bluesky.tools import varexplorer as ve
from bluesky.tools.simtime import timed_function

# Register settings defaults
settings.set_variable_defaults(plugin_path='plugins', enabled_plugins=['datafeed'])

# Dict of descriptions of plugins found for this instance of bluesky
plugin_descriptions = dict()
# Dict of loaded plugins for this instance of bluesky
active_plugins = dict()

class Plugin:
    def __init__(self, fname):
        fname = path.normpath(path.splitext(fname)[0].replace('\\', '/'))
        self.module_path, self.module_name = path.split(fname)
        self.module_imp = fname.replace('/', '.')
        self.plugin_doc   = ''
        self.plugin_name  = ''
        self.plugin_type  = ''
        self.plugin_stack = []

def check_plugin(fname):
    plugin = None
    with open(fname, 'rb') as f:
        source = f.read()
        try:
            tree = ast.parse(source)
        except:
            return None

        ret_dicts = []
        ret_names = ['', '']
        for item in tree.body:
            if isinstance(item, ast.FunctionDef) and item.name == 'init_plugin':
                # This is the initialization function of a bluesky plugin. Parse the contents
                plugin = Plugin(fname)
                plugin.plugin_doc = ast.get_docstring(tree)
                for iitem in reversed(item.body):
                    # Return value of init_plugin should always be a tuple of two dicts
                    # The first dict is the plugin config dict, the second dict is the stack function dict
                    if isinstance(iitem, ast.Return):
                        ret_dicts = iitem.value.elts
                        if not len(ret_dicts) == 2:
                            print(fname + " looks like a plugin, but init_plugin() doesn't return two dicts")
                            return None
                        ret_names = [el.id if isinstance(el, ast.Name) else '' for el in ret_dicts]

                    # Check if this is the assignment of one of the return values
                    if isinstance(iitem, ast.Assign) and isinstance(iitem.value, ast.Dict):
                        for i in range(2):
                            if iitem.targets[0].id == ret_names[i]:
                                ret_dicts[i] = iitem.value

                # Parse the config dict
                cfgdict = {k.s:v for k,v in zip(ret_dicts[0].keys, ret_dicts[0].values)}
                plugin.plugin_name = cfgdict['plugin_name'].s
                plugin.plugin_type = cfgdict['plugin_type'].s

                # Parse the stack function dict
                stack_keys       = [el.s for el in ret_dicts[1].keys]
                stack_docs       = [el.elts[-1].s for el in ret_dicts[1].values]
                plugin.plugin_stack = list(zip(stack_keys, stack_docs))
    return plugin

def manage(cmd='LIST', plugin_name=''):
    ''' Stack function interaction for plugin system.'''
    if cmd == 'LIST':
        running   = set(active_plugins.keys())
        available = set(plugin_descriptions.keys()) - running
        text  = '\nCurrently running plugins: %s' % ', '.join(running)
        if available:
            text += '\nAvailable plugins: %s' % ', '.join(available)
        else:
            text += '\nNo additional plugins available.'
        return True, text

    if cmd in ['LOAD', 'ENABLE']:
        return load(plugin_name)
    elif cmd in ['REMOVE', 'UNLOAD', 'DISABLE']:
        return remove(plugin_name)
    elif cmd != '': # If no command is given, assume user tries to load a plugin
        return load(cmd)
    return False

def init(mode):
    ''' Initialization function of the plugin system.'''
    # Add plugin path to module search path
    sys.path.append(path.abspath(settings.plugin_path))
    # Set plugin type for this instance of BlueSky
    req_type = 'sim' if mode[:3] == 'sim' else 'gui'

    # Find available plugins
    for fname in Path(settings.plugin_path).rglob('*.py'):
        p = check_plugin(fname)
        # This is indeed a plugin, and it is meant for us
        if p and p.plugin_type == req_type:
            plugin_descriptions[p.plugin_name.upper()] = p
    # Load plugins selected in config
    for pname in settings.enabled_plugins:
        success = load(pname.upper())
        print(success[1])


# Sim implementation of plugin management
preupdate_funs = dict()
update_funs    = dict()
reset_funs     = dict()

def load(name):
    ''' Load a plugin. '''
    try:
        if name in active_plugins:
            return False, 'Plugin %s already loaded' % name
        descr  = plugin_descriptions.get(name)
        if not descr:
            return False, 'Error loading plugin: plugin %s not found.' % name
        # Load the plugin
        mod    = imp.find_module(descr.module_name, [descr.module_path])
        plugin = imp.load_module(descr.module_name, *mod)
        # Initialize the plugin
        config, stackfuns    = plugin.init_plugin()
        active_plugins[name] = plugin
        dt     = max(config.get('update_interval', 0.0), bs.sim.simdt)
        prefun = config.get('preupdate')
        updfun = config.get('update')
        rstfun = config.get('reset')
        if prefun:
            if hasattr(prefun, '__istimed'):
                preupdate_funs[name] = prefun
            else:
                preupdate_funs[name] = timed_function(f'{name}.{prefun.__name__}', dt)(prefun)

        if updfun:
            if hasattr(updfun, '__istimed'):
                update_funs[name] = updfun
            else:
                update_funs[name] = timed_function(f'{name}.{updfun.__name__}', dt)(updfun)

        if rstfun:
            reset_funs[name]     = rstfun
        # Add the plugin's stack functions to the stack
        bs.stack.append_commands(stackfuns)
        # Add the plugin as data parent to the variable explorer
        ve.register_data_parent(plugin, name.lower())
        return True, 'Successfully loaded plugin %s' % name
    except ImportError as e:
        print('BlueSky plugin system failed to load', name, ':', e)
        return False, 'Failed to load %s' % name

def remove(name):
    ''' Remove a loaded plugin. '''
    if name not in active_plugins:
        return False, 'Plugin %s not loaded' % name
    preset = reset_funs.pop(name, None)
    if preset:
        # Call module reset first to clear plugin state just in case.
        preset()
    descr  = plugin_descriptions.get(name)
    cmds, _ = list(zip(*descr.plugin_stack))
    bs.stack.remove_commands(cmds)
    active_plugins.pop(name)
    preupdate_funs.pop(name)
    update_funs.pop(name)

def preupdate():
    ''' Update function executed before traffic update.'''
    for fun in preupdate_funs.values():
        fun()


def update():
    ''' Update function executed after traffic update.'''
    for fun in update_funs.values():
        fun()


def reset():
    ''' Reset all plugins.'''
    # Call plugin reset for plugins that have one
    for fun in reset_funs.values():
        fun()
