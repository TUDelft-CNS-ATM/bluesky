""" Implementation of BlueSky's plugin system. """
import ast
from os import path
import sys
from glob import glob
import imp
import bluesky as bs
from bluesky import settings

# Register settings defaults
settings.set_variable_defaults(plugin_path='plugins', enabled_plugins=['datafeed'])

# Dict of descriptions of plugins found for this instance of bluesky
plugin_descriptions = dict()
# Dict of loaded plugins for this instance of bluesky
active_plugins = dict()

class Plugin(object):
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
        source         = f.read()
        tree           = ast.parse(source)
        retnames       = ['', '']
        for item in tree.body:
            if isinstance(item, ast.FunctionDef) and item.name == 'init_plugin':
                # This is the initialization function of a bluesky plugin. Parse the contents
                plugin = Plugin(fname)
                plugin.plugin_doc = ast.get_docstring(tree)
                for iitem in reversed(item.body):
                    # Return value of init_plugin should always be a tuple of two dicts
                    # The first dict is the plugin config dict, the second dict is the stack function dict
                    if isinstance(iitem, ast.Return):
                        retnames = [el.id for el in iitem.value.elts]

                    # Check if this is the assignment of one of the return values
                    if isinstance(iitem, ast.Assign) and isinstance(iitem.value, ast.Dict):
                        # Is this the config dict?
                        if iitem.targets[0].id == retnames[0]:
                            cfgdict = {k.s:v for k,v in zip(iitem.value.keys, iitem.value.values)}
                            plugin.plugin_name = cfgdict['plugin_name'].s
                            plugin.plugin_type = cfgdict['plugin_type'].s

                        # Is this the stack function dict?
                        if iitem.targets[0].id == retnames[1]:
                            stack_keys       = [el.s for el in iitem.value.keys]
                            stack_docs       = [el.elts[-1].s for el in iitem.value.values]
                            plugin.plugin_stack = zip(stack_keys, stack_docs)
    return plugin

def manage(cmd, plugin_name=''):
    if cmd == 'LIST':
        running   = set(active_plugins.keys())
        available = set(plugin_descriptions.keys()) - running
        text  = '\nCurrently running plugins: %s' % ', '.join(running)
        if len(available) > 0:
            text += '\nAvailable plugins: %s' % ', '.join(available)
        else:
            text += '\nNo additional plugins available.'
        return True, text
    if cmd not in ['LOAD', 'REMOVE']:
        return False
    p = plugin_descriptions.get(plugin_name)
    if not p:
        return False, 'Plugin %s not found'
    if cmd == 'LOAD':
        if plugin_name in active_plugins:
            return False, 'Plugin %s already loaded' % plugin_name
        return load(plugin_name, p)
    if cmd == 'REMOVE':
        if plugin_name not in active_plugins:
            return False, 'Plugin %s not loaded' % plugin_name
        return remove(plugin_name, p)
    return False

def init():
    # Add plugin path to module search path
    sys.path.append(path.abspath(settings.plugin_path))
    # Set plugin type for this instance of BlueSky
    req_type = 'sim' if settings.node_only else 'gui'
    # Find available plugins
    for fname in glob(path.join(settings.plugin_path, '*.py')):
        p = check_plugin(fname)
        # This is indeed a plugin, and it is meant for us
        if p and p.plugin_type == req_type:
            plugin_descriptions[p.plugin_name.upper()] = p
    # Load plugins selected in config
    for pname in settings.enabled_plugins:
        pname = pname.upper()
        p = plugin_descriptions.get(pname)
        if not p:
            print 'Error loading plugin: plugin %s not found.' % pname
        else:
            success = load(pname, p)
            print success[1]

if settings.node_only:
    # Sim implementation of plugin management
    preupdate_funs = dict()
    update_funs    = dict()

    def load(name, descr):
        try:
            # Load the plugin
            mod    = imp.find_module(descr.module_name, [descr.module_path])
            plugin = imp.load_module(descr.module_name, *mod)
            # Initialize the plugin
            config, stackfuns    = plugin.init_plugin()
            active_plugins[name] = plugin
            dt     = max(config.get('update_interval', 0.0), bs.sim.simdt)
            prefun = config.get('preupdate', None)
            updfun = config.get('update', None)
            if prefun:
                preupdate_funs[name] = [bs.sim.simt + dt, dt, prefun]
            if updfun:
                update_funs[name]    = [bs.sim.simt + dt, dt, updfun]
            # Add the plugin's stack functions to the stack
            bs.stack.append_commands(stackfuns)
            return True, 'Successfully loaded %s' % name
        except ImportError as e:
            print 'BlueSky plugin system failed to load', name, ':', e
            return False, 'Failed to load %s' % name

    def remove(name, descr):
        cmds, docs = zip(*descr.plugin_stack)
        bs.stack.remove_commands(cmds)
        active_plugins.pop(name)
        preupdate_funs.pop(name)
        update_funs.pop(name)

    def preupdate(simt):
        for fun in preupdate_funs:
            # Call function if its update interval has passed
            if simt >= fun[0]:
                # Set the next triggering time for this function
                fun[0] += fun[1]
                # Call the function
                fun[2]()

    def update(simt):
        for fun in update_funs.values():
            # Call function if its update interval has passed
            if simt >= fun[0]:
                # Set the next triggering time for this function
                fun[0] += fun[1]
                # Call the function
                fun[2]()

    def reset():
        for fun in preupdate_funs.values():
            fun[0] = 0.0

        for fun in update_funs:
            fun[0] = 0.0

else:
    def load(name, descr):
        pass

    def remove(name, descr):
        pass
