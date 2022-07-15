from bluesky.stack.importer import Importer


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'IMPORTEX',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class ImportExample(Importer):
    def __init__(self):
        super().__init__(filetype='Example', extensions=('txt', 'dat', 'log'))

    def load(self, fname):
        # Our file loader should return two lists:
        # - a list of timestamps (in (fractions of) seconds). This list is optional,
        #   for untimestamped commands use an empty list
        # - a list of stack commands
        return [], ['MCRE 1']
