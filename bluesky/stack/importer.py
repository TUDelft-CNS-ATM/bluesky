from pathlib import Path
from bluesky.core import Entity
from bluesky.stack import commandgroup
from bluesky.stack.simstack import process, merge

class Importer(Entity):
    @commandgroup(name='IMPORT')
    @staticmethod
    def importcmd(fname:str='', *args):
        ''' Importer: file importer for 3rd party scenario data.
        
            Arguments:
            - fname: The filename of the file to import
            - args: Optional arguments to perform e.g., wildcard replacement
        '''
        if not fname:
            ret = Importer.importcmd.__doc__
            ret += '\nKnown importers:'
            for impl in Importer.derived().values():
                if impl is not Importer:
                    importer = impl.implinstance()
                    ret += f'\n{importer.filetype} (extensions: {", ".join(importer.extensions)})'
            return True, ret
        # Get the correct importer implementation
        ext = Path(fname).suffix.strip('.')
        for impl in Importer.derived().values():
            if impl is not Importer and ext in impl.implinstance().extensions:
                importer = impl.implinstance()
                break
        else:
            return False, f"No importer found for file with extension {ext}"

        return importer._load_and_process(fname)

    def __init__(self, filetype, extensions=None):
        super().__init__()
        self.filetype = filetype
        self.extensions = extensions or tuple()

    def _load_and_process(self, fname, *args):
        # Load the data, and merge with current scenario stack
        scentime, scencmd = self.load(fname, *args)
        # If there are no timestamps, process immediately
        if not scentime:
            process(zip(scencmd, len(scencmd) * [None]))
            return True, f'Successfully loaded {fname}'
        # If we have timestamps, merge the commands with the existing scenario stack
        merge(zip(scentime, scencmd))
        return True, f'Successfully loaded and merged {fname}'

    def load(fname, *args):
        ''' Importer file loading.
        
            Arguments:
            - fname: Name of the file to load.
            - args: Optional arguments to perform e.g., wildcard replacement

            Returns:
            - (scentime, scencmd): Tuple of two lists:
                - scentime: list of timestamps (in (fractions of) seconds). 
                  May be an empty list if commands are not timestamped.
                - scencmd: list of corresponding stack commands
        '''
        return ([], [])

    def __init_subclass__(cls):
        # First do base class initialisation
        super().__init_subclass__(replaceable=False, skipbase=True)
        # Then immediately instantiate
        cls._instance = cls()
        # Add subcommand for this specific importer
        Importer.importcmd.subcommand(cls._instance._load_and_process, name=cls._instance.filetype.upper())
