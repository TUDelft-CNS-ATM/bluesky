from pathlib import Path
from bluesky.core import Entity
from bluesky.stack import command
from bluesky.stack.simstack import process, merge

class Importer(Entity):
    @command(name='IMPORT')
    @staticmethod
    def importcmd(fname:str='', ftype:str=''):
        ''' Importer: file importer for 3rd party scenario data.
        
            Arguments:
            - fname: The filename of the file to import
            - ftype: The type of the file (optional)
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
        if ftype:
            impl = Importer.derived().get(ftype)
            if impl is None:
                return False, f"No importer found for file type {ftype}"
            importer = impl.implinstance()
        else:
            ext = Path(fname).suffix.strip('.')
            for impl in Importer.derived().values():
                if impl is not Importer and ext in impl.implinstance().extensions:
                    importer = impl.implinstance()
                    break
            else:
                return False, f"No importer found for file with extension {ext}"

        # Load the data, and merge with current scenario stack
        scentime, scencmd = importer.load(fname)
        # If there are no timestamps, process immediately
        if not scentime:
            process(zip(scencmd, len(scencmd) * [None]))
            return True
        # If we have timestamps, merge the commands with the existing scenario stack
        merge(zip(scentime, scencmd))
        return True

    def __init__(self, filetype, extensions=None):
        super().__init__()
        self.filetype = filetype
        self.extensions = extensions or tuple()

    def load(fname):
        ''' Importer file loading.
        
            Arguments:
            - fname: Name of the file to load.

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
