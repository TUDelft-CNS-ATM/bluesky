''' BlueSky resource access '''
import shutil
import itertools
from pathlib import Path
try:
    from importlib.resources import files
    from importlib.readers import MultiplexedPath
except ImportError:
    # Python < 3.9 only provides deprecated resources API
    from importlib_resources import files
    from importlib_resources.readers import MultiplexedPath


class ResourcePath(MultiplexedPath):
    def __init__(self, *paths):
        base = files('bluesky.resources')
        paths = list(paths) + base._paths if isinstance(base, MultiplexedPath) else base
        super().__init__(*paths)

    def appendpath(self, path):
        self._paths.append(path)

    def insertpath(self, path, pos=0):
        self._paths.insert(pos, path)

    def bases(self):
        for p in self._paths:
            yield p

    def base(self, idx):
        return self._paths[idx]

    @property
    def nbases(self):
        return len(self._paths)

    def as_posix(self, idx=0):
        return self._paths[idx].as_posix()

    def glob(self, pattern: str):
        files = set()
        for res in itertools.chain.from_iterable(p.glob(pattern) for p in self._paths):
            if res.is_file() and res.name in files:
                continue
            files.add(res.name)
            yield res

    def joinpath(self, *descendants):
        # first try to find child in current paths
        paths = []
        if not descendants:
            return self
        
        for path in (p.joinpath(*descendants) for p in self._paths):
            if path.exists():
                if path.is_dir():
                    # If it's a dir, try to combine it with others
                    paths.append(path)
                else:
                    # If it's a file, immediately return it
                    return path

        # if it does not exist, construct it with the first path
        return ResourcePath(*paths) if len(paths) > 1 else \
            paths[0] if len(paths) == 1 else self._paths[0].joinpath(*descendants)
    
    __truediv__ = joinpath


def resource(*descendants):
    ''' Get a path pointing to a BlueSky resource.
    
        Arguments:
        - descendants: Zero or more path-like objects (Path or str)

        Returns:
        - Path pointing to resource (file or directory)
          If arguments form an absolute path it is returned directly,
          otherwise a path relative to BlueSky's resource paths is returned.
    '''
    ret = Path(*descendants)
    if ret.is_absolute():
        return ret

    return resource.path.joinpath(*descendants)
resource.path = ResourcePath()


def init(workdir=None):
    ''' Initialise BlueSky resource paths. '''
    if workdir is None:
        if files('bluesky').parent == Path.cwd():
            # Assume BlueSky is running from source, e.g., cloned from GitHub
            # In this case additional resources (scenarios, plugins)
            # are found in the current working directory
            workdir = Path.cwd()
        else:
            # Assume BlueSky is running as a pip package
            # In this case additional resources are found in $home/bluesky
            workdir = Path.home().joinpath('bluesky')

            # Initialise this folder if it doesn't exist
            if not workdir.exists():
                # Create directory and populate with basics
                print(f'Creating BlueSky base directory "{workdir.absolute()}"')
                workdir.mkdir()

    elif not Path(workdir).exists():
        print(f"Specified working directory {workdir} doesn't exist!")

    # Ensure existence of scenario, plugins, output, and cache directories
    for subdir in map(workdir.joinpath, ('scenario', 'plugins', 'output', 'cache')):
        if not subdir.exists():
            print(f'Creating directory "{subdir}"')
            subdir.mkdir()
    # Ensure existence of config file
    cfgfile = workdir.joinpath("settings.cfg")
    if not cfgfile.exists():
        print(f'Copying default configfile to {cfgfile}')
        shutil.copy(resource('default.cfg'), cfgfile)

    # Set correct search paths for resource function
    resource.path = ResourcePath(workdir)
