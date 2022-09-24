import pickle
import bluesky as bs

## Default settings
bs.settings.set_variable_defaults(cache_path='cache')


def openfile(*args):
    return CacheFile(*args)


class CacheError(Exception):
    ''' Exception class for CacheFile errors. '''
    pass


class CacheFile():
    ''' Convenience class for loading and saving pickle cache files. '''
    def __init__(self, fname, version_ref='1'):
        self.fname = bs.resource(bs.settings.cache_path).joinpath(fname)
        self.version_ref = version_ref
        self.file = None

    def check_cache(self):
        ''' Check whether the cachefile exists, and is of the correct version. '''
        if not self.fname.is_file():
            raise CacheError('Cachefile not found: ' + str(self.fname))

        self.file = open(self.fname, 'rb')
        version = pickle.load(self.file)

        # Version check
        if not version == self.version_ref:
            self.file.close()
            self.file = None
            raise CacheError('Cache file out of date: ' + str(self.fname))
        print('Reading cache:', self.fname)

    def load(self):
        ''' Load a variable from the cache file. '''
        if self.file is None:
            self.check_cache()

        return pickle.load(self.file)

    def dump(self, var):
        ''' Dump a variable to the cache file. '''
        if self.file is None:
            self.file = open(self.fname, 'wb')
            pickle.dump(self.version_ref, self.file, pickle.HIGHEST_PROTOCOL)
            print("Writing cache:", self.fname)
        pickle.dump(var, self.file, pickle.HIGHEST_PROTOCOL)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
