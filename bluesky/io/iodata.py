''' IOData: convenience object for BlueSky network communication. '''

class IOData(object):
    def __init__(self, objtype, *args, **kwargs):
        self._type = objtype
        self.__dict__.update(kwargs)
        if args:
            self.data = args

    def type(self):
        return self._type
