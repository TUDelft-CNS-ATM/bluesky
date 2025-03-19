import os
import inspect


class FuncObjectMeta(type):
    def __call__(cls, func, *args, **kwargs):
        # Retrieve preexisting FuncObject, if it exists
        fobj = getattr(inspect.unwrap(func), '__func_object__', None)
        # Don't return the preexisting FuncObject if func is a method bound to an instance that is
        # not the first instance of its type.
        if fobj is not None:
            if not inspect.ismethod(fobj.func) or func is fobj.func:
                return fobj
        return super().__call__(func, *args, **kwargs)



class FuncObject(metaclass=FuncObjectMeta):
    ''' Function reference object that is automatically updated
        on implementation selection for replaceables, and on creation of
        instances.
    '''
    __slots__ = ['func', 'callback']

    def __init__(self, func) -> None:
        ifunc = inspect.unwrap(func, stop=lambda f:not isinstance(f, (staticmethod, classmethod)))
        self.update(ifunc)
        ufunc = inspect.unwrap(func)
        setattr(getattr(ufunc, '__func__', ufunc), '__func_object__', self)

    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs)

    def __repr__(self) -> str:
        return repr(self.func)

    def __eq__(self, value) -> bool:
        return self is value or \
            self.func == inspect.unwrap(value)

    def notimplemented(self, *args, **kwargs):
        if self.func is None:
            print('Trying to call callback without assigned function or method')
        else:
            print(f'Trying to call method {self.func.__name__} for uninstantiated object')

    def update(self, func):
        self.func = func
        self.callback = func if self.valid else self.notimplemented

    def info(self):
        msg = ''
        if self.func.__name__ == '<lambda>':
            msg += 'Anonymous (lambda) function, implemented in '
        else:
            msg += f'Function {self.func.__name__}(), implemented in '
        if hasattr(self.func, '__code__'):
            fname = self.func.__code__.co_filename
            fname_stripped = fname.replace(os.getcwd(), '').lstrip('/')
            firstline = self.func.__code__.co_firstlineno
            msg += f'<a href="file://{fname}">{fname_stripped} on line {firstline}</a>'
        else:
            msg += f'module {self.func.__module__}'

        return msg

    @property
    def __wrapped__(self):
        return self.func

    @property
    def __name__(self):
        return self.func.__name__

    @property
    def valid(self):
        if self.func is None:
            return False
        spec = inspect.signature(self.func)
        # Check if this is an unbound class/instance method
        return spec.parameters.get('self') is None and \
            spec.parameters.get('cls') is None
