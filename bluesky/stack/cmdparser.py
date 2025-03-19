''' Stack Command implementation. '''
import inspect
from typing import Dict

from bluesky.core.funcobject import FuncObject
from bluesky.network.publisher import state_publisher
from bluesky.stack.argparser import Parameter, getnextarg, ArgumentError


class Command:
    ''' Stack command object. '''
    # Dictionary with all command objects
    cmddict: Dict[str, 'Command'] = dict()

    @staticmethod
    @state_publisher(topic='STACKCMDS')
    def pubcmdlist():
        ''' Send a dictionary with available stack commands when requested. '''
        return {'cmddict': {cmd : val.brief[len(cmd) + 1:] for cmd, val in Command.cmddict.items()}}

    @classmethod
    def addcommand(cls, func, parent=None, name='', **kwargs):
        ''' Add 'func' as a stack command. '''
        # Stack command name
        fname = inspect.unwrap(func).__name__.upper()
        name = (name.upper() or fname)

        # When a parent is passed this function is a subcommand or alt command
        target = Command.cmddict if parent is None else parent

        # Check if this command already exists
        cmdobj = target.get(name)
        if not cmdobj:
            # If command doesn't exist yet create it, and put it in the command
            # dict under its name and all its aliases
            cmdobj = cls(func, name, **kwargs)
            target[name] = cmdobj
            for alias in cmdobj.aliases:
                target[alias] = cmdobj
        elif cls is CommandGroup and isinstance(cmdobj, CommandGroup):
            # Multiple calls to @commandgroup should add command alternatives
            # TODO: @commandgroup repeated calls in subclasses should not create
            # alt commands
            altcmd = Command(func, name, parent=cmdobj, **kwargs)
            cmdobj.altcmds[func.__name__.upper()] = altcmd
            cmdobj = altcmd
        else:
            # for subclasses reimplementing stack functions we keep only one
            # Command object
            print(f'Attempt to reimplement {name} from {cmdobj.callback} to {func}')
            if not isinstance(cmdobj, cls):
                raise TypeError(f'Error reimplementing {name}: '
                                f'A {type(cmdobj).__name__} cannot be '
                                f'reimplemented as a {cls.__name__}')

    def __init__(self, func, name='', parent=None, **kwargs):
        self.name = name
        self.help = inspect.cleandoc(kwargs.get('help', ''))
        self.brief = kwargs.get('brief', '')
        self.aliases = kwargs.get('aliases', tuple())
        self.annotations = get_annot(kwargs.get('annotations', ''))
        self.params = list()
        self._callback = None
        self.callback = func

    def __call__(self, argstring):
        args = []
        param = None
        # Use callback-specified parameter parsers to generate param list from strings
        for param in self.params:
            result = param(argstring)
            argstring = result[-1]
            args.extend(result[:-1])

        # Parse repeating final args
        while argstring:
            if param is None or not param.gobble:
                msg = f'{self.name} takes {len(self.params)} argument'
                if len(self.params) > 1:
                    msg += 's'
                count = len(self.params)
                while argstring:
                    _, argstring = getnextarg(argstring)
                    count += 1
                msg += f', but {count} were given'
                raise ArgumentError(msg)
            result = param(argstring)
            argstring = result[-1]
            args.extend(result[:-1])

        # Call callback function with parsed parameters
        ret = self.callback(*args)
        # Always return a tuple with a success value and a message string
        if ret is None:
            return True, ''
        if isinstance(ret, (tuple, list)) and ret:
            if len(ret) > 1:
                # Assume that (success, echotext) is returned
                return ret[:2]
            ret = ret[0]
        return ret, ''

    def __repr__(self):
        if self.valid:
            return f'<Stack Command {self.name}, callback={self.callback}>'
        return f'<Stack Command {self.name} (invalid), callback=unbound method {self.callback}'

    @property
    def callback(self):
        ''' Callback pointing to the actual function that implements this
            stack command.
        '''
        return self._callback

    @callback.setter
    def callback(self, function):
        self._callback = FuncObject(function)
        spec = inspect.signature(function)
        self.brief = self.brief or (
            self.name + ' ' + ','.join(spec.parameters))
        self.help = self.help or inspect.cleandoc(
            inspect.getdoc(function) or '')
        paramspecs = list(filter(Parameter.canwrap, spec.parameters.values()))
        if self.annotations:
            self.params = list()
            pos = 0
            for annot, isopt in self.annotations:
                if annot == '...':
                    if paramspecs[-1].kind != paramspecs[-1].VAR_POSITIONAL:
                        raise IndexError('Repeating arguments (...) given for function'
                                            ' not ending in starred (variable-length) argument')
                    self.params[-1].gobble = True
                    break

                param = Parameter(paramspecs[pos], annot, isopt)
                if param:
                    pos = min(pos + param.size(), len(paramspecs) - 1)
                    self.params.append(param)
            if len(self.params) > len(paramspecs) and \
                paramspecs[-1].kind != paramspecs[-1].VAR_POSITIONAL:
                raise IndexError(f'More annotations given than function '
                                    f'{self.callback.__name__} has arguments.')
        else:
            self.params = [p for p in map(Parameter, paramspecs) if p]

    def helptext(self, subcmd=''):
        ''' Return complete help text. '''
        msg = f'<div style="white-space: pre;">{self.help}</div>\nUsage:\n{self.brief}'
        if self.aliases:
            msg += ('\nCommand aliases: ' + ','.join(self.aliases))
        return msg + '\n' + self.callback.info()

    def brieftext(self):
        ''' Return the brief usage text. '''
        return self.brief


class CommandGroup(Command):
    ''' Stack command group object.
        Command groups can have subcommands.
    '''
    def __init__(self, func, name='', **kwargs):
        super().__init__(func, name, **kwargs)
        self.subcmds = dict()
        self.altcmds = dict()

        # Give the function a method to add subcommands
        func.subcommand = lambda fun=None, **kwargs: command(
            fun, parent=self.subcmds, **kwargs)

        # Give the function a method to add alternative command implementations
        func.altcommand = lambda fun=None, **kwargs: command(
            fun, parent=self.altcmds, **kwargs)

    def __call__(self, strargs):
        # First check subcommand
        if strargs:
            subcmd, subargs = getnextarg(strargs)
            subcmdobj = self.subcmds.get(subcmd.upper())
            if subcmdobj:
                return subcmdobj(subargs)

        ret = super().__call__(strargs)
        msg = ret[1] if isinstance(ret, tuple) else ''
        success = ret[0] if isinstance(ret, tuple) else \
                  ret if isinstance(ret, bool) else True

        if not success:
            for altcmdobj in self.altcmds.values():
                ret = altcmdobj(strargs)
                success = ret[0] if isinstance(ret, tuple) else \
                          ret if isinstance(ret, bool) else True
                if isinstance(ret, tuple):
                    msg += '\n' + ret[1]
                if success:
                    return ret
        return success, msg


    def helptext(self, subcmd=''):
        ''' Return complete help text. '''
        if subcmd:
            obj = self.subcmds.get(subcmd)
            return obj.helptext() if obj else f'{subcmd} is not a subcommand of {self.name}'

        msg = super().helptext()
        if self.subcmds:
            msg += '\nSubcommands:'
            for cmdobj in self.subcmds.values():
                msg += f'\n{self.name} {cmdobj.brief} ({cmdobj.callback.info()})'
        if self.altcmds:
            msg += '\nAlternative command implementations:'
            for cmdobj in self.altcmds.values():
                msg += f'\n{self.name} {cmdobj.brief} ({cmdobj.callback.info()})'

        return msg

    def brieftext(self):
        ''' Return the brief usage text. '''
        msg = self.brief
        for subcmd in self.subcmds.values():
            msg += f'\n{self.name} {subcmd.brief}'
        return msg

def commandgroup(fun=None, parent=None, **kwargs):
    ''' BlueSky stack command decorator for command groups.

        Functions decorated with this decorator become available as stack
        functions, and have the ability to have subcommands.
    '''
    return command(fun, CommandGroup, parent, **kwargs)


def command(fun=None, cls=Command, parent=None, **kwargs):
    ''' BlueSky stack command decorator.

        Functions decorated with this decorator become available as stack
        functions.
    '''
    def deco(fun):
        # Construct the stack command object, but return the original function
        cls.addcommand(fun, parent, **kwargs)
        return fun
    # Allow both @command and @command(args)
    return deco(fun) if fun else deco


def append_commands(newcommands, syndict=None):
    """ Append additional functions to the stack command dictionary """
    for name, (brief, annotations, fun, hlp) in newcommands.items():
        if syndict:
            aliases = tuple(k for k,v in syndict.items() if v == name)
        else:
            aliases = tuple()
        # Use the command decorator function to register each new command
        Command.addcommand(fun, name=name, annotations=annotations, brief=brief, help=hlp, aliases=aliases)


def remove_commands(commands):
    """ Remove functions from the stack """
    for cmd in commands:
        Command.cmddict.pop(cmd)


def get_commands():
    """ Return the stack dictionary of commands. """
    return Command.cmddict


def get_annot(annotations):
    ''' Get annotations from string, or tuple/list. '''
    if isinstance(annotations, (tuple, list)):
        return tuple(annotations)
    # Assume it is a comma-separated string
    argtypes = []

    # Process and reduce annotation string from left to right
    # First cut at square brackets, then take separate argument types
    while annotations:
        opt = annotations[0] == "["
        cut = (annotations.find("]") if opt else annotations.find(
            "[") if "[" in annotations else len(annotations))

        types = annotations[:cut].strip("[,]").split(",")
        # Returned argtypes are tuples of type and optional status
        argtypes += zip(types, [opt or t == "..." for t in types])
        annotations = annotations[cut:].lstrip(",]")

    return tuple(argtypes)
