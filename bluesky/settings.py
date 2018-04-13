'''BlueSky global configuration module'''
import os
import sys
import shutil

def init():
    '''Initialize configuration.
       Import config settings from settings.cfg if this exists, if it doesn't
       create an initial config file'''
    rundir = ''
    srcdir = ''
    # Determine gui preference from whether bluesky was started with
    # BlueSky.py, BlueSky_qtgl.py, or BlueSky_pygame.py
    gui = 'pygame' if 'pygame' in sys.argv[0] \
        else ('qtgl' if 'qtgl' in sys.argv[0] or 'pytest' in sys.argv[0]
              else 'ask')

    # If BlueSky is run from a compiled bundle instead of from source, adjust the startup path
    # and change the path of configurable files to $home/bluesky
    if getattr(sys, 'frozen', False):
        srcdir = os.path.dirname(sys.executable)
        rundir = os.path.join(os.path.expanduser('~'), 'bluesky')

    cachedir   = os.path.join(rundir, 'data/cache')
    badadir    = os.path.join(rundir, 'data/performance/BADA')
    badasrc    = os.path.join(srcdir, 'data/performance/BADA')
    perfdir    = os.path.join(srcdir, 'data/performance')
    gfxdir     = os.path.join(srcdir, 'data/graphics')
    navdir     = os.path.join(srcdir, 'data/navdata')
    scnsrc     = os.path.join(srcdir, 'scenario')
    scndir     = os.path.join(rundir, 'scenario')
    outdir     = os.path.join(rundir, 'output')
    plgsrc     = os.path.join(srcdir, 'plugins')
    plgdir     = os.path.join(rundir, 'plugins')
    configfile = os.path.join(rundir, 'settings.cfg')
    configsrc  = os.path.join(srcdir, 'data/default.cfg')

    # Check if alternate config file is passed
    for i in range(len(sys.argv)):
        if len(sys.argv) > i + 1:
            if sys.argv[i] == '--config-file':
                configfile = sys.argv[i + 1]
            elif sys.argv[i] == '--scenfile':
                globals()['scenfile'] = sys.argv[i + 1]

    # Create config file if it doesn't exist yet. Ask for gui settings if bluesky
    # was started with BlueSky.py
    if not os.path.isfile(configfile):
        print()
        print('No config file settings.cfg found in your BlueSky starting directory!')
        print()
        print('This config file contains several default settings related to the simulation loop and the graphics.')
        print('A default version will be generated, which you can change if necessary before the next time you run BlueSky.')
        print()
        if gui == 'ask':
            print('BlueSky has several user interfaces to choose from. Please select which one to start by default.')
            print('You can always change this behavior by changing the settings.cfg file.')
            print()
            print('1. QtGL:    This is the most current interface of BlueSky, but requires a graphics card that supports at least OpenGL 3.3.')
            print('2. Pygame:  Use this version if your pc doesn\'t support OpenGL 3.3.')
            # print '3. Console: Run a console-only version of BlueSky. This is useful if you want to do batch simulations on a remote server.'
            print()
            ans = int(input('Default UI version: '))
            if ans == 1:
                gui = 'qtgl'
            elif ans == 2:
                gui = 'pygame'
            # elif ans == 3:
            #     gui = 'console'

        with open(configsrc, 'r') as fin, open(configfile, 'w') as fout:
            for line in fin:
                if line[:3] == 'gui':
                    line = "gui = '" + gui + "'\n"
                elif line[:10] == 'cache_path':
                    line = "cache_path = '" + cachedir.replace('\\', '/') + "'\n"
                elif line[:8] == 'log_path':
                    line = "log_path = '" + outdir.replace('\\', '/') + "'\n"
                elif line[:13] == 'scenario_path':
                    line = "scenario_path = '" + scndir.replace('\\', '/') + "'\n"
                elif line[:11] == 'plugin_path':
                    line = "plugin_path = '" + plgdir.replace('\\', '/') + "'\n"
                elif line[:14] == 'perf_path_bada':
                    line = "perf_path_bada = '" + badadir.replace('\\', '/') + "'\n"
                elif line[:9] == 'perf_path':
                    line = "perf_path = '" + perfdir.replace('\\', '/') + "'\n"
                elif line[:8] == 'gfx_path':
                    line = "gfx_path = '" + gfxdir.replace('\\', '/') + "'\n"
                elif line[:12] == 'navdata_path':
                    line = "navdata_path = '" + navdir.replace('\\', '/') + "'\n"

                fout.write(line)

    else:
        print('Reading config from settings.cfg')

    exec(compile(open(configfile).read(), configfile, 'exec'), globals())
    if not gui == 'ask':
        globals()['gui'] = gui
    elif 'gui' not in globals():
        globals()['gui'] = 'qtgl'

    # Update cachedir with python version-specific subfolder
    cachedir = os.path.join(cachedir, 'py%d' % sys.version_info[0])
    globals()['cache_path'] = cachedir

    # Create default directories if they don't exist yet
    for d in (outdir, cachedir):
        if not os.path.isdir(d):
            print('Creating directory "%s"' % d)
            os.makedirs(d)
    for d in [(badasrc, badadir), (scnsrc, scndir), (plgsrc, plgdir)]:
        if not os.path.isdir(d[1]):
            print('Creating directory "%s", and copying default files' % d[1])
            shutil.copytree(*d)

    return True

def set_variable_defaults(**kwargs):
    ''' Register a default value for a configuration variable. Use this functionality
        in plugins to make sure that configuration variables are available upon usage.

        Example:
            from bluesky import settings
            settings.set_variable_defaults(var1=1.0, var2=[1, 2, 3])

            This will make settings.var1 and settings.var2 available, with the
            provided default values.'''
    for key, value in kwargs.items():
        if key not in globals():
            globals()[key] = value

# Call settings.init() at creation
gui = ''
initialized = init()

### Parse command-line arguments ###
# This file is used to start the gui mainloop, a single node simulation loop,
# or, in case of the pygame version, both.
is_detached = ('--detached' in sys.argv)
is_client = ('--client' in sys.argv)
is_headless = ('--headless' in sys.argv)
is_sim = ('--sim' in sys.argv) or gui == 'pygame'
is_gui = not (is_sim or is_headless) or gui == 'pygame'
start_server = not (is_client or is_sim or gui == 'pygame')
if ('--discoverable' in sys.argv or is_headless):
    enable_discovery = True
