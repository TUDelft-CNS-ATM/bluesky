'''BlueSky global configuration module'''
import os
import sys
import shutil
import site


def init(cfgfile=''):
    '''Initialize configuration.
       Import config settings from settings.cfg if this exists, if it doesn't
       create an initial config file'''
    rundir = ''
    srcdir = ''

    # If BlueSky is run from a compiled bundle instead of from source,
    # adjust the startup path and change the path of
    # configurable files to $home/bluesky
    if getattr(sys, 'frozen', False):
        srcdir = os.path.dirname(sys.executable)
        rundir = os.path.join(os.path.expanduser('~'), 'bluesky')

    # If BlueSky is installed as a python package the location of the data files need to
    # be adjusted so that importing will not fail when copying config file below
    if not os.path.isfile(os.path.join(rundir, 'data/default.cfg')):
        # collate list of possible data install roots
        root_dirs = site.getusersitepackages()
        root_dirs = [root_dirs] if type(root_dirs) is str else root_dirs
        root_dirs += site.getsitepackages()

        # search for bluesky shared data directory
        for root_dir in root_dirs:
            dirpath = os.path.join(root_dir, 'share', 'bluesky')
            if os.path.exists(dirpath):
                srcdir = dirpath
                break

    datadir = os.path.join(rundir, 'data')
    cachedir = os.path.join(rundir, 'data/cache')
    badadir = os.path.join(rundir, 'data/performance/BADA')
    badasrc = os.path.join(srcdir, 'data/performance/BADA')
    perfdir = os.path.join(srcdir, 'data/performance')
    gfxdir = os.path.join(srcdir, 'data/graphics')
    navdir = os.path.join(srcdir, 'data/navdata')
    scnsrc = os.path.join(srcdir, 'scenario')
    scndir = os.path.join(rundir, 'scenario')
    outdir = os.path.join(rundir, 'output')
    plgsrc = os.path.join(srcdir, 'plugins')
    plgdir = os.path.join(rundir, 'plugins')
    configfile = os.path.join(rundir, 'settings.cfg')
    configsrc = os.path.join(srcdir, 'data/default.cfg')

    if not cfgfile:
        cfgfile = os.path.join(rundir, 'settings.cfg')
    
    # Check if alternate config file is passed
    for i in range(len(sys.argv)):
        if len(sys.argv) > i + 1:
            if sys.argv[i] == '--config-file':
                configfile = sys.argv[i + 1]
            elif sys.argv[i] == '--scenfile':
                globals()['scenfile'] = sys.argv[i + 1]

    # Create config file if it doesn't exist yet. Ask for gui settings if bluesky
    # was started with BlueSky.py
    if not os.path.isfile(cfgfile):
        print()
        print('No config file settings.cfg found in your BlueSky starting directory!')
        print()
        print('This config file contains several default settings related to the simulation loop and the graphics.')
        print('A default version will be generated, which you can change if necessary before the next time you run BlueSky.')
        print()

        with open(configsrc, 'r') as fin, open(cfgfile, 'w') as fout:
            for line in fin:
                if line[:9] == 'data_path':
                    line = "data_path = '" + datadir.replace('\\', '/') + "'\n"
                if line[:10] == 'cache_path':
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

    exec(compile(open(cfgfile).read(), cfgfile, 'exec'), globals())

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
            try:
                shutil.copytree(*d)
            except FileNotFoundError:
                print('Unable to copy "%s" files to "%s"' %(d[0], d[1]), file=sys.stderr)

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
