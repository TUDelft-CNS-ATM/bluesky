# Select the gui implementation. options: 'qtgl', 'pygame'
# Try the pygame implementation if you are having issues with qtgl.
gui = 'qtgl'

# Select the performance model. options: 'bluesky', 'bada'
performance_model = 'bluesky'

# Indicate the scenario path
scenario_path = 'scenario'

# Indicate the path for the aircraft performance data
perf_path = 'data/coefficients/BS_aircraft'

# Indicate the path for the BADA aircraft performance data (leave empty if BADA is not available)
perf_path_bada = 'data/coefficients/BADA'


# Simulation timestep [seconds]
simdt = 0.05

# Simulation thread update rate [Hz]
sim_update_rate = 20

#=============================================================================
#=   QTGL Gui specific settings below
#=   Pygame Gui options in /data/graphics/scr_cfg.dat
#=============================================================================

# Radarscreen font size in pixels
text_size = 10

# The size in pixels of the font texture of one character (Change this if the font scaling doesn't look good)
text_texture_size = 62

# Font for the radar screen. The default is courier, because it is monospaced.
font_family = 'Courier'

# Font weight. A weight of 0 is ultralight, whilst 99 will be an extremely black. 50 is normal, and 75 is bold.
font_weight = 99

# Radarscreen airport symbol size in pixels
apt_size = 10

# Radarscreen waypoint symbol size in pixels
wpt_size = 10

# Radarscreen aircraft symbol size in pixels
ac_size = 16

# END OF SETTINGS

# Import config settings from settings.cfg if this exists, if it doesn't create an initial config file
import os, sys
configfile = 'settings.cfg'
for i in range(len(sys.argv)):
    if sys.argv[i] == '--config-file':
        configfile = sys.argv[i+1]
        break

if not os.path.isfile(configfile):
    print
    print 'No config file settings.cfg found in your BlueSky starting directory!'
    print
    print 'This config file contains several default settings related to the simulation loop and the graphics'
    print 'You can specify your own settings or use the default.'
    print 'Leave empty to use the default settings.'
    print
    manual_input = (raw_input('Do you want to want to define your own settings? (yes/[no]): ').lower().find('y') >= 0)
    print
    lines = ''
    with open(os.path.dirname(__file__) + '/settings.py') as fin:
        line = fin.readline().strip('\n')
        while line[:5] != '# END':
            if manual_input:
                if len(line) > 0 and line[0] != '#' and line.find('=') >= 0:
                    # Get input from user
                    c = line.split('=')
                    ans = raw_input('[' + c[1].strip(' \'') + ']: ')
                    if len(ans) > 0:
                        line = c[0] + ' = '
                        if c[1].find('\'') >= 0:
                            ans = '\'' + ans + '\''
                        line += ans
                    print

                elif line[:2] == '# ':
                    # Variable info: also print
                    print line.strip('# ')

            lines += line + '\n'
            line = fin.readline().strip('\n')

    with open(configfile, 'w') as fout:
        fout.write(lines)
else:
    print 'Reading config from settings.cfg'

execfile(configfile)

#if len(sys.argv) > 1:
#    args = str.join(',', sys.argv[1:])
#    if args.find('pygame'):
#        gui = 'pygame'
#    elif args.find('qtgl'):
#        gui = 'qtgl'
