# Select the gui implementation. options: 'qtgl', 'pygame'
# Try the pygame implementation if you are having issues with qtgl.
gui = 'qtgl'

# Select the scenario path
scenario_path = 'data/scenario'

# Simulation timestep [seconds]
simdt = 0.02

# Simulation thread update rate [Hz]
sim_update_rate = 50

#=============================================================================
#=   QTGL Gui specific settings below
#=   Pygame Gui options in /data/grapics/scr_cfg.dat
#=============================================================================

# Radarscreen font size in pixels
text_size = 16

# The size in pixels of the font texture of one character (Change this if the font scaling doesn't look good)
text_texture_size = 62

# Font for the radar screen. The default is courier, because it is monospaced.
font_family = 'Courier'

# Font weight. A weight of 0 is ultralight, whilst 99 will be an extremely black. 50 is normal, and 75 is bold.
font_weight = 60

# Radarscreen airport symbol size in pixels
apt_size = 16

# Radarscreen waypoint symbol size in pixels
wpt_size = 16

# Radarscreen aircraft symbol size in pixels
ac_size = 20

# END OF SETTINGS

# Import conig settings from settings.cfg if this exists, if it doesn't create an initial config file
import os
if not os.path.isfile('settings.cfg'):
    print 'No config file settings.cfg found in your BlueSky starting directory!'
    print 'This config file contains several default settings related to the simulation loop and the graphics'
    print 'You can specify your own settings or use the default.'
    print 'Leave empty to use the default settings.'
    manual_input = (raw_input('Do you want to want to define your own settings? (yes/[no]: ').lower().find('y') >= 0)
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

                elif line[:2] == '# ':
                    # Variable info: also print
                    print line.strip('# ')

            lines += line + '\n'
            line = fin.readline().strip('\n')

    with open('settings.cfg', 'w') as fout:
        fout.write(lines)
else:
    print 'Reading config from settings.cfg'

execfile('settings.cfg')
