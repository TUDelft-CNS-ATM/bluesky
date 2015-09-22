
import glob
from ...settings import scenario_path

previous_g = ''


def complete(cmd):
    lcmd = cmd.split()
    newcmd = cmd
    displaytext = ''

    if lcmd[0] == 'IC':
        global previous_g
        g = scenario_path
        striplen = len(g)
        if g[-1] is not '/':
            g += '/'
            striplen += 1
        if len(lcmd) == 2:
            g += lcmd[1]

        files = glob.glob(g + '*')
        if len(files) > 0:
            if g == previous_g:
                for f in files:
                    displaytext += f[striplen:] + '  '
            else:
                previous_g = g

                idx = len(g)
                if len(files) == 1:
                    newcmd = 'IC ' + files[0][striplen:]
                else:
                    while len(files) is len(glob.glob(g + files[0][idx] + '*')) and idx < len(files[0]):
                        g += files[0][idx]
                        idx += 1

                    newcmd = 'IC ' + g[striplen:]

    return newcmd, displaytext
