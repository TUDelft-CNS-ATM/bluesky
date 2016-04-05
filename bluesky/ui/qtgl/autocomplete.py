
import glob
from ...settings import scenario_path

previous_g = ''


# Case insensitive file search
def iglob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))


def reset():
    global previous_g
    previous_g = ''


def complete(cmd):
    """ Command line IC + filename completion function """
    lcmd = cmd.split()
    newcmd = cmd
    displaytext = ''

    if lcmd[0] == 'IC' or lcmd[0] == 'BATCH':
        global previous_g
        g = scenario_path
        striplen = len(g)
        if g[-1] != '/':
            g += '/'
            striplen += 1
        if len(lcmd) == 2:
            g += lcmd[1].strip()
        files = iglob(g + '*')

        if len(files) > 0:
            if len(files) == 1:
                newcmd = lcmd[0] + ' ' + files[0][striplen:]
            elif g == previous_g:
                for f in files:
                    displaytext += f[striplen:] + '  '
            else:
                previous_g = g
                idx        = len(g)

                while len(files) is len(iglob(g + files[0][idx] + '*')) and idx < len(files[0]):
                    g += files[0][idx].upper()
                    idx += 1

                newcmd = lcmd[0] + ' ' + g[striplen:]

    return newcmd, displaytext
