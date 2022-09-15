""" Autocomplete filenames in the BlueSky console."""
import bluesky as bs

previous_g = ''


# Case insensitive file search
def iglob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return list(bs.resource('scenario').glob(''.join(map(either, pattern))))


def reset():
    global previous_g
    previous_g = ''


def complete(cmd):
    """ Command line IC + filename completion function """
    lcmd = cmd.upper().split()
    newcmd = cmd
    displaytext = ''

    if lcmd[0] in ['IC', 'BATCH', 'CALL', 'PCALL']:
        global previous_g
        g = ''
        if len(lcmd) == 2:
            g += lcmd[1].strip()
        files = iglob(g + '*')

        if len(files) > 0:
            if len(files) == 1:
                newcmd = lcmd[0] + ' ' + files[0].name
            elif g == previous_g:
                displaytext += ' '.join(f.name for f in files)

            else:
                previous_g = g
                idx        = len(g)
                while idx < len(files[0].name) and len(files) == len(iglob(g + files[0].name[idx] + '*')):
                    g += files[0].name[idx].upper()
                    idx += 1

                newcmd = lcmd[0] + ' ' + g

    return newcmd, displaytext
