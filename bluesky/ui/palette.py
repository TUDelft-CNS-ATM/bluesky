''' BlueSky colour palette loader. '''
from os import path
from bluesky import settings
settings.set_variable_defaults(colour_palette='bluesky-lvnl', gfx_path='data/graphics')

def init():
    # Load the palette file selected in settings
    pfile = path.join(settings.gfx_path, 'palettes', settings.colour_palette)
    if path.isfile(pfile):
        print('Loading palette ' + settings.colour_palette)
        exec(compile(open(pfile).read(), pfile, 'exec'), globals())
        return True
    else:
        print('Palette file not found ' + pfile)
        return False


def set_default_colours(**kwargs):
    ''' Register a default value for a colour. Use this functionality in the source file
        where you intend to use those colours so that defaults are always available.

        Example:
            from bluesky.ui import palette
            palette.set_default_colours(mycolor1=(255, 0, 0), mycolor2=(0, 0, 0))

            This will make settings.mycolor1 and settings.mycolor2 available,
            with the provided default values.'''
    for key, value in kwargs.items():
        if key not in globals():
            globals()[key] = value

initialized = init()
