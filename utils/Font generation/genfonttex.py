''' BlueSky font generation tool for the QtGL version.

    This script can be used to generate custom font textures for the QtGL version
    of BlueSky. The fonts are based on Valve's multi-channel signed distance field method,
    see Valve's publication: http://www.valvesoftware.com/publications/2007/SIGGRAPH2007_AlphaTestedMagnification.pdf

    The distance field textures can be generated with the msdfgen tool, which can be found here:
    https://github.com/Chlumsky/msdfgen

    Currently, the default font for BlueSky is the SourceCodePro regular monospaced font, which can be found here:
    https://www.fontsquirrel.com/fonts/source-code-pro
    Generation settings: size=26x32, pxrange=4
'''
from subprocess import call


for i in range(32, 127):
    call('./msdfgen msdf -font source-code-pro/SourceCodePro-Regular.otf %d -o font/%d.png -size 26 32 -pxrange 4 -autoframe' % (i, i), shell=True)
