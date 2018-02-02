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
    if i in [ord(c) for c in ["'", '"', ',', '.']]:
        call('./msdfgen msdf -font source-code-pro/SourceCodePro-Regular.otf %d -o font/%d.png -size 26 32 -pxrange 4 -scale 2.25882352941 -range 1.77083333333 -translate 1.03645833333 1.95833333333' % (i, i), shell=True)
    else:
        call('./msdfgen msdf -font source-code-pro/SourceCodePro-Regular.otf %d -o font/%d.png -size 26 32 -pxrange 4 -autoframe' % (i, i), shell=True)

# Arrows: use ascii code 30 and 31
call('./msdfgen msdf -font source-code-pro/SourceCodePro-Regular.otf 0x2191 -o font/30.png -size 26 32 -pxrange 4 -autoframe', shell=True)
call('./msdfgen msdf -font source-code-pro/SourceCodePro-Regular.otf 0x2193 -o font/31.png -size 26 32 -pxrange 4 -autoframe', shell=True)
