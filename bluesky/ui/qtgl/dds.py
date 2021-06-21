''' Very incomplete implementation of a DirectDraw texture file loader.
    Basically only able to load BlueSky's world texture, which is in DXT1 format.

    DDS file format taken from
    https://docs.microsoft.com/en-us/windows/desktop/direct3ddds/dx-graphics-dds-pguide
'''
from ctypes import c_char, c_uint32, Structure, sizeof

# DDS flags
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000
DDSD_DEPTH = 0x800000

# Pixel format flags
DDPF_ALPHAPIXELS = 0x1
DDPF_ALPHA = 0x2
DDPF_FOURCC = 0x4
DDPF_RGB = 0x40
DDPF_YUV = 0x200
DDPF_LUMINANCE = 0x20000

class PixelFormat(Structure):
    _fields_ = [('dwSize', c_uint32), ('dwFlags', c_uint32),
        ('dwFourCC', c_char * 4), ('dwRGBBitCount', c_uint32),
        ('dwRBitMask', c_uint32), ('dwGBitMask', c_uint32),
        ('dwBBitMask', c_uint32), ('dwABitMask', c_uint32)]


class DDSHeader(Structure):
    _fields_ = [('dwSize', c_uint32), ('dwFlags', c_uint32),
        ('dwHeight', c_uint32), ('dwWidth', c_uint32),
        ('dwPitchOrLinearSize', c_uint32), ('dwDepth', c_uint32),
        ('dwMipMapCount', c_uint32), ('dwReserved1', c_uint32 * 11),
        ('ddspf', PixelFormat), ('dwCaps', c_uint32),
        ('dwCaps2', c_uint32), ('dwCaps3', c_uint32),
        ('dwCaps4', c_uint32), ('dwReserved2', c_uint32)]


class DX10Header(Structure):
    _fields_ = [('dxgiFormat', c_uint32), ('resourceDimension', c_uint32),
        ('miscFlag', c_uint32), ('arraySize', c_uint32),
        ('miscFlags2', c_uint32)]


class DDSError(Exception):
    pass


class DDSTexture:
    ''' Loader class for Direct Draw texture files. '''
    def __init__(self, fname):
        header = DDSHeader()
        dx10header = None
        with open(fname, 'rb') as fin:
            # Read magic number
            magic = fin.read(4)
            if magic != b'DDS ':
                raise DDSError('File not recognised as DDS texure')

            # Read DDS file header
            size = fin.readinto(header)
            if size != sizeof(header):
                raise DDSError('File not recognised as DDS texure')

            # If extended format, extract extended header
            if header.ddspf.dwFourCC == b'DX10':
                dx10header = DX10Header()
                size = fin.readinto(dx10header)
                if size != sizeof(dx10header):
                    raise DDSError('File not recognised as DDS texure')

            # Keep useful attributes
            self.width = header.dwWidth
            self.height = header.dwHeight
            if header.dwCaps & DDSD_MIPMAPCOUNT:
                self.mipmapcount = header.dwMipMapCount
            else:
                self.mipmapcount = 0

            # load data
            self.is_compressed = (header.ddspf.dwFlags & DDPF_FOURCC != 0)

            # Calculate size of current texture level
            calcsize = lambda w, h: max(1, ((w + 3) // 4)) * max(1, ((h + 3) // 4)) * 8

            # Read main texture and possible mipmaps
            w, h = self.width, self.height
            self.data = fin.read(calcsize(w, h))
            self.mipmaps = list()
            for _ in range(self.mipmapcount):
                w = w // 2
                h = h // 2
                self.mipmaps.append((tsize, fin.read(calcsize(w, h))))
