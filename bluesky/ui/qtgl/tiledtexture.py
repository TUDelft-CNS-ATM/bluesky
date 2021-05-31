''' Tile texture manager for BlueSky Qt/OpenGL gui. '''
import math
from os import makedirs, path
import weakref
from collections import OrderedDict

from PyQt5.QtCore import QObject, QRunnable, QThread, QThreadPool, pyqtSignal
import numpy as np
from urllib.request import urlopen
from urllib.error import URLError

from PyQt5.QtGui import QImage

import bluesky as bs
from bluesky.core import Signal
import bluesky.ui.qtgl.glhelpers as glh


bs.settings.set_variable_defaults(
    tile_array_size=100,
    max_download_workers=2,
    tile_sources={
        'opentopomap': ['https://a.tile.opentopomap.org/{zoom}/{x}/{y}.png',
                        'https://b.tile.opentopomap.org/{zoom}/{x}/{y}.png',
                        'https://c.tile.opentopomap.org/{zoom}/{x}/{y}.png']
    })


class Tile:
    ''' Wrapper object for tile data and properties. '''

    def __init__(self, source, zoom, tilex, tiley, idxx, idxy):
        super().__init__()
        self.source = source
        self.zoom = zoom
        self.tilex = tilex
        self.tiley = tiley
        self.idxx = idxx
        self.idxy = idxy

        self.image = None
        # For the image data, check cache path first
        fpath = path.join(bs.settings.cache_path, source, str(zoom), str(tilex))
        fname = path.join(fpath, f'{tiley}.png')
        if path.exists(fname):
            self.image = QImage(fname).convertToFormat(QImage.Format_ARGB32)
        else:
            # Make sure cache directory exists
            makedirs(fpath, exist_ok=True)
            for url in bs.settings.tile_sources[source]:
                try:
                    url_request = urlopen(url.format(
                        zoom=zoom, x=tilex, y=tiley))
                    data = url_request.read()
                    self.image = QImage.fromData(
                        data).convertToFormat(QImage.Format_ARGB32)
                    with open(fname, 'wb') as fout:
                        fout.write(data)
                    break
                except URLError as e:
                    print(e)
                    pass


class TileLoader(QRunnable):
    class Signals(QObject):
        finished = pyqtSignal(Tile)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = TileLoader.Signals()

    def run(self):
        tile = Tile(*self.args, **self.kwargs)
        self.signals.finished.emit(tile)


class TiledTextureMeta(type(glh.Texture)):
    ''' TileTexture meta class that stores weak references to created textures so that they can be used in
        multiple objects, but are deleted when the last reference to them is removed. '''
    tiletextures = weakref.WeakValueDictionary()
    def __call__(cls, *args, **kwargs):
        name = kwargs.get('tilesource', 'opentopomap')
        if name not in cls.tiletextures:
            cls.tiletextures[name] = super().__call__(*args, **kwargs)
        return cls.tiletextures[name]


class TiledTexture(glh.Texture, metaclass=TiledTextureMeta):
    def __init__(self, glsurface, tilesource='opentopomap'):
        super().__init__(target=glh.Texture.Target2DArray)
        # TODO: take min of settings and tilesource limitations
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(bs.settings.max_download_workers)
        self.tilesource = tilesource
        self.tilesize = (256,256)
        self.curtiles = OrderedDict()
        self.fullscreen = False
        self.offsetscale = np.array([0, 0, 1], dtype=np.float32)
        self.bbox = list()
        self.glsurface = glsurface
        self.indextexture = glh.Texture(target=glh.Texture.Target2D)
        self.indexsampler_loc = 0
        self.arraysampler_loc = 0
        bs.net.actnodedata_changed.connect(self.actdata_changed)
        Signal('panzoom').connect(self.on_panzoom_changed)


    def add_bounding_box(self, lat0, lon0, lat1, lon1):
        if abs(lat1 - lat0) >= 178 and abs(lon1 - lon0) >= 358:
            self.fullscreen = True
        else:
            self.bbox.append((lat0, lon0, lat1, lon1))

    def create(self):
        if self.isCreated():
            return
        super().create()
        # Fetch a temporary tile image to get dimensions
        tmptile = Tile(self.tilesource, 1, 1, 1, 0, 0)
        img = tmptile.image
        self.setFormat(glh.Texture.RGBA8_UNorm)
        self.tilesize = (img.width(), img.height())
        self.setSize(img.width(), img.height())
        self.setLayers(bs.settings.tile_array_size)
        super().bind()
        self.allocateStorage()
        self.setWrapMode(glh.Texture.DirectionS,
                         glh.Texture.ClampToBorder)
        self.setWrapMode(glh.Texture.DirectionT,
                         glh.Texture.ClampToBorder)
        self.setMinMagFilters(glh.Texture.Linear, glh.Texture.Linear)

        # Initialize index texture
        # RG = texcoord offset, B = zoom factor, A = array index
        itexw = int(np.sqrt(bs.settings.tile_array_size) * 4 / 3 + 10)
        itexh = int(np.sqrt(bs.settings.tile_array_size) * 3 / 4 + 10)
        self.indextexture.create()
        self.indextexture.setFormat(glh.Texture.RGBA32I)
        self.indextexture.setSize(itexw, itexh)
        self.indextexture.bind(1)
        # self.indextexture.allocateStorage(glh.Texture.RGBA_Integer, glh.Texture.Int32)

        idxdata = np.array(itexw * itexh *
                           [(0, 0, 0, -1)], dtype=np.int32)
        glh.gl.glTexImage2D_alt(glh.Texture.Target2D, 0, glh.Texture.RGBA32I, itexw, itexh, False,
                          glh.Texture.RGBA_Integer, glh.Texture.Int32, idxdata.tobytes())
        # self.indextexture.setData(glh.Texture.RGBA_Integer, glh.Texture.Int32, sip.voidptr(idxdata))
        self.indextexture.setWrapMode(glh.Texture.DirectionS,
                         glh.Texture.ClampToBorder)
        self.indextexture.setWrapMode(glh.Texture.DirectionT,
                         glh.Texture.ClampToBorder)
        self.indextexture.setMinMagFilters(glh.Texture.Nearest, glh.Texture.Nearest)

        shader = glh.ShaderSet.get_shader('tiled')
        self.indexsampler_loc = shader.uniformLocation('tile_index')
        self.arraysampler_loc = shader.uniformLocation('tile_texture')

    def bind(self, unit):
        ''' Bind this texture for drawing. '''
        # Set sampler locations
        glh.ShaderProgram.bound_shader.setUniformValue(self.indexsampler_loc, 2)
        glh.ShaderProgram.bound_shader.setUniformValue(self.arraysampler_loc, 4)
        # Bind index texture to texture unit 0
        self.indextexture.bind(2)
        # Bind tile array texture to texture unit 1
        super().bind(4)

    def actdata_changed(self, nodeid, nodedata, changed_elems):
        ''' Update tile buffers when a different node is selected, or when
            the data of the current node is updated. '''
        # Update pan/zoom
        if 'PANZOOM' in changed_elems:
            self.on_panzoom_changed(True)

    def on_panzoom_changed(self, finished=False):
        # Check if textures need to be updated
        viewport = self.glsurface.viewportlatlon()
        surfwidth_px = self.glsurface.width()
        # First determine floating-point, hypothetical values to calculate the required tile zoom level
        # floating-point number of tiles that fit in the width of the view
        ntiles_hor = surfwidth_px / self.tilesize[0]
        # Estimated width in longitude of one tile
        est_tilewidth = abs(viewport[3] - viewport[1]) / ntiles_hor

        zoom = tilezoom(est_tilewidth)
        # With the tile zoom level get the required number of tiles
        # Top-left and bottom-right tiles:
        x0, y0 = latlon2tilenum(*viewport[:2], zoom)
        x1, y1 = latlon2tilenum(*viewport[2:], zoom)
        nx = abs(x1 - x0) + 1
        ny = abs(y1 - y0) + 1

        # Calculate the offset of the top-left tile w.r.t. the screen top-left corner
        tile0_topleft = np.array(tilenum2latlon(x0, y0, zoom))
        tile0_bottomright = np.array(tilenum2latlon(x0 + 1, y0 + 1, zoom))
        tilesize_latlon0 = np.abs(tile0_bottomright - tile0_topleft)
        offset_latlon0 = viewport[:2] - tile0_topleft
        tex_y0, tex_x0 = np.abs(offset_latlon0 / tilesize_latlon0)

        # Calculate the offset of the bottom-right tile w.r.t. the screen bottom right corner
        tile1_topleft = np.array(tilenum2latlon(x1, y1, zoom))
        tile1_bottomright = np.array(tilenum2latlon(x1 + 1, y1 + 1, zoom))
        tilesize_latlon1 = np.abs(tile1_bottomright - tile1_topleft)
        offset_latlon1 = viewport[2:] - tile1_topleft
        tex_y1, tex_x1 = np.abs(offset_latlon1 / tilesize_latlon1) + [ny - 1, nx - 1]
        # Store global offset and scale for shader uniform
        self.offsetscale = np.array(
            [tex_x0, tex_y0, tex_x1 - tex_x0, tex_y1 - tex_y0], dtype=np.float32)
        # Determine required tiles
        index_tex = []
        curtiles = OrderedDict()
        curtiles_difzoom = OrderedDict()
        for j, y in enumerate(range(y0, y1 + 1)):
            for i, x in enumerate(range(x0, x1 + 1)):
                # Find tile index if already loaded
                idx = self.curtiles.pop((x, y, zoom), None)
                if idx is not None:
                    # correct zoom, so dx,dy=0, zoomfac=1
                    index_tex.extend((0, 0, 1, idx))
                    curtiles[(x, y, zoom)] = idx
                else:
                    if finished:
                        # Tile not loaded yet, fetch in the background
                        task = TileLoader(self.tilesource, zoom, x, y, i, j)
                        task.signals.finished.connect(self.load_tile)
                        self.threadpool.start(task)

                    # In the mean time, check if more zoomed-out tiles are loaded that can be used
                    for z in range(zoom - 1, max(2, zoom - 5), -1):
                        zx, zy, dx, dy = zoomout_tilenum(x, y, z - zoom)
                        idx = self.curtiles.pop((zx, zy, z), None)
                        if idx is not None:
                            curtiles_difzoom[(zx, zy, z)] = idx
                            zoomfac = int(2 ** (zoom - z))
                            dxi = int(round(dx * zoomfac))
                            dyi = int(round(dy * zoomfac))
                            # offset zoom, so possible offset dxi, dyi
                            index_tex.extend((dxi, dyi, zoomfac, idx))
                            break
                    else:
                        # No useful tile found
                        index_tex.extend((0, 0, 0, -1))
        # Update curtiles ordered dict
        curtiles.update(curtiles_difzoom)
        curtiles.update(self.curtiles)
        self.curtiles = curtiles
        data = np.array(index_tex, dtype=np.int32)
        self.glsurface.makeCurrent()
        self.indextexture.bind(2)
        glh.gl.glTexSubImage2D_alt(glh.Texture.Target2D, 0,
                                    0, 0, nx, ny, glh.Texture.RGBA_Integer, glh.Texture.Int32, data.tobytes())

    def load_tile(self, tile):
        ''' Send loaded image data to GPU texture array.

            This function is called on callback from the
            asynchronous image load function.
        '''
        layer = len(self.curtiles)
        if layer >= bs.settings.tile_array_size:
            # we're exceeding the size of the GL texture array. Replace the least-recent tile
            _, layer = self.curtiles.popitem()
        self.curtiles[(tile.tilex, tile.tiley, tile.zoom)] = layer
        idxdata = np.array([0, 0, 1, layer], dtype=np.int32)
        self.glsurface.makeCurrent()
        self.indextexture.bind(2)
        glh.gl.glTexSubImage2D_alt(glh.Texture.Target2D, 0,
                               tile.idxx, tile.idxy, 1, 1, glh.Texture.RGBA_Integer, glh.Texture.Int32, idxdata.tobytes())
        super().bind(4)
        self.setLayerData(layer, tile.image)
        self.indextexture.release()
        self.release()


def latlon2tilenum(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def tilenum2latlon(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def tilewidth(zoom):
    n = 2 ** zoom
    dlon_deg = 1.0 / n * 360.0
    return dlon_deg


def tilezoom(dlon):
    n = 1.0 / dlon * 360.0
    zoom = math.log2(n)
    return round(zoom)


def zoomout_tilenum(xtile_in, ytile_in, delta_zoom):
    zoomfac = 2 ** delta_zoom
    xtilef = xtile_in * zoomfac
    ytilef = ytile_in * zoomfac
    xtile = int(xtilef)
    dx = xtilef - xtile
    ytile = int(ytilef)
    dy = ytilef - ytile
    return xtile, ytile, dx, dy
