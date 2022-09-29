''' Tile texture manager for BlueSky Qt/OpenGL gui. '''
import traceback
import math
import weakref
from collections import OrderedDict
from urllib.request import urlopen
from urllib.error import URLError
import numpy as np
import gzip

try:
    from PyQt5.Qt import Qt
    from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot, QT_VERSION_STR
    from PyQt5.QtGui import QImage, qRgba
except ImportError:
    from PyQt6.QtCore import Qt
    from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot, QT_VERSION_STR
    from PyQt6.QtGui import QImage, qRgba

import bluesky as bs
from bluesky.core import Signal
from bluesky.ui.qtgl import glhelpers as glh


bs.settings.set_variable_defaults(
    tile_array_size=100,
    max_download_workers=2,
    max_tile_zoom=18,
    tile_sources={
        'opentopomap': {
            'source': ['https://a.tile.opentopomap.org/{zoom}/{x}/{y}.png',
                       'https://b.tile.opentopomap.org/{zoom}/{x}/{y}.png',
                       'https://c.tile.opentopomap.org/{zoom}/{x}/{y}.png'],
            'max_download_workers': 2,
            'max_tile_zoom': 17,
            'license': 'map data: © OpenStreetMap contributors, SRTM | map style: © OpenTopoMap.org (CC-BY-SA)'},
        'cartodb': {
            'source': ['https://cartodb-basemaps-b.global.ssl.fastly.net/light_nolabels/{zoom}/{x}/{y}.png'],
            'max_tile_zoom': 20,
            'license': 'CartoDB Grey and white, no labels'
        },
        'nasa': {
            'source': ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}.jpg'],
            'max_tile_zoom': 13,
            'license': 'Satellite images from NASA via ESRI'
        }
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
        self.ext = source[source.rfind('.'):]
        self.ext = '.' + bs.settings.tile_sources[source]['source'][0].split('.')[-1]

        self.image = None
        # For the image data, check cache path first
        fpath = bs.resource(bs.settings.cache_path) / source / str(zoom) / str(tilex)
        fname = fpath / f'{tiley}{self.ext}'
        if fname.exists():
            self.image = QImage(fname.as_posix()).convertToFormat(QImage.Format.Format_ARGB32)
        else:
            # Make sure cache directory exists
            fpath.mkdir(parents=True, exist_ok=True)
            for url in bs.settings.tile_sources[source]['source']:
                try:
                    url_request = urlopen(url.format(
                        zoom=zoom, x=tilex, y=tiley))
                    
                    if url_request.status == 204:
                        # if no content load a blank tile
                        self.image = QImage(256, 256, QImage.Format.Format_ARGB32)
                        self.image.fill(qRgba(255,255,255,0))

                    else:
                        data = url_request.read()

                        if url_request.headers['Content-Encoding'] == 'gzip':
                            # There is a chance that data may come as a gzip so decompress
                            data = gzip.decompress(data)
                                        
                        self.image = QImage.fromData(
                            data).convertToFormat(QImage.Format.Format_ARGB32)

                        with open(fname, 'wb') as fout:
                            fout.write(data)
                    break
                except URLError as e:
                    print(f'Error loading {url.format(zoom=zoom, x=tilex, y=tiley)}:')
                    print(traceback.format_exc())


class TileLoader(QRunnable):
    ''' Thread worker to load tiles in the background. '''
    class Signals(QObject):
        ''' Container class for worker signals. '''
        finished = pyqtSignal(Tile)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = TileLoader.Signals()

    def run(self):
        ''' Function to execute in the worker thread. '''
        tile = Tile(*self.args, **self.kwargs)
        if tile.image is not None:
            self.signals.finished.emit(tile)


class TiledTextureMeta(type(glh.Texture)):
    ''' TileTexture meta class that stores weak references to created textures
        so that they can be used in multiple objects, but are deleted when the
        last reference to them is removed. '''
    tiletextures = weakref.WeakValueDictionary()
    def __call__(cls, *args, **kwargs):
        name = kwargs.get('tilesource', 'opentopomap')
        tex = cls.tiletextures.get(name)
        if tex is None:
            tex = super().__call__(*args, **kwargs)
            cls.tiletextures[name] = tex
        return tex


class TiledTexture(glh.Texture, metaclass=TiledTextureMeta):
    ''' Tiled texture implementation for the BlueSky GL gui. '''
    class SlotHolder(QObject):
        ''' Wrapper class for Qt slot, which can only be owned by a
            QObject-derived parent. We need slots to allow signal receiver
            to be executed in the receiving (main) thread. '''
        def __init__(self, callback):
            super().__init__()
            self.cb = callback

        @pyqtSlot(Tile)
        def slot(self, *args, **kwargs):
            self.cb(*args, **kwargs)

    def __init__(self, glsurface, tilesource='opentopomap'):
        super().__init__(target=glh.Texture.Target.Target2DArray)
        self.threadpool = QThreadPool()
        tileinfo = bs.settings.tile_sources.get(tilesource)
        if not tileinfo:
            raise KeyError(f'Tile source {tilesource} not found!')
        max_dl = tileinfo.get('max_download_workers', bs.settings.max_download_workers)
        self.maxzoom = tileinfo.get('max_tile_zoom', bs.settings.max_tile_zoom)
        self.threadpool.setMaxThreadCount(min(bs.settings.max_download_workers, max_dl))
        self.tileslot = TiledTexture.SlotHolder(self.load_tile)
        self.tilesource = tilesource
        self.tilesize = (256, 256)
        self.curtileext = (0, 0, 0, 0)
        self.curtilezoom = 1
        self.curtiles = OrderedDict()
        self.fullscreen = False
        self.offsetscale = np.array([0, 0, 1], dtype=np.float32)
        self.bbox = list()
        self.glsurface = glsurface
        self.indextexture = glh.Texture(target=glh.Texture.Target.Target2D)
        self.indexsampler_loc = 0
        self.arraysampler_loc = 0
        bs.net.actnodedata_changed.connect(self.actdata_changed)
        Signal('panzoom').connect(self.on_panzoom_changed)

    def add_bounding_box(self, lat0, lon0, lat1, lon1):
        ''' Add the bounding box of a textured shape.

            These bounding boxes are used to determine if tiles need to be
            downloaded. '''
        if abs(lat1 - lat0) >= 178 and abs(lon1 - lon0) >= 358:
            self.fullscreen = True
        else:
            self.bbox.append((lat0, lon0, lat1, lon1))

    def create(self):
        ''' Create this texture in GPU memory. '''

        if self.isCreated():
            return
        super().create()

        # Fetch a temporary tile image to get dimensions
        tmptile = Tile(self.tilesource, 1, 1, 1, 0, 0)
        img = tmptile.image
        self.setFormat(glh.Texture.TextureFormat.RGBA8_UNorm)
        self.tilesize = (img.width(), img.height())
        self.setSize(img.width(), img.height())
        self.setLayers(bs.settings.tile_array_size)
        super().bind()
        self.allocateStorage()
        self.setWrapMode(glh.Texture.CoordinateDirection.DirectionS,
                         glh.Texture.WrapMode.ClampToBorder)
        self.setWrapMode(glh.Texture.CoordinateDirection.DirectionT,
                         glh.Texture.WrapMode.ClampToBorder)
        self.setMinMagFilters(glh.Texture.Filter.Linear, glh.Texture.Filter.Linear)

        # Initialize index texture
        # RG = texcoord offset, B = zoom factor, A = array index
        itexw = int(np.sqrt(bs.settings.tile_array_size) * 4 / 3 + 10)
        itexh = int(np.sqrt(bs.settings.tile_array_size) * 3 / 4 + 10)
        self.indextexture.create()
        self.indextexture.setFormat(glh.Texture.TextureFormat.RGBA32I)
        self.indextexture.setSize(itexw, itexh)
        self.indextexture.bind(1)
        # self.indextexture.allocateStorage(glh.Texture.RGBA_Integer, glh.Texture.Int32)

        idxdata = np.array(itexw * itexh *
                           [(0, 0, 0, -1)], dtype=np.int32)
        if QT_VERSION_STR[0] == '5':
            target_value = glh.Texture.Target.Target2D
            text_rgba_value = glh.Texture.TextureFormat.RGBA32I
            pixel_rgba_value = glh.Texture.PixelFormat.RGBA_Integer
            pixel_type_value = glh.Texture.PixelType.Int32

        if QT_VERSION_STR[0] == '6':
            target_value = glh.Texture.Target.Target2D.value
            text_rgba_value = glh.Texture.TextureFormat.RGBA32I.value
            pixel_rgba_value = glh.Texture.PixelFormat.RGBA_Integer.value
            pixel_type_value = glh.Texture.PixelType.Int32.value

        glh.gl.glTexImage2D_alt(target_value, 0, text_rgba_value,
                                itexw, itexh, 0, pixel_rgba_value,
                                pixel_type_value, idxdata.tobytes())

        self.indextexture.setWrapMode(glh.Texture.CoordinateDirection.DirectionS,
                                      glh.Texture.WrapMode.ClampToBorder)
        self.indextexture.setWrapMode(glh.Texture.CoordinateDirection.DirectionT,
                                      glh.Texture.WrapMode.ClampToBorder)
        self.indextexture.setMinMagFilters(glh.Texture.Filter.Nearest, glh.Texture.Filter.Nearest)

        shader = glh.ShaderSet.get_shader('tiled')
        self.indexsampler_loc = shader.uniformLocation('tile_index')
        self.arraysampler_loc = shader.uniformLocation('tile_texture')

    def bind(self, unit=0):
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
        ''' Update textures whenever pan/zoom changes. 
            
            Arguments:
            - finished: False when still in the process of panning/zooming.
        '''
        # Check if textures need to be updated
        viewport = self.glsurface.viewportlatlon()
        surfwidth_px = self.glsurface.width()

        # First determine floating-point, hypothetical values
        # to calculate the required tile zoom level
        # floating-point number of tiles that fit in the width of the view
        ntiles_hor = surfwidth_px / self.tilesize[0]
        # Estimated width in longitude of one tile
        est_tilewidth = abs(viewport[3] - viewport[1]) / ntiles_hor

        self.curtilezoom = tilezoom(est_tilewidth, self.maxzoom)
        # With the tile zoom level get the required number of tiles
        # Top-left and bottom-right tiles:
        x0, y0 = latlon2tilenum(*viewport[:2], self.curtilezoom)
        x1, y1 = latlon2tilenum(*viewport[2:], self.curtilezoom)
        nx = abs(x1 - x0) + 1
        ny = abs(y1 - y0) + 1
        self.curtileext = (x0, y0, x1, y1)

        # Calculate the offset of the top-left tile w.r.t. the screen top-left corner
        tile0_topleft = np.array(tilenum2latlon(x0, y0, self.curtilezoom))
        tile0_bottomright = np.array(tilenum2latlon(x0 + 1, y0 + 1, self.curtilezoom))
        tilesize_latlon0 = np.abs(tile0_bottomright - tile0_topleft)
        offset_latlon0 = viewport[:2] - tile0_topleft
        tex_y0, tex_x0 = np.abs(offset_latlon0 / tilesize_latlon0)

        # Calculate the offset of the bottom-right tile w.r.t. the screen bottom right corner
        tile1_topleft = np.array(tilenum2latlon(x1, y1, self.curtilezoom))
        tile1_bottomright = np.array(tilenum2latlon(x1 + 1, y1 + 1, self.curtilezoom))
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
                idx = self.curtiles.pop((x, y, self.curtilezoom), None)
                if idx is not None:
                    # correct zoom, so dx,dy=0, zoomfac=1
                    index_tex.extend((0, 0, 1, idx))
                    curtiles[(x, y, self.curtilezoom)] = idx
                else:
                    if finished:
                        # Tile not loaded yet, fetch in the background
                        task = TileLoader(self.tilesource, self.curtilezoom, x, y, i, j)
                        task.signals.finished.connect(self.tileslot.slot, Qt.ConnectionType.QueuedConnection)
                        self.threadpool.start(task)

                    # In the mean time, check if more zoomed-out tiles are loaded that can be used
                    for z in range(self.curtilezoom - 1, max(2, self.curtilezoom - 5), -1):
                        zx, zy, dx, dy = zoomout_tilenum(x, y, z - self.curtilezoom)
                        idx = self.curtiles.pop((zx, zy, z), None)
                        if idx is not None:
                            curtiles_difzoom[(zx, zy, z)] = idx
                            zoomfac = int(2 ** (self.curtilezoom - z))
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

        if QT_VERSION_STR[0] == '5':
            target_value = glh.Texture.Target.Target2D
            pixel_rgba_value = glh.Texture.PixelFormat.RGBA_Integer
            pixel_type_value = glh.Texture.PixelType.Int32

        if QT_VERSION_STR[0] == '6':
            target_value = glh.Texture.Target.Target2D.value
            pixel_rgba_value = glh.Texture.PixelFormat.RGBA_Integer.value
            pixel_type_value = glh.Texture.PixelType.Int32.value

        glh.gl.glTexSubImage2D_alt(target_value, 0, 0, 0, nx, ny,
                                  pixel_rgba_value, pixel_type_value, data.tobytes())

    def load_tile(self, tile):
        ''' Send loaded image data to GPU texture array.

            This function is called on callback from the
            asynchronous image load function.
        '''
        # First check whether tile is still relevant
        # If there's a lot of panning/zooming, sometimes tiles are loaded that
        # become obsolete before they are even downloaded.
        if not ((self.curtilezoom == tile.zoom) and
                (self.curtileext[0] <= tile.tilex <= self.curtileext[2]) and
                (self.curtileext[1] <= tile.tiley <= self.curtileext[3])):
            return

        # Make sure our GL context is current
        self.glsurface.makeCurrent()

        # Check if tile is already loaded. This can happen here with multiple
        # pans/zooms shortly after each other
        layer = self.curtiles.get((tile.tilex, tile.tiley, tile.zoom), None)
        if layer is None:
            # This tile is not loaded yet. Select layer to upload it to
            layer = len(self.curtiles)
            if layer >= bs.settings.tile_array_size:
                # we're exceeding the size of the GL texture array.
                # Replace the least-recent tile
                _, layer = self.curtiles.popitem()

            self.curtiles[(tile.tilex, tile.tiley, tile.zoom)] = layer
            # Upload tile to texture array
            super().bind(4)
            self.setLayerData(layer, tile.image)
            self.release()

        # Update the ordering of the tile dict: the new tile should be on top
        self.curtiles.move_to_end(
            (tile.tilex, tile.tiley, tile.zoom), last=False)

        # Update index texture
        idxdata = np.array([0, 0, 1, layer], dtype=np.int32)
        self.indextexture.bind(2)

        if QT_VERSION_STR[0] == '5':
            target_value = glh.Texture.Target.Target2D
            pixel_rgba_value = glh.Texture.PixelFormat.RGBA_Integer
            pixel_type_value = glh.Texture.PixelType.Int32

        if QT_VERSION_STR[0] == '6':
            target_value = glh.Texture.Target.Target2D.value
            pixel_rgba_value = glh.Texture.PixelFormat.RGBA_Integer.value
            pixel_type_value = glh.Texture.PixelType.Int32.value


        glh.gl.glTexSubImage2D_alt(target_value, 0, tile.idxx, tile.idxy,
                                   1, 1, pixel_rgba_value,
                                   pixel_type_value, idxdata.tobytes())

        self.indextexture.release()


def latlon2tilenum(lat_deg, lon_deg, zoom):
    ''' Generate tile x/y numbers for a given lat/lon/zoom. '''
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def tilenum2latlon(xtile, ytile, zoom):
    ''' Generate lat/lon coordinates for the top-left corner of a tile, given
        the x/y tilenumbers and zoom factor. '''
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def tilewidth(zoom):
    ''' Calculate the width of a tile in degrees longitude,
        given a zoom factor. '''
    n = 2 ** zoom
    dlon_deg = 1.0 / n * 360.0
    return dlon_deg


def tilezoom(dlon, maxzoom=18):
    ''' Calculate a zoom factor, given a (hypothetical) width of a tile in
        degrees longitude. '''
    n = 1.0 / dlon * 360.0
    zoom = math.log2(n)
    return min(maxzoom, round(zoom))


def zoomout_tilenum(xtile_in, ytile_in, delta_zoom):
    ''' Calculate corresponding tile x/y number for the overlapping tile
        that is 'delta_zoom' steps zoomed out. '''
    zoomfac = 2 ** delta_zoom
    xtilef = xtile_in * zoomfac
    ytilef = ytile_in * zoomfac
    xtile = int(xtilef)
    dx = xtilef - xtile
    ytile = int(ytilef)
    dy = ytilef - ytile
    return xtile, ytile, dx, dy
