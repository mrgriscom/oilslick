import maptile as mt
import math
import os
import shutil
import hashlib
import settings
from Polygon import *

EARTH_MEAN_RAD = 6371009.
EARTH_EQ_RAD = 6378137.

TILE_SIZE = 256 # px
CHUNK_DEPTH = 3 # z-levels
CHUNK_FRINGE = 4 # px

# gdalbuildvrt test.vrt -hidenodata -vrtnodata 0 ~x/gis/ferranti/hgts/*.hgt
# cd ~x/gis/ferranti/anti && python ~/dev/oilslick/clean/antify.py
# gdalbuildvrt anti.vrt -hidenodata -vrtnodata 0 ~x/gis/ferranti/anti/*.hgt

def children(tile):
    for ch in mt.quad_children(tile.x, tile.y):
        yield mt.Tile(z=tile.z + 1, x=ch[0], y=ch[1])

def render_all(resolution, bound):
    fz = math.log(2*math.pi*EARTH_MEAN_RAD / TILE_SIZE / resolution, 2)
    zmax = math.ceil(fz)
    brackets = list(mt.calc_scale_brackets(fz % 1.))
    def max_z(z, y, exact=False):
        if not exact:
            return zmax - mt.zoom_adjust(brackets, z, y)
        else:
            lat = mt.mercator_to_ll((0, mt.ref_mercy(z, y)))[0]
            return fz + math.log(math.cos(math.radians(lat)), 2)

    root = mt.Tile(z=0, x=0, y=0)
    render_tile(bound, root, max_z)
    postprocess_tile(bound, root, max_z)

def pct_complete(qt):
    pct = 1.
    for c in reversed(qt):
        pct = (pct + int(c)) / 4.
    return pct

def subset_bound(bound, tile):
    if bound is None or bound is True:
        return bound
    tilebound = tile.extent()
    if not bound.overlaps(tilebound):
        return None
    elif bound.covers(tilebound):
        return True
    else:
        return bound & tilebound

def render_tile(bound, tile, max_z, z_extracted=None):
    bound = subset_bound(bound, tile)
    if bound is None:
        return

    if RESUME and os.path.exists(tmptile(tile)):
        print 'already rendered %d %d %d [%s]' % (tile.z, tile.x, tile.y, tile.qt)
        if z_extracted is None or tile.z < z_extracted:
            for child in children(tile):
                postprocess_tile(bound, child, max_z)
        return

    if z_extracted is None:
        zmax = max_z(tile.z, tile.y)
        if zmax - tile.z <= CHUNK_DEPTH:
            z_extracted = extract(tile, zmax)

    if z_extracted is None or tile.z < z_extracted:
        for child in children(tile):
            render_tile(bound, child, max_z, z_extracted)
        render_parent(tile)
        for child in children(tile):
            postprocess_tile(bound, child, max_z)

    print 'rendered %d %d %d [%s] (%.4f%%)' % (tile.z, tile.x, tile.y, tile.qt, 100.*pct_complete(tile.qt))

def dstpath(tile, ext):
    #return os.path.join(os.path.expanduser(settings.TILE_ROOT), 'oilslick_%s.%s' % (frag, ext))
    root = settings.TILE_ROOT
    assert not root.endswith('/')
    if ext == '.png':
        root += '-ref'
    return [os.path.expanduser(root), 'z%d' % tile.z, str(tile.x), '%d%s' % (tile.y, ext)]

def postprocess_tile(bound, tile, max_z):
    bound = subset_bound(bound, tile)
    if bound is None:
        return

    if RESUME and all(os.path.exists(os.path.join(*dstpath(tile, ext))) for ext in ('.png', '.jpg')):
        print 'already post-processed %d %d %d [%s]' % (tile.z, tile.x, tile.y, tile.qt)
        return

    path = tmptile(tile)
    if tile.z <= max_z(tile.z, tile.y):
        os.popen('convert %s %s' % (path, '/tmp/tile.png'))

        overzoom = tile.z - max_z(tile.z, tile.y, True)
        assert overzoom < 1. + 1e-6
        OVZ0, OVZ1 = .3, 1.
        Q0, Q1 = 92, 70
        if overzoom < OVZ0:
            quality = Q0
        else:
            quality = (overzoom - OVZ0) / (OVZ1 - OVZ0) * (Q1 - Q0) + Q0
        os.popen('convert -quality %d %s %s' % (int(round(quality)), path, '/tmp/tile.jpg'))

        DEDUP = False
        if DEDUP:
            # really slow!
            def clone_tile(layer):
                return mt.Tile(layer=layer, z=tile.z, x=tile.x, y=tile.y)
            lossless = clone_tile('oilslick-ref')
            lossy = clone_tile('oilslick')
            lossless.savefile('/tmp/tile.png', 'png')
            lossy.savefile('/tmp/tile.jpg', 'jpg')
            SESS.add(lossless)
            SESS.add(lossy)
            commit()
        else:
            #n = int(math.ceil(tile.z * math.log(2, 10)))
            #frag = '%02d_%0*d_%0*d' % (tile.z, n, tile.x, n, tile.y)
            def move(src):
                dst = dstpath(tile, os.path.splitext(src)[1])
                for i in xrange(1, len(dst) - 1):
                    interim = os.path.join(*dst[:i+1])
                    if not os.path.exists(interim):
                        os.mkdir(interim)
                shutil.move(src, os.path.join(*dst))
            move('/tmp/tile.png')
            move('/tmp/tile.jpg')

    os.remove(path)

def commit():
    global COUNT
    COMMIT_INTERVAL = 100
    COUNT += 1
    if COUNT % COMMIT_INTERVAL == 0:
        SESS.commit()

def tmptile(tile):
    return '/tmp/rawtile_%d_%d_%d.tiff' % (tile.z, tile.x, tile.y)

def extract(tile, zmax):
    z = max(zmax, tile.z)
    print 'extracting %d %d %d to z%d' % (tile.z, tile.x, tile.y, z)

    size = int(2**(z - tile.z))
    inner_width = TILE_SIZE * size
    width = inner_width + 2 * CHUNK_FRINGE

    res = 2*math.pi*EARTH_EQ_RAD / TILE_SIZE / 2**z
    p0 = [EARTH_EQ_RAD * k for k in mt.xy_to_mercator(mt.tilef_to_xy([tile.x, tile.y + 1], tile.z))]

    # fucking-A
    crosses_idl = False
    if tile.x == 0:
        crosses_idl = True
        p0[0] += math.pi * EARTH_EQ_RAD
    elif tile.x == 2**tile.z - 1:
        crosses_idl = True
        p0[0] -= math.pi * EARTH_EQ_RAD

    CHUNK_DEM = '/tmp/dem_chunk.tiff'
    CHUNK_COLOR = '/tmp/color_chunk.tiff'

    os.popen('gdalwarp -t_srs EPSG:3857 -tr %(res)s %(res)s -te %(xmin)s %(ymin)s %(xmax)s %(ymax)s -r cubic -multi %(dem)s %(chunk_dem)s' % {
            'res': res,
            'xmin': p0[0] - res * CHUNK_FRINGE,
            'ymin': p0[1] - res * CHUNK_FRINGE,
            'xmax': p0[0] + res * (width - CHUNK_FRINGE),
            'ymax': p0[1] + res * (width - CHUNK_FRINGE),
            'dem': '/home/drew/dev/oilslick/%s.vrt' % ('test' if not crosses_idl else 'anti'),
            'chunk_dem': CHUNK_DEM,
        })
    os.popen('gdaldem color-relief %s ~/tmp/demlegend %s' % (CHUNK_DEM, CHUNK_COLOR))
    os.popen('convert -crop %(width)sx%(width)s+%(fringe)s+%(fringe)s %(chunk)s tiff:- | convert +repage -crop %(tilesize)sx%(tilesize)s tiff:- /tmp/rawtile%%d.tiff' % {
            'width': inner_width,
            'fringe': CHUNK_FRINGE,
            'chunk': CHUNK_COLOR,
            'tilesize': TILE_SIZE,
        })
    xbase = tile.x * 2**(z - tile.z)
    ybase = tile.y * 2**(z - tile.z)
    for i in xrange(size * size):
        r = i // size
        c = i % size
        shutil.move('/tmp/rawtile%d.tiff' % i, tmptile(mt.Tile(z=z, x=xbase + c, y=ybase + r)))
    os.remove(CHUNK_DEM)
    os.remove(CHUNK_COLOR)

    return z

def render_parent(tile):
    def _path(child):
        p = tmptile(child)
        if not os.path.exists(p):
            p = os.path.join(*dstpath(child, '.png'))
        return p
    childpaths = [_path(child) for child in children(tile)]
    os.popen('montage -mode Concatenate -tile 2x2 %s tif:- | convert -filter Box -geometry 50%% tif:- %s' % (' '.join(childpaths), tmptile(tile)))

RESUME = False

if __name__ == "__main__":

    #boundary = mt.quadrant(-26.2, -25.9, 149.3, 149.9)[0]
    #boundary = mt.quadrant(-86, 86, 179.9, 180)[0]
    boundary = mt.quadrant(-86, 86, -180, -179.9)[0]

    bound = Polygon([mt.mercator_to_xy(mt.ll_to_mercator(p)) for p in boundary])

    COUNT = 0
    SESS = mt.dbsess()
    render_all(2*math.pi*EARTH_MEAN_RAD / 360. / 1200., bound)
    SESS.commit()
