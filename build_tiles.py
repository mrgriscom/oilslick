import maptile as mt
import math
import os
import shutil
import hashlib
import settings

EARTH_MEAN_RAD = 6371009.
EARTH_EQ_RAD = 6378137.

TILE_SIZE = 256 # px
CHUNK_DEPTH = 3 # z-levels
CHUNK_FRINGE = 4 # px

# gdalbuildvrt test.vrt -hidenodata -vrtnodata 0 ~x/gis/ferranti/hgts/*.hgt

def children(tile):
    for ch in mt.quad_children(tile.x, tile.y):
        yield mt.Tile(z=tile.z + 1, x=ch[0], y=ch[1])

def render_all(resolution):
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
    render_tile(root, max_z)
    postprocess_tile(root, max_z)

def pct_complete(qt):
    pct = 1.
    for c in reversed(qt):
        pct = (pct + int(c)) / 4.
    return pct

def render_tile(tile, max_z, z_extracted=None):
    if z_extracted is None:
        zmax = max_z(tile.z, tile.y)
        if zmax - tile.z <= CHUNK_DEPTH:
            z_extracted = extract(tile, zmax)

    if z_extracted is None or tile.z < z_extracted:
        for child in children(tile):
            render_tile(child, max_z, z_extracted)
        render_parent(tile)
        for child in children(tile):
            postprocess_tile(child, max_z)

    print 'rendered %d %d %d [%s] (%.4f%%)' % (tile.z, tile.x, tile.y, tile.qt, 100.*pct_complete(tile.qt))

def postprocess_tile(tile, max_z):
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
            def dstpath(ext):
                #return os.path.join(os.path.expanduser(settings.TILE_ROOT), 'oilslick_%s.%s' % (frag, ext))
                return [os.path.expanduser(settings.TILE_ROOT), 'z%d' % tile.z, str(tile.x), '%d%s' % (tile.y, ext)]
            def move(src):
                dst = dstpath(os.path.splitext(src)[1])
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
    # super-annoyingly, gdalwarp can't handle crossing the IDL (do people even *think* about
    # this??)... as a workaround, we drop the fringe when this will happen. the only consequence
    # is we lose cubic resampling on the very edge pixel
    fringe_left = CHUNK_FRINGE if tile.x > 0 else -.01
    fringe_right = CHUNK_FRINGE if tile.x < 2**tile.z - 1 else -.01
    width = int(round(inner_width + fringe_left + fringe_right))
    height = inner_width + 2 * CHUNK_FRINGE

    res = 2*math.pi*EARTH_EQ_RAD / TILE_SIZE / 2**z
    p0 = [EARTH_EQ_RAD * k for k in mt.xy_to_mercator(mt.tilef_to_xy([tile.x, tile.y + 1], tile.z))]

    CHUNK_DEM = '/tmp/dem_chunk.tiff'
    CHUNK_COLOR = '/tmp/color_chunk.tiff'

    os.popen('gdalwarp -t_srs EPSG:3857 -tr %(res)s %(res)s -te %(xmin)s %(ymin)s %(xmax)s %(ymax)s -r cubic -multi %(dem)s %(chunk_dem)s' % {
            'res': res,
            'xmin': p0[0] - res * fringe_left,
            'ymin': p0[1] - res * CHUNK_FRINGE,
            'xmax': p0[0] + res * (width - fringe_left),
            'ymax': p0[1] + res * (height - CHUNK_FRINGE),
            'dem': '/tmp/e/test.vrt',
            'chunk_dem': CHUNK_DEM,
        })
    os.popen('gdaldem color-relief %s ~/tmp/demlegend %s' % (CHUNK_DEM, CHUNK_COLOR))
    os.popen('convert -crop %(width)sx%(width)s+%(xfringe)s+%(yfringe)s %(chunk)s tiff:- | convert +repage -crop %(tilesize)sx%(tilesize)s tiff:- /tmp/rawtile%%d.tiff' % {
            'width': inner_width,
            'xfringe': int(round(fringe_left)),
            'yfringe': CHUNK_FRINGE,
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
    childpaths = [tmptile(child) for child in children(tile)]
    os.popen('montage -mode Concatenate -tile 2x2 %s tif:- | convert -filter Box -geometry 50%% tif:- %s' % (' '.join(childpaths), tmptile(tile)))

if __name__ == "__main__":
    COUNT = 0
    SESS = mt.dbsess()
    render_all(2*math.pi*EARTH_MEAN_RAD / 360. / 1200.)
    SESS.commit()
