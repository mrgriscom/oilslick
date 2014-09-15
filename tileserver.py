from tornado.ioloop import IOLoop
import tornado.web as web
import tornado.gen as gen
from tornado.httpclient import HTTPError, AsyncHTTPClient
from tornado.template import Template
import settings
from optparse import OptionParser
import maptile as mt

class TileRequestHandler(web.RequestHandler):

    PATTERN = '([A-Za-z0-9_-]+)/([0-9]+)/([0-9]+),([0-9]+)'

    def get(self, layer, z, x, y):
        self._get(mt.Tile(layer=layer, z=int(z), x=int(x), y=int(y)))

class TileHandler(TileRequestHandler):

    def initialize(self, dbsess):
        self.sess = dbsess

    def _get(self, tile):
        self.set_header('Access-Control-Allow-Origin', '*')

        t = sess.query(mt.Tile).get(tile.pk())
        if not t:
            self.set_status(404)
            return

        self.set_header('Cache-Control', 'public, max-age=86400')
        self.redirect('http://localhost/tiles/%s/%s/%s.jpg' % (t.layer, t.uuid[:3], t.uuid))


sess = mt.dbsess()

if __name__ == "__main__":
    parser = OptionParser()
    (options, args) = parser.parse_args()
    try:
        port = int(args[0])
    except IndexError:
        port = 8000
    ssl = None #{'certfile': web_path('ssl.crt')} if options.ssl else None

    application = web.Application([
        (r'/' + TileRequestHandler.PATTERN, TileHandler, {'dbsess': sess}),
    ], debug=True)
    application.listen(port, ssl_options=ssl)

    try:
        IOLoop.instance().start()
    except KeyboardInterrupt:
        pass
    except Exception, e:
        print e
        raise

    logging.info('shutting down...')
