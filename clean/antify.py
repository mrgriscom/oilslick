import os


def parselon(x):
    return {'E': 1, 'W': -1}[x[0]] * int(x[1:])

def fmtlon(x):
    return ('W' if x < 0 else 'E') + '%03d' % abs(x)

def antilon(lon):
    lon += 180
    return (lon + 180) % 360 - 180

if __name__ == "__main__":

    dems = os.listdir('../hgts')

    for d in dems:
        lon = parselon(d[3:7])
        anti = antilon(lon)
        e = d[:3] + fmtlon(anti) + d[7:]
        
        os.popen('ln -s /mnt/ext/gis/ferranti/hgts/%s %s' % (d, e))

