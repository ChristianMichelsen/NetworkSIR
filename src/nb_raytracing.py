from numba import jit
from numba.pycc import CC

cc = CC("nbspatial")


@cc.export("ray_tracing", "b1(f8, f8, f8[:,:])")
@jit(nopython=True)
def ray_tracing(x, y, poly):
    # https://stackoverflow.com/a/48760556
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


if __name__ == "__main__":
    cc.compile()
