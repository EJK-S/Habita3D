"""Circunferencias (punto medio) y polígonos (contorno y relleno por líneas de barrido)."""
import math
import numpy as np
from .lineas import linea


def puntos_circulo(cx, cy, r):
    """Circunferencia por el algoritmo del punto medio con simetría de 8 octantes."""
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
    puntos = set()
    x, y, p = 0, r, 1 - r
    while x <= y:
        for dx, dy in ((x, y), (y, x)):
            for sx in (1, -1):
                for sy in (1, -1):
                    puntos.add((cx + sx * dx, cy + sy * dy))
        x += 1
        if p < 0:
            p += 2 * x + 1
        else:
            y -= 1
            p += 2 * (x - y) + 1
    return sorted(puntos)


def circulo(fb, cx, cy, r, color=(0, 0, 0), grosor=1, relleno=None):
    """Circunferencia con grosor y, opcionalmente, relleno de otro color."""
    if relleno is not None:
        cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(r))
        ys = np.arange(max(cy_i - r_i, 0), min(cy_i + r_i, fb.alto - 1) + 1)
        xs = np.arange(max(cx_i - r_i, 0), min(cx_i + r_i, fb.ancho - 1) + 1)
        if len(xs) and len(ys):
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            dentro = (xx - cx_i) ** 2 + (yy - cy_i) ** 2 <= r_i * r_i + r_i
            fb.datos[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1][dentro] = relleno
    for x, y in puntos_circulo(cx, cy, r):
        fb.estampar(x, y, color, grosor)


def dibujar_poligono(fb, vertices, color=(0, 0, 0), grosor=1, cerrar=True, algoritmo="bresenham"):
    """Contorno del polígono (o polilínea si cerrar=False)."""
    v = [tuple(p) for p in vertices]
    n = len(v)
    for i in range(n if cerrar else n - 1):
        x1, y1 = v[i]
        x2, y2 = v[(i + 1) % n]
        linea(fb, x1, y1, x2, y2, color, grosor, algoritmo)


def rellenar_poligono(fb, vertices, relleno, borde=None, grosor=1):
    """Relleno por líneas de barrido con regla par-impar; sirve para polígonos cóncavos.

    Para cada fila y se calculan las intersecciones con los lados (intervalo semiabierto
    [ymin, ymax) para no contar dos veces un vértice compartido), se ordenan en x y se
    pintan los tramos entre pares de intersecciones.
    """
    v = [tuple(map(float, p)) for p in vertices]
    n = len(v)
    if n < 3:
        return
    lados = []
    for i in range(n):
        (xa, ya), (xb, yb) = v[i], v[(i + 1) % n]
        if ya == yb:
            continue                     # los lados horizontales no aportan intersecciones
        if ya > yb:
            xa, ya, xb, yb = xb, yb, xa, ya
        lados.append((ya, yb, xa, (xb - xa) / (yb - ya)))
    ymin = max(int(math.ceil(min(p[1] for p in v))), 0)
    ymax = min(int(math.floor(max(p[1] for p in v))), fb.alto - 1)
    for y in range(ymin, ymax + 1):
        xs = sorted(x0 + m * (y - y0) for (y0, y1, x0, m) in lados if y0 <= y < y1)
        for i in range(0, len(xs) - 1, 2):
            xa = max(int(math.ceil(xs[i])), 0)
            xb = min(int(math.floor(xs[i + 1])), fb.ancho - 1)
            if xa <= xb:
                fb.datos[y, xa:xb + 1] = relleno
    if borde is not None:
        dibujar_poligono(fb, v, borde, grosor)
