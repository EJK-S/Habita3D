"""Algoritmos de rasterización de líneas: DDA, Bresenham y Wu (antialiasing)."""
import math


def _redondear(v):
    """Redondeo clásico (0.5 hacia arriba), no el 'bancario' de Python."""
    return int(math.floor(v + 0.5))


# --------------------------------------------------------------------------- DDA
def puntos_dda(x1, y1, x2, y2):
    """DDA: pasos = max(|dx|, |dy|); se suman incrementos reales y se redondea en cada paso."""
    dx, dy = x2 - x1, y2 - y1
    pasos = _redondear(max(abs(dx), abs(dy)))
    if pasos == 0:
        return [(_redondear(x1), _redondear(y1))]
    inc_x, inc_y = dx / pasos, dy / pasos
    x, y = float(x1), float(y1)
    puntos = []
    for _ in range(pasos + 1):
        puntos.append((_redondear(x), _redondear(y)))
        x += inc_x
        y += inc_y
    return puntos


# ----------------------------------------------------------------------- Bresenham
def puntos_bresenham(x1, y1, x2, y2):
    """Bresenham con parámetro de decisión y solo aritmética entera, en los 8 octantes."""
    x1, y1, x2, y2 = (_redondear(v) for v in (x1, y1, x2, y2))
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1
    puntos = []
    x, y = x1, y1
    if dy <= dx:                       # pendiente suave: avanza siempre en x
        p = 2 * dy - dx
        for _ in range(dx + 1):
            puntos.append((x, y))
            if p < 0:
                p += 2 * dy
            else:
                y += sy
                p += 2 * (dy - dx)
            x += sx
    else:                              # pendiente fuerte: avanza siempre en y
        p = 2 * dx - dy
        for _ in range(dy + 1):
            puntos.append((x, y))
            if p < 0:
                p += 2 * dx
            else:
                x += sx
                p += 2 * (dx - dy)
            y += sy
    return puntos


def linea_dda(fb, x1, y1, x2, y2, color=(0, 0, 0), grosor=1):
    for x, y in puntos_dda(x1, y1, x2, y2):
        fb.estampar(x, y, color, grosor)


def linea_bresenham(fb, x1, y1, x2, y2, color=(0, 0, 0), grosor=1):
    for x, y in puntos_bresenham(x1, y1, x2, y2):
        fb.estampar(x, y, color, grosor)


# ----------------------------------------------------------------------------- Wu
def linea_wu(fb, x1, y1, x2, y2, color=(0, 0, 0)):
    """Línea con antialiasing de Xiaolin Wu: en cada columna reparte la intensidad
    entre los dos píxeles más cercanos a la recta ideal."""
    ipart = math.floor
    fpart = lambda v: v - math.floor(v)
    rfpart = lambda v: 1 - fpart(v)

    if x1 == x2 and y1 == y2:
        fb.mezclar_pixel(_redondear(x1), _redondear(y1), color, 1.0)
        return
    empinada = abs(y2 - y1) > abs(x2 - x1)
    if empinada:
        x1, y1, x2, y2 = y1, x1, y2, x2
    if x1 > x2:
        x1, x2, y1, y2 = x2, x1, y2, y1

    def punto(x, y, a):
        if empinada:
            fb.mezclar_pixel(y, x, color, a)
        else:
            fb.mezclar_pixel(x, y, color, a)

    dx, dy = x2 - x1, y2 - y1
    grad = dy / dx if dx != 0 else 1.0

    # extremo inicial
    xend = _redondear(x1)
    yend = y1 + grad * (xend - x1)
    xgap = rfpart(x1 + 0.5)
    xp1, yp1 = xend, ipart(yend)
    punto(xp1, yp1, rfpart(yend) * xgap)
    punto(xp1, yp1 + 1, fpart(yend) * xgap)
    interseccion = yend + grad

    # extremo final
    xend = _redondear(x2)
    yend = y2 + grad * (xend - x2)
    xgap = fpart(x2 + 0.5)
    xp2, yp2 = xend, ipart(yend)
    punto(xp2, yp2, rfpart(yend) * xgap)
    punto(xp2, yp2 + 1, fpart(yend) * xgap)

    for x in range(xp1 + 1, xp2):
        punto(x, ipart(interseccion), rfpart(interseccion))
        punto(x, ipart(interseccion) + 1, fpart(interseccion))
        interseccion += grad


def linea(fb, x1, y1, x2, y2, color=(0, 0, 0), grosor=1, algoritmo="bresenham"):
    """Dibuja una línea con el algoritmo elegido ('dda', 'bresenham' o 'wu')."""
    if algoritmo == "dda":
        linea_dda(fb, x1, y1, x2, y2, color, grosor)
    elif algoritmo == "wu":
        linea_wu(fb, x1, y1, x2, y2, color)
    else:
        linea_bresenham(fb, x1, y1, x2, y2, color, grosor)
