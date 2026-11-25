"""Framebuffer: arreglo NumPy (alto, ancho, 3) de píxeles RGB.

Origen (0, 0) en la esquina superior izquierda; x crece hacia la derecha y y hacia abajo.
Todas las primitivas del proyecto escriben aquí; Pillow solo se usa para guardar la imagen.
"""
import numpy as np
from PIL import Image, ImageDraw


class Framebuffer:
    def __init__(self, ancho, alto, fondo=(255, 255, 255)):
        self.ancho = int(ancho)
        self.alto = int(alto)
        self.fondo = tuple(fondo)
        self.datos = np.empty((self.alto, self.ancho, 3), dtype=np.uint8)
        self.limpiar()
        self._pinceles = {}

    def limpiar(self, color=None):
        self.datos[:, :] = self.fondo if color is None else color

    def dentro(self, x, y):
        return 0 <= x < self.ancho and 0 <= y < self.alto

    def poner_pixel(self, x, y, color):
        if 0 <= x < self.ancho and 0 <= y < self.alto:
            self.datos[y, x] = color

    def leer_pixel(self, x, y):
        return tuple(int(c) for c in self.datos[y, x])

    def mezclar_pixel(self, x, y, color, alfa):
        """Mezcla el color con el que ya hay: resultado = alfa·color + (1−alfa)·fondo."""
        if 0 <= x < self.ancho and 0 <= y < self.alto and alfa > 0:
            alfa = min(1.0, alfa)
            viejo = self.datos[y, x].astype(float)
            self.datos[y, x] = np.rint(alfa * np.asarray(color, float) + (1 - alfa) * viejo)

    def _pincel(self, grosor):
        """Máscara circular de diámetro 'grosor' (se guarda para no recalcularla)."""
        if grosor not in self._pinceles:
            r = (grosor - 1) / 2.0
            n = int(np.ceil(r))
            yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
            self._pinceles[grosor] = (n, (xx * xx + yy * yy) <= r * r + 0.25)
        return self._pinceles[grosor]

    def estampar(self, x, y, color, grosor=1):
        """Dibuja un pixel o un 'pincel' circular de diámetro 'grosor' centrado en (x, y)."""
        if grosor <= 1:
            self.poner_pixel(x, y, color)
            return
        n, mascara = self._pincel(grosor)
        x0, y0 = x - n, y - n
        xa, ya = max(x0, 0), max(y0, 0)
        xb, yb = min(x0 + mascara.shape[1], self.ancho), min(y0 + mascara.shape[0], self.alto)
        if xa >= xb or ya >= yb:
            return
        sub = mascara[ya - y0:yb - y0, xa - x0:xb - x0]
        self.datos[ya:yb, xa:xb][sub] = color

    def a_imagen(self, zoom=1, rejilla=False):
        """Convierte a imagen Pillow. Con zoom > 1 cada píxel se ve como un cuadro (para estudiar el aliasing)."""
        img = Image.fromarray(self.datos, "RGB")
        if zoom > 1:
            img = img.resize((self.ancho * zoom, self.alto * zoom), Image.NEAREST)
            if rejilla and zoom >= 6:
                d = ImageDraw.Draw(img)
                for i in range(self.ancho + 1):
                    d.line([(i * zoom, 0), (i * zoom, img.height)], fill=(200, 200, 200))
                for j in range(self.alto + 1):
                    d.line([(0, j * zoom), (img.width, j * zoom)], fill=(200, 200, 200))
        return img

    def guardar(self, ruta, zoom=1, rejilla=False):
        self.a_imagen(zoom, rejilla).save(ruta)
