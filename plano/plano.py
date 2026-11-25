"""Modelo 2D de la vivienda. Las medidas están en metros con el eje Y hacia arriba;
solo al dibujar se convierten a píxeles (T · S con Mat3), tal como plantea el diseño:
todo se guarda de forma geométrica y se rasteriza al final."""
from dataclasses import dataclass, field
import numpy as np
from mathlib import Mat3
from raster2d import rellenar_poligono, dibujar_poligono, linea

COLOR_MURO = (70, 70, 80)
COLOR_VENTANA = (80, 150, 220)
COLOR_PUERTA = (150, 95, 45)


@dataclass
class Muro:
    p1: tuple
    p2: tuple
    espesor: float = 0.15

    def largo(self):
        return float(np.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]))

    def direccion(self):
        d = np.array(self.p2, float) - np.array(self.p1, float)
        return d / np.linalg.norm(d)

    def poligono(self):
        """Rectángulo del muro; se prolonga medio espesor en cada extremo para que las esquinas cierren."""
        u = self.direccion()
        n = np.array([-u[1], u[0]])
        h = self.espesor / 2
        a = np.array(self.p1, float) - u * h
        b = np.array(self.p2, float) + u * h
        return np.array([a + n * h, b + n * h, b - n * h, a - n * h])


@dataclass
class Abertura:
    muro: int            # índice del muro en el plano
    pos: float           # posición del centro sobre el muro (0 a 1)
    ancho: float = 0.9
    tipo: str = "puerta"  # "puerta" o "ventana"


@dataclass
class MuebleEnPlanta:
    nombre: str
    ancho: float
    largo: float
    x: float = 0.0
    y: float = 0.0
    angulo: float = 0.0
    color: tuple = (190, 160, 120)

    def matriz_modelo(self):
        """M = T · R · S: el mueble se define centrado en el origen, se gira y luego se traslada."""
        return Mat3.traslacion(self.x, self.y) @ Mat3.rotacion(self.angulo) @ Mat3.escalamiento(1, 1)

    def poligono(self):
        a, l = self.ancho / 2, self.largo / 2
        local = np.array([(-a, -l), (a, -l), (a, l), (-a, l)])
        return self.matriz_modelo().aplicar_puntos(local)


@dataclass
class Plano:
    muros: list = field(default_factory=list)
    aberturas: list = field(default_factory=list)
    muebles: list = field(default_factory=list)

    def agregar_muro(self, p1, p2, espesor=0.15):
        self.muros.append(Muro(p1, p2, espesor))
        return len(self.muros) - 1

    def agregar_abertura(self, muro, pos, ancho=0.9, tipo="puerta"):
        self.aberturas.append(Abertura(muro, pos, ancho, tipo))

    def agregar_mueble(self, mueble):
        self.muebles.append(mueble)

    def agregar_habitacion(self, x0, y0, x1, y1, espesor=0.15):
        """Cuatro muros en rectángulo; devuelve los índices (sur, este, norte, oeste)."""
        return [self.agregar_muro((x0, y0), (x1, y0), espesor),
                self.agregar_muro((x1, y0), (x1, y1), espesor),
                self.agregar_muro((x1, y1), (x0, y1), espesor),
                self.agregar_muro((x0, y1), (x0, y0), espesor)]

    def _hueco(self, ab):
        """Centro, dirección y normal de una abertura sobre su muro."""
        m = self.muros[ab.muro]
        u = m.direccion()
        c = np.array(m.p1, float) + u * (ab.pos * m.largo())
        return m, c, u, np.array([-u[1], u[0]])

    def dibujar(self, fb, escala=60.0, origen=(40, None), cuadricula=True):
        """Dibuja el plano en el framebuffer. 'escala' = píxeles por metro."""
        oy = fb.alto - 40 if origen[1] is None else origen[1]
        a_pantalla = Mat3.traslacion(origen[0], oy) @ Mat3.escalamiento(escala, -escala)
        pant = lambda pts: a_pantalla.aplicar_puntos(np.asarray(pts, float))

        if cuadricula:
            for k in range(-1, int((fb.ancho) / escala) + 2):
                p = pant([(k, -50), (k, 50)])
                linea(fb, p[0][0], p[0][1], p[1][0], p[1][1], (232, 232, 232))
            for k in range(-1, int((fb.alto) / escala) + 2):
                p = pant([(-50, k), (50, k)])
                linea(fb, p[0][0], p[0][1], p[1][0], p[1][1], (232, 232, 232))

        for m in self.muros:
            rellenar_poligono(fb, pant(m.poligono()), COLOR_MURO)

        for ab in self.aberturas:           # el hueco se pinta encima del muro
            m, c, u, n = self._hueco(ab)
            a, h = ab.ancho / 2, m.espesor / 2 + 0.005
            hueco = [c - u * a + n * h, c + u * a + n * h, c + u * a - n * h, c - u * a - n * h]
            rellenar_poligono(fb, pant(hueco), fb.fondo)
            if ab.tipo == "ventana":
                p = pant([c - u * a, c + u * a])
                linea(fb, p[0][0], p[0][1], p[1][0], p[1][1], COLOR_VENTANA, 3)
            else:                           # hoja de la puerta abierta a 90° + arco de giro
                bisagra = c - u * a
                punta = bisagra + n * ab.ancho
                p = pant([bisagra, punta])
                linea(fb, p[0][0], p[0][1], p[1][0], p[1][1], COLOR_PUERTA, 2)
                ang = np.linspace(0, np.pi / 2, 14)
                arco = [bisagra + ab.ancho * (np.cos(t) * u + np.sin(t) * n) for t in ang]
                dibujar_poligono(fb, pant(arco), COLOR_PUERTA, 1, cerrar=False)

        for mu in self.muebles:
            rellenar_poligono(fb, pant(mu.poligono()), mu.color, (60, 45, 30), 1)
