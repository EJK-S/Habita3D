"""Modelo 2D de la vivienda. Las medidas están en metros con el eje Y hacia arriba;
solo al dibujar se convierten a píxeles (T · S con Mat3), tal como plantea el diseño:
todo se guarda de forma geométrica y se rasteriza al final."""
from dataclasses import dataclass, field
import numpy as np
from mathlib import Mat3
from raster2d import rellenar_poligono, dibujar_poligono, linea, circulo

COLOR_MURO = (70, 70, 80)
COLOR_VENTANA = (80, 150, 220)
COLOR_PUERTA = (150, 95, 45)
MARGEN_SELECCION_MURO = 0.10    # metros extra para que un muro delgado sea fácil de seleccionar


def punto_en_poligono(x, y, poligono):
    """Prueba par-impar (ray casting): se lanza un rayo horizontal desde (x, y) hacia +x
    y se cuentan los cruces con los lados; si el número es impar, el punto está dentro.
    Usa el mismo intervalo semiabierto que el relleno por líneas de barrido."""
    dentro = False
    n = len(poligono)
    for i in range(n):
        xa, ya = poligono[i]
        xb, yb = poligono[(i + 1) % n]
        if (ya > y) != (yb > y):
            x_cruce = xa + (y - ya) * (xb - xa) / (yb - ya)
            if x < x_cruce:
                dentro = not dentro
    return dentro


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
    escala_x: float = 1.0    # escalamiento a lo largo del ancho (eje x local del mueble)
    escala_y: float = 1.0    # escalamiento a lo largo del largo (eje y local del mueble)

    def matriz_modelo(self):
        """M = T · R · S: el mueble se define centrado en el origen, se escala en sus
        ejes locales, se gira y al final se traslada (vectores columna: S se aplica primero)."""
        return (Mat3.traslacion(self.x, self.y) @ Mat3.rotacion(self.angulo)
                @ Mat3.escalamiento(self.escala_x, self.escala_y))

    def poligono(self):
        a, l = self.ancho / 2, self.largo / 2
        local = np.array([(-a, -l), (a, -l), (a, l), (-a, l)])
        return self.matriz_modelo().aplicar_puntos(local)


@dataclass
class Figura:
    """
    Primitiva libre del editor 2D (módulo 1): 'punto', 'linea', 'circulo' o 'poligono'.
    Se guarda de forma geométrica, en metros: `puntos` son los vértices (el centro en el
    círculo) y `radio` el radio del círculo. Sus atributos gráficos son el color de trazo,
    el color de relleno (None = sin relleno), el grosor del pincel en píxeles y el algoritmo
    de líneas ('bresenham', 'dda' o 'wu' con antialiasing).
    """
    tipo: str
    puntos: list
    color: tuple = (30, 30, 30)
    relleno: tuple | None = None
    grosor: int = 2
    algoritmo: str = "bresenham"
    radio: float = 0.0

    def contorno(self, margen=0.06):
        """Polígono en metros que rodea la figura (para resaltarla y para detectar clics)."""
        if self.tipo == "circulo":
            t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            cx, cy = self.puntos[0]
            r = self.radio + margen
            return np.column_stack((cx + r * np.cos(t), cy + r * np.sin(t)))
        if self.tipo == "poligono":
            return np.array(self.puntos, float)
        if self.tipo == "linea":
            a, b = np.array(self.puntos[0], float), np.array(self.puntos[1], float)
            d = b - a
            largo = np.linalg.norm(d)
            if largo > 1e-9:
                u = d / largo
                n = np.array([-u[1], u[0]]) * margen
                u = u * margen
                return np.array([a - u + n, b + u + n, b + u - n, a - u - n])
        # punto (o línea de largo cero): un cuadrado pequeño
        x, y = self.puntos[0]
        return np.array([(x - margen, y - margen), (x + margen, y - margen),
                         (x + margen, y + margen), (x - margen, y + margen)])

    def dibujar(self, fb, a_pantalla):
        """Rasteriza la figura con los algoritmos propios (Bresenham/DDA/Wu, punto medio y
        relleno por líneas de barrido); `a_pantalla` es la matriz metros → píxeles."""
        pts = a_pantalla.aplicar_puntos(np.array(self.puntos, float))
        if self.tipo == "punto":
            fb.estampar(int(round(pts[0][0])), int(round(pts[0][1])), self.color, max(self.grosor, 1))
        elif self.tipo == "linea":
            linea(fb, pts[0][0], pts[0][1], pts[1][0], pts[1][1], self.color, self.grosor, self.algoritmo)
        elif self.tipo == "circulo":
            escala = float(np.hypot(*a_pantalla.m[:2, 0]))     # píxeles por metro
            circulo(fb, pts[0][0], pts[0][1], self.radio * escala, self.color, self.grosor, self.relleno)
        elif self.tipo == "poligono":
            if self.relleno is not None:
                rellenar_poligono(fb, pts, self.relleno)
            dibujar_poligono(fb, pts, self.color, self.grosor, cerrar=True, algoritmo=self.algoritmo)


@dataclass
class Plano:
    muros: list = field(default_factory=list)
    aberturas: list = field(default_factory=list)
    muebles: list = field(default_factory=list)
    figuras: list = field(default_factory=list)   # primitivas libres del editor 2D

    def agregar_figura(self, figura):
        self.figuras.append(figura)
        return len(self.figuras) - 1

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

    def poligono_abertura(self, ab):
        """Rectángulo del hueco de una puerta o ventana (ancho × espesor del muro)."""
        m, c, u, n = self._hueco(ab)
        a, h = ab.ancho / 2, m.espesor / 2 + 0.005
        return np.array([c - u * a + n * h, c + u * a + n * h, c + u * a - n * h, c - u * a - n * h])

    # ---- selección y borrado ----
    def poligono_elemento(self, tipo, i):
        """Contorno en metros del elemento ('figura', 'mueble', 'abertura' o 'muro') de índice i."""
        if tipo == "figura":
            return self.figuras[i].contorno()
        if tipo == "mueble":
            return self.muebles[i].poligono()
        if tipo == "abertura":
            return self.poligono_abertura(self.aberturas[i])
        return self.muros[i].poligono()

    def mueble_en(self, x, y):
        """Índice del mueble que está en el punto (x, y) en metros, o None.
        Si hay muebles encimados devuelve el de arriba (el último dibujado)."""
        for i in reversed(range(len(self.muebles))):
            if punto_en_poligono(x, y, self.muebles[i].poligono()):
                return i
        return None

    def elemento_en(self, x, y):
        """Devuelve (tipo, índice) del elemento que está en el punto (x, y) en metros, o None.
        Se revisa en el orden inverso al dibujo: primero las figuras libres (van encima de
        todo), luego muebles, puertas y ventanas, y al final muros."""
        for i in reversed(range(len(self.figuras))):
            if punto_en_poligono(x, y, self.figuras[i].contorno()):
                return ("figura", i)
        i = self.mueble_en(x, y)
        if i is not None:
            return ("mueble", i)
        for i in reversed(range(len(self.aberturas))):
            if punto_en_poligono(x, y, self.poligono_abertura(self.aberturas[i])):
                return ("abertura", i)
        i = self.muro_en(x, y)
        return ("muro", i) if i is not None else None

    def muro_en(self, x, y):
        """Índice del muro en el punto (x, y), con un margen para que sea fácil de apuntar."""
        for i in reversed(range(len(self.muros))):
            m = self.muros[i]
            if m.largo() < 1e-9:
                continue
            ampliado = Muro(m.p1, m.p2, m.espesor + MARGEN_SELECCION_MURO)
            if punto_en_poligono(x, y, ampliado.poligono()):
                return i
        return None

    # ---- colocación de puertas y ventanas ----
    def posicion_en_muro(self, i, x, y, ancho):
        """
        Posición (0 a 1, centro del vano) más cercana al punto (x, y) sobre el muro i:
        se proyecta el punto sobre la dirección del muro (producto punto) y se limita para
        que el vano quede entero, sin invadir los muros que llegan a las esquinas.
        Devuelve None si el vano no cabe en el muro.
        """
        m = self.muros[i]
        largo = m.largo()
        margen = ancho / 2 + m.espesor / 2 + 0.05
        if largo < 2 * margen:
            return None
        t = float(np.dot(np.array((x, y), float) - np.array(m.p1, float), m.direccion()))
        return min(max(t, margen), largo - margen) / largo

    def abertura_se_superpone(self, i, pos, ancho, separacion=0.0):
        """¿Un vano de `ancho` centrado en `pos` del muro i se superpone con otro vano de ese
        muro? Dos vanos que solo se tocan (p. ej. dos ventanas juntas) no se superponen."""
        largo = self.muros[i].largo()
        centro = pos * largo
        for ab in self.aberturas:
            if ab.muro == i and abs(centro - ab.pos * largo) < (ancho + ab.ancho) / 2 + separacion - 1e-6:
                return True
        return False

    def punto_libre(self, x, y, radio=0.25):
        """¿Puede estar una persona (círculo de `radio` m) en (x, y)? Los muros bloquean;
        el paso por una puerta queda libre (por su centro) y las ventanas no se cruzan."""
        for i, m in enumerate(self.muros):
            if m.largo() < 1e-9:
                continue
            u = m.direccion()
            rel = np.array((x, y), float) - np.array(m.p1, float)
            t = float(np.dot(rel, u))                      # a lo largo del muro
            d = float(u[0] * rel[1] - u[1] * rel[0])       # distancia perpendicular
            h = m.espesor / 2 + radio
            if abs(d) >= h or not (-h < t < m.largo() + h):
                continue
            pasa = any(ab.muro == i and ab.tipo == "puerta"
                       and abs(t - ab.pos * m.largo()) <= max(ab.ancho / 2 - 0.1, 0.05)
                       for ab in self.aberturas)
            if not pasa:
                return False
        return True

    def ajustar_junto_a_vecinos(self, i, pos, ancho, iman=0.15):
        """
        Ajuste magnético de un vano sobre el muro i. Si queda encima de otro vano o a menos
        de `iman` metros de él, se desplaza para quedar pegado a ese vano (del lado más cercano
        al cursor), así dos ventanas se pueden juntar sin tener que atinar al píxel.
        Devuelve la posición ajustada (0 a 1), o None si ahí no cabe sin superponerse.
        El ajuste nunca mueve el vano más de su propio ancho + `iman`, para que no salte lejos
        del cursor (por ejemplo, al otro lado de una ventana vecina).
        """
        m = self.muros[i]
        largo = m.largo()
        margen = ancho / 2 + m.espesor / 2 + 0.05
        centro = pos * largo
        vecinos = [(ab.pos * largo, ab.ancho / 2) for ab in self.aberturas if ab.muro == i]

        def valido(c):
            return margen - 1e-9 <= c <= largo - margen + 1e-9 and not self.abertura_se_superpone(i, c / largo, ancho)

        cercanos = [(cv, hv) for cv, hv in vecinos if abs(centro - cv) < ancho / 2 + hv + iman]
        if not cercanos:
            return pos if valido(centro) else None
        # Posiciones pegadas a cada vecino cercano, de la más próxima al cursor a la más lejana
        pegadas = sorted((c for cv, hv in cercanos for c in (cv - hv - ancho / 2, cv + hv + ancho / 2)),
                         key=lambda c: abs(c - centro))
        for c in pegadas:
            if abs(c - centro) <= ancho + iman and valido(c):
                return c / largo
        return None

    def eliminar(self, tipo, i):
        """Borra un elemento. Al borrar un muro se borran también sus puertas y ventanas,
        y se corrigen los índices de muro de las aberturas que quedan."""
        if tipo == "figura":
            del self.figuras[i]
        elif tipo == "mueble":
            del self.muebles[i]
        elif tipo == "abertura":
            del self.aberturas[i]
        elif tipo == "muro":
            self.aberturas = [ab for ab in self.aberturas if ab.muro != i]
            for ab in self.aberturas:
                if ab.muro > i:
                    ab.muro -= 1
            del self.muros[i]
        else:
            raise ValueError(f"Tipo de elemento desconocido: {tipo}")

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
            a = ab.ancho / 2
            rellenar_poligono(fb, pant(self.poligono_abertura(ab)), fb.fondo)
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

        for fig in self.figuras:            # primitivas libres, encima de todo
            fig.dibujar(fb, a_pantalla)
