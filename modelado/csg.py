"""
Geometría constructiva de sólidos (CSG) general con árboles BSP.

Un sólido se describe por los polígonos convexos de su frontera (B-Rep), orientados con la
normal hacia afuera. Para combinar dos sólidos A y B:

1. Se construye un árbol BSP (partición binaria del espacio) con los polígonos de cada uno:
   cada nodo guarda un plano; los polígonos que están delante van al subárbol "frente" y los
   que están detrás, al subárbol "atrás"; un polígono que cruza el plano se divide en dos.
2. Con el árbol de B se recortan los polígonos de A que quedan dentro de B (y viceversa):
   un polígono que baja por el árbol y termina en una hoja "atrás" vacía está dentro del sólido.
3. Invirtiendo los árboles (dentro ↔ fuera) y combinando lo que queda se obtienen:
     A ∪ B (unión), A ∩ B (intersección) y A − B (diferencia).

Es el algoritmo clásico de CSG por BSP (como el de la biblioteca csg.js), implementado aquí
desde cero. Funciona con sólidos cualesquiera (cajas, esferas, cilindros y sus combinaciones).
Además se ofrece un árbol CSG explícito (`HojaCSG` / `OperacionCSG`): las hojas son primitivas
y los nodos internos son operaciones, tal como se describe en el marco teórico.
"""
from functools import reduce
import numpy as np
from .malla import Malla

EPS = 1e-5                                   # tolerancia para decidir si un punto está en el plano
COPLANAR, FRENTE, ATRAS, AMBOS = 0, 1, 2, 3


# --------------------------------------------------------------------------- planos y polígonos
class PlanoBSP:
    __slots__ = ("normal", "w")

    def __init__(self, normal: np.ndarray, w: float):
        self.normal = normal
        self.w = w                           # el plano es {p : normal · p = w}

    @staticmethod
    def desde_puntos(a, b, c) -> "PlanoBSP":
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        return PlanoBSP(n, float(np.dot(n, a)))

    def invertido(self) -> "PlanoBSP":
        return PlanoBSP(-self.normal, -self.w)


class Poligono:
    """Polígono convexo plano con su color (RGB 0–255) y textura, que se conservan al dividirlo."""
    __slots__ = ("vertices", "plano", "color", "textura")

    def __init__(self, vertices: list, color=(200, 200, 200), textura=None, plano: PlanoBSP | None = None):
        self.vertices = vertices
        self.color = color
        self.textura = textura
        self.plano = plano if plano is not None else PlanoBSP.desde_puntos(*vertices[:3])

    def invertido(self) -> "Poligono":
        return Poligono(self.vertices[::-1], self.color, self.textura, self.plano.invertido())


def dividir_poligono(plano: PlanoBSP, poli: Poligono, cop_frente: list, cop_atras: list,
                     frente: list, atras: list) -> None:
    """Clasifica `poli` respecto a `plano` y lo agrega a la lista que corresponde; si lo cruza,
    lo corta en dos polígonos (uno a cada lado) interpolando los vértices sobre el plano."""
    tipos = []
    tipo_poli = COPLANAR
    for v in poli.vertices:
        t = float(np.dot(plano.normal, v)) - plano.w
        tipo = ATRAS if t < -EPS else (FRENTE if t > EPS else COPLANAR)
        tipo_poli |= tipo
        tipos.append(tipo)

    if tipo_poli == COPLANAR:
        (cop_frente if np.dot(plano.normal, poli.plano.normal) > 0 else cop_atras).append(poli)
    elif tipo_poli == FRENTE:
        frente.append(poli)
    elif tipo_poli == ATRAS:
        atras.append(poli)
    else:                                    # cruza el plano: se divide
        f, b = [], []
        n = len(poli.vertices)
        for i in range(n):
            j = (i + 1) % n
            ti, tj = tipos[i], tipos[j]
            vi, vj = poli.vertices[i], poli.vertices[j]
            if ti != ATRAS:
                f.append(vi)
            if ti != FRENTE:
                b.append(vi)
            if (ti | tj) == AMBOS:           # la arista cruza el plano: punto de corte
                t = (plano.w - np.dot(plano.normal, vi)) / np.dot(plano.normal, vj - vi)
                v = vi + t * (vj - vi)
                f.append(v)
                b.append(v)
        if len(f) >= 3:
            frente.append(Poligono(f, poli.color, poli.textura, poli.plano))
        if len(b) >= 3:
            atras.append(Poligono(b, poli.color, poli.textura, poli.plano))


# --------------------------------------------------------------------------------- árbol BSP
class NodoBSP:
    """Nodo de un árbol BSP. Los recorridos son iterativos (pila) para no depender del
    límite de recursión de Python con mallas de muchos polígonos."""
    __slots__ = ("plano", "frente", "atras", "poligonos")

    def __init__(self, poligonos: list | None = None):
        self.plano = None
        self.frente = None
        self.atras = None
        self.poligonos = []
        if poligonos:
            self.construir(poligonos)

    def _nodos(self):
        pila = [self]
        while pila:
            n = pila.pop()
            yield n
            pila.extend(x for x in (n.frente, n.atras) if x is not None)

    def invertir(self) -> None:
        """Intercambia dentro y fuera del sólido."""
        for n in list(self._nodos()):
            n.poligonos = [p.invertido() for p in n.poligonos]
            if n.plano is not None:
                n.plano = n.plano.invertido()
            n.frente, n.atras = n.atras, n.frente

    def recortar_poligonos(self, poligonos: list) -> list:
        """Devuelve las partes de `poligonos` que quedan fuera del sólido de este árbol."""
        resultado = []
        pila = [(self, poligonos)]
        while pila:
            n, polis = pila.pop()
            if n.plano is None:
                resultado.extend(polis)
                continue
            f, b = [], []
            for p in polis:
                dividir_poligono(n.plano, p, f, b, f, b)
            if n.frente is not None:
                pila.append((n.frente, f))
            else:
                resultado.extend(f)
            if n.atras is not None:
                pila.append((n.atras, b))
            # sin subárbol "atrás": esas partes están dentro del sólido y se descartan
        return resultado

    def recortar_con(self, otro: "NodoBSP") -> None:
        """Quita de este árbol todo lo que está dentro del sólido de `otro`."""
        for n in list(self._nodos()):
            n.poligonos = otro.recortar_poligonos(n.poligonos)

    def todos_los_poligonos(self) -> list:
        return [p for n in self._nodos() for p in n.poligonos]

    def construir(self, poligonos: list) -> None:
        pila = [(self, poligonos)]
        while pila:
            n, polis = pila.pop()
            if not polis:
                continue
            if n.plano is None:
                n.plano = polis[0].plano
            f, b = [], []
            for p in polis:
                dividir_poligono(n.plano, p, n.poligonos, n.poligonos, f, b)
            if f:
                if n.frente is None:
                    n.frente = NodoBSP()
                pila.append((n.frente, f))
            if b:
                if n.atras is None:
                    n.atras = NodoBSP()
                pila.append((n.atras, b))


# ------------------------------------------------------------------------------- sólidos CSG
class SolidoCSG:
    """Sólido como lista de polígonos de frontera. Operadores: a | b, a & b, a - b."""

    def __init__(self, poligonos: list):
        self.poligonos = poligonos

    def union(self, otro: "SolidoCSG") -> "SolidoCSG":
        a, b = NodoBSP(self.poligonos), NodoBSP(otro.poligonos)
        a.recortar_con(b)            # quita de A lo que está dentro de B
        b.recortar_con(a)            # quita de B lo que está dentro de A
        b.invertir()
        b.recortar_con(a)            # elimina caras coplanares repetidas
        b.invertir()
        a.construir(b.todos_los_poligonos())
        return SolidoCSG(a.todos_los_poligonos())

    def diferencia(self, otro: "SolidoCSG") -> "SolidoCSG":
        a, b = NodoBSP(self.poligonos), NodoBSP(otro.poligonos)
        a.invertir()                 # A − B = complemento(complemento(A) ∪ B)
        a.recortar_con(b)
        b.recortar_con(a)
        b.invertir()
        b.recortar_con(a)
        b.invertir()
        a.construir(b.todos_los_poligonos())
        a.invertir()
        return SolidoCSG(a.todos_los_poligonos())

    def interseccion(self, otro: "SolidoCSG") -> "SolidoCSG":
        a, b = NodoBSP(self.poligonos), NodoBSP(otro.poligonos)
        a.invertir()                 # A ∩ B = complemento(complemento(A) ∪ complemento(B))
        b.recortar_con(a)
        b.invertir()
        a.recortar_con(b)
        b.recortar_con(a)
        a.construir(b.todos_los_poligonos())
        a.invertir()
        return SolidoCSG(a.todos_los_poligonos())

    __or__ = union
    __and__ = interseccion
    __sub__ = diferencia

    @staticmethod
    def desde_malla(malla: Malla) -> "SolidoCSG":
        polis = []
        for k, cara in enumerate(malla.caras):
            v = [malla.vertices[i].copy() for i in cara]
            if np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0])) < 1e-12:
                continue                     # triángulo degenerado
            color = tuple(int(c) for c in malla.colores_vertice[cara[0]])
            polis.append(Poligono(v, color, malla.texturas_cara[k]))
        return SolidoCSG(polis)

    def a_malla(self, nombre: str = "csg", escala_uv: float = 1.0) -> Malla:
        """Triangula cada polígono (en abanico) y arma una Malla; cada polígono lleva sus
        propios vértices, así la normal de cada cara es exacta (aristas vivas)."""
        verts, caras, colores, texturas = [], [], [], []
        for p in self.poligonos:
            base = len(verts)
            verts.extend(p.vertices)
            colores.extend([p.color] * len(p.vertices))
            for i in range(1, len(p.vertices) - 1):
                caras.append((base, base + i, base + i + 1))
                texturas.append(p.textura)
        if not caras:
            return Malla(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), nombre=nombre)
        color0 = self.poligonos[0].color
        malla = Malla(np.array(verts, dtype=np.float64), np.array(caras, dtype=np.int32),
                      color=color0, textura_id=self.poligonos[0].textura, nombre=nombre,
                      colores_vertice=np.array(colores, dtype=np.float64), texturas_cara=texturas)
        malla.aplicar_mapeado_uv(tipo="cubico", escala=escala_uv)
        return malla


# -------------------------------------------------------------------------------- primitivas
def _orientar_hacia_afuera(polis: list, centro: np.ndarray) -> list:
    """En un sólido convexo cada cara debe mirar en sentido contrario al centro."""
    resultado = []
    for p in polis:
        c = np.mean(p.vertices, axis=0)
        resultado.append(p if np.dot(p.plano.normal, c - centro) > 0 else p.invertido())
    return resultado


def _sin_repetidos(vertices: list) -> list:
    """Quita vértices consecutivos repetidos (en los polos de la esfera un cuadrilátero es un triángulo)."""
    limpio = []
    for v in vertices:
        if not limpio or np.linalg.norm(v - limpio[-1]) > 1e-12:
            limpio.append(v)
    if len(limpio) > 1 and np.linalg.norm(limpio[0] - limpio[-1]) <= 1e-12:
        limpio.pop()
    return limpio


def caja(cx, cy, cz, ancho, alto, largo, color=(200, 200, 200), textura=None) -> SolidoCSG:
    """Caja alineada con los ejes, centrada en (cx, cy, cz): 6 caras cuadrilaterales."""
    c = np.array([cx, cy, cz], dtype=np.float64)
    r = np.array([ancho, alto, largo], dtype=np.float64) / 2
    esquina = lambda i: c + r * np.array([1 if i & 1 else -1, 1 if i & 2 else -1, 1 if i & 4 else -1])
    caras = [(0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3), (0, 2, 3, 1), (4, 5, 7, 6)]
    polis = [Poligono([esquina(i) for i in cara], color, textura) for cara in caras]
    return SolidoCSG(_orientar_hacia_afuera(polis, c))


def esfera(cx, cy, cz, radio, color=(200, 200, 200), textura=None, sectores=20, anillos=12) -> SolidoCSG:
    """Esfera aproximada por cuadriláteros (y triángulos en los polos) de latitud y longitud."""
    c = np.array([cx, cy, cz], dtype=np.float64)

    def punto(i, j):
        th, ph = 2 * np.pi * i / sectores, np.pi * j / anillos
        return c + radio * np.array([np.cos(th) * np.sin(ph), np.cos(ph), np.sin(th) * np.sin(ph)])

    polis = []
    for i in range(sectores):
        for j in range(anillos):
            v = _sin_repetidos([punto(i, j), punto(i + 1, j), punto(i + 1, j + 1), punto(i, j + 1)])
            if len(v) >= 3:
                polis.append(Poligono(v, color, textura))
    return SolidoCSG(_orientar_hacia_afuera(polis, c))


def cilindro(cx, cy, cz, radio, altura, color=(200, 200, 200), textura=None, sectores=24) -> SolidoCSG:
    """Cilindro vertical (eje Y) con base en y = cy: caras laterales y dos tapas poligonales."""
    ang = np.linspace(0, 2 * np.pi, sectores, endpoint=False)
    abajo = [np.array([cx + radio * np.cos(a), cy, cz + radio * np.sin(a)]) for a in ang]
    arriba = [p + np.array([0.0, altura, 0.0]) for p in abajo]
    polis = [Poligono(list(abajo), color, textura), Poligono(list(arriba), color, textura)]
    for i in range(sectores):
        j = (i + 1) % sectores
        polis.append(Poligono([abajo[i], abajo[j], arriba[j], arriba[i]], color, textura))
    return SolidoCSG(_orientar_hacia_afuera(polis, np.array([cx, cy + altura / 2, cz])))


# ------------------------------------------------------------------------- árbol CSG explícito
class HojaCSG:
    """Hoja del árbol CSG: una primitiva."""

    def __init__(self, nombre: str, solido: SolidoCSG):
        self.nombre = nombre
        self.solido = solido

    def evaluar(self) -> SolidoCSG:
        return self.solido

    def describir(self) -> str:
        return self.nombre


class OperacionCSG:
    """Nodo interno del árbol CSG: 'union', 'interseccion' o 'diferencia' de sus hijos."""
    SIMBOLOS = {"union": "∪", "interseccion": "∩", "diferencia": "−"}

    def __init__(self, operacion: str, *hijos):
        if operacion not in self.SIMBOLOS:
            raise ValueError(f"Operación CSG desconocida: {operacion}")
        self.operacion = operacion
        self.hijos = hijos

    def evaluar(self) -> SolidoCSG:
        solidos = [h.evaluar() for h in self.hijos]
        metodo = {"union": SolidoCSG.union, "interseccion": SolidoCSG.interseccion,
                  "diferencia": SolidoCSG.diferencia}[self.operacion]
        return reduce(metodo, solidos)

    def describir(self) -> str:
        return "(" + f" {self.SIMBOLOS[self.operacion]} ".join(h.describir() for h in self.hijos) + ")"


def volumen(malla: Malla) -> float:
    """Volumen encerrado por una malla cerrada (teorema de la divergencia: suma de tetraedros)."""
    v = malla.vertices[malla.caras]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)
