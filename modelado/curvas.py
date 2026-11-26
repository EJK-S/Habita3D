"""Curvas de Bézier (algoritmo de De Casteljau) y superficies de revolución."""
import numpy as np
from .malla import Malla


class Bezier:
    """Curva de Bézier de grado n evaluada mediante interpolación lineal sucesiva (De Casteljau)."""

    def __init__(self, puntos_control: list[tuple[float, float]] | np.ndarray):
        self.puntos_control = np.asarray(puntos_control, dtype=np.float64)
        if self.puntos_control.ndim != 2 or len(self.puntos_control) < 2:
            raise ValueError("La curva de Bézier requiere al menos 2 puntos de control.")

    def evaluar(self, t: float) -> np.ndarray:
        pts = self.puntos_control.copy()
        n = len(pts)
        for r in range(1, n):
            pts[: n - r] = (1.0 - t) * pts[: n - r] + t * pts[1 : n - r + 1]
        return pts[0]

    def niveles_de_casteljau(self, t: float) -> list[np.ndarray]:
        """
        Todos los niveles de la construcción de De Casteljau para el parámetro t:
        el nivel 0 son los puntos de control; cada nivel siguiente interpola linealmente
        los puntos consecutivos del anterior, (1 − t)·Pi + t·Pi+1, hasta quedar un solo
        punto, que es B(t). Sirve para dibujar la construcción geométrica de la curva.
        """
        niveles = [self.puntos_control.copy()]
        while len(niveles[-1]) > 1:
            p = niveles[-1]
            niveles.append((1.0 - t) * p[:-1] + t * p[1:])
        return niveles

    def muestrear(self, pasos: int = 24) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, pasos + 1)
        return np.array([self.evaluar(float(t)) for t in ts], dtype=np.float64)


def perfil_con_grosor(perfil: np.ndarray, grosor: float) -> np.ndarray:
    """
    Convierte un perfil abierto (cara exterior, de abajo hacia arriba) en un perfil cerrado
    con pared de espesor `grosor`: sube por la cara exterior, cruza el borde de la boca y
    baja por la cara interior, que es la exterior desplazada `grosor` hacia adentro a lo
    largo de la normal 2D del perfil.

    Si el perfil empieza en el eje (r = 0), la cara interior también termina en el eje y
    el fondo queda cerrado (jarrón, maceta). Si no, se vuelve al primer punto para cerrar
    el anillo inferior (pantalla de lámpara abierta por abajo).
    """
    tangente = np.gradient(perfil, axis=0)
    # Normal exterior del perfil recorrido hacia arriba: (dy, −dr). En un tramo vertical
    # apunta hacia +r (afuera); en el fondo casi horizontal apunta hacia abajo.
    normal = np.column_stack((tangente[:, 1], -tangente[:, 0]))
    largo = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.where(largo < 1e-12, 1.0, largo)

    interior = perfil - grosor * normal
    interior[:, 0] = np.maximum(interior[:, 0], 0.0)
    if perfil[0, 0] <= 1e-6:
        interior[0] = (0.0, perfil[0, 1] + grosor)        # fondo interior sobre el eje

    cerrado = [perfil, interior[::-1]]
    if perfil[0, 0] > 1e-6:
        cerrado.append(perfil[:1])                         # cierra el anillo inferior
    return np.vstack(cerrado)


def revolucion(
    curva: Bezier,
    segmentos_radiales: int = 24,
    pasos_perfil: int = 20,
    color: tuple[int, int, int] = (195, 115, 85),
    textura_id: str | None = "ceramica",
    nombre: str = "revolucion",
    grosor: float | None = None,
) -> Malla:
    """
    Genera una malla 3D rotando un perfil 2D (radio r >= 0, altura y) alrededor del eje Y.

    Sin `grosor` se obtiene la superficie abierta de una sola cara (útil para estudiar la
    superficie de revolución). Con `grosor` el perfil se cierra (cara exterior, borde y
    cara interior) y el objeto se vuelve un sólido hueco: así, al mirarlo desde arriba,
    se ve su interior en lugar de desaparecer por el descarte de caras traseras.
    """
    perfil = curva.muestrear(pasos_perfil)
    if grosor is not None and grosor > 0:
        perfil = perfil_con_grosor(perfil, grosor)
    radios = np.maximum(perfil[:, 0], 0.0)
    alturas = perfil[:, 1]

    n_filas = len(perfil)
    n_cols = segmentos_radiales + 1
    angulos = np.linspace(0.0, 2.0 * np.pi, n_cols)

    vertices, uvs = [], []
    for i in range(n_filas):
        r = radios[i]
        y = alturas[i]
        v_tex = i / max(n_filas - 1, 1)
        for j in range(n_cols):
            th = angulos[j]
            vertices.append([r * np.cos(th), y, -r * np.sin(th)])
            uvs.append([j / segmentos_radiales, v_tex])

    caras = []
    for i in range(n_filas - 1):
        for j in range(segmentos_radiales):
            p00 = i * n_cols + j
            p01 = i * n_cols + (j + 1)
            p10 = (i + 1) * n_cols + j
            p11 = (i + 1) * n_cols + (j + 1)

            # Orientación CCW vista desde el exterior radial
            if radios[i] > 1e-6:
                caras.append([p00, p01, p10])
            if radios[i + 1] > 1e-6:
                caras.append([p01, p11, p10])

    return Malla(
        vertices=np.array(vertices, dtype=np.float64),
        caras=np.array(caras, dtype=np.int32),
        uvs=np.array(uvs, dtype=np.float64),
        color=color,
        textura_id=textura_id,
        nombre=nombre,
    )
