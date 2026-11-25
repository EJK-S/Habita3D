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

    def muestrear(self, pasos: int = 24) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, pasos + 1)
        return np.array([self.evaluar(float(t)) for t in ts], dtype=np.float64)


def revolucion(
    curva: Bezier,
    segmentos_radiales: int = 24,
    pasos_perfil: int = 20,
    color: tuple[int, int, int] = (195, 115, 85),
    textura_id: str | None = "ceramica",
    nombre: str = "revolucion",
) -> Malla:
    """
    Genera una malla 3D rotando un perfil 2D (radio r >= 0, altura y) alrededor del eje Y.
    """
    perfil = curva.muestrear(pasos_perfil)
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
