"""Modelo de iluminación de Phong y sombreados plano y de Gouraud."""
import numpy as np
from mathlib import Vec3


class Luz:
    """Fuente de luz puntual en 3D con parámetros del modelo de reflexión de Phong."""

    def __init__(
        self,
        posicion: Vec3 | tuple = (3.0, 2.4, -2.2),
        intensidad_ambiental: float = 0.32,
        intensidad_luz: float = 0.85,
        ka: float = 1.0,
        kd: float = 0.85,
        ks: float = 0.35,
        brillo_n: float = 24.0,
    ):
        self.posicion = posicion if isinstance(posicion, Vec3) else Vec3(*posicion)
        self.ia = float(intensidad_ambiental)
        self.il = float(intensidad_luz)
        self.ka = float(ka)
        self.kd = float(kd)
        self.ks = float(ks)
        self.n = float(brillo_n)


def calcular_phong_vectorizado(
    puntos: np.ndarray,
    normales: np.ndarray,
    colores_base: np.ndarray,
    pos_camara: np.ndarray,
    luz: Luz,
) -> np.ndarray:
    """
    Evalúa I = Ia*ka + Il*kd*max(0, N·L) + Il*ks*max(0, R·V)^n de forma vectorizada
    sobre un arreglo de puntos (M, 3) y normales unitarias (M, 3).
    Devuelve factores o colores RGB multiplicados en [0, 1] de forma (M, 3).
    """
    pos_luz = np.array([luz.posicion.x, luz.posicion.y, luz.posicion.z], dtype=np.float64)

    L = pos_luz[None, :] - puntos
    dist_l = np.linalg.norm(L, axis=1, keepdims=True)
    dist_l = np.where(dist_l < 1e-9, 1.0, dist_l)
    L = L / dist_l

    V = pos_camara[None, :] - puntos
    dist_v = np.linalg.norm(V, axis=1, keepdims=True)
    dist_v = np.where(dist_v < 1e-9, 1.0, dist_v)
    V = V / dist_v

    ndotl = np.maximum(0.0, np.einsum("ij,ij->i", normales, L))

    # Vector reflejado R = 2*(N·L)*N - L
    R = 2.0 * ndotl[:, None] * normales - L
    rdotv = np.maximum(0.0, np.einsum("ij,ij->i", R, V))

    ambiental = luz.ia * luz.ka
    difusa = luz.il * luz.kd * ndotl
    especular = np.where(ndotl > 0.0, luz.il * luz.ks * (rdotv ** luz.n), 0.0)

    # Color final: (ambiental + difusa) * color_objeto + especular blanco
    intensidad_mat = (ambiental + difusa)[:, None]
    rgb = colores_base * intensidad_mat + especular[:, None]
    return np.clip(rgb, 0.0, 1.0)


def sombreado_plano(
    vertices: np.ndarray,
    caras: np.ndarray,
    normales_cara: np.ndarray,
    color_base: tuple[int, int, int],
    pos_camara: np.ndarray,
    luz: Luz,
) -> np.ndarray:
    """
    Calcula un color RGB por cada triángulo evaluando Phong en el centroide de la cara.
    Devuelve arreglo (M, 3) en [0, 1].
    """
    if len(caras) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    v0 = vertices[caras[:, 0]]
    v1 = vertices[caras[:, 1]]
    v2 = vertices[caras[:, 2]]
    centroides = (v0 + v1 + v2) / 3.0
    col_norm = np.tile(np.asarray(color_base, dtype=np.float64) / 255.0, (len(caras), 1))
    return calcular_phong_vectorizado(centroides, normales_cara, col_norm, pos_camara, luz)


def sombreado_gouraud(
    vertices: np.ndarray,
    normales_vertice: np.ndarray,
    color_base: tuple[int, int, int],
    pos_camara: np.ndarray,
    luz: Luz,
) -> np.ndarray:
    """
    Calcula un color RGB por cada vértice evaluando Phong con su normal promediada.
    Luego el rasterizador interpola estos colores en el interior del triángulo.
    Devuelve arreglo (N, 3) en [0, 1].
    """
    if len(vertices) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    col_norm = np.tile(np.asarray(color_base, dtype=np.float64) / 255.0, (len(vertices), 1))
    return calcular_phong_vectorizado(vertices, normales_vertice, col_norm, pos_camara, luz)
