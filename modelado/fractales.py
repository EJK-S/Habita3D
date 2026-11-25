"""Generación recursiva de plantas decorativas mediante árboles fractales 3D."""
import numpy as np
from .malla import Malla


def _prisma_rama(
    inicio: np.ndarray,
    fin: np.ndarray,
    r_base: float,
    r_tope: float,
    lados: int = 5,
    color: tuple[int, int, int] = (115, 78, 46),
) -> Malla:
    eje = fin - inicio
    largo = float(np.linalg.norm(eje))
    if largo < 1e-6:
        return Malla(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), color=color)

    t = eje / largo
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    n = np.cross(t, ref)
    n /= np.linalg.norm(n)
    b = np.cross(t, n)

    angulos = np.linspace(0.0, 2.0 * np.pi, lados, endpoint=False)
    verts, uvs = [], []
    for k, a in enumerate(angulos):
        dir_r = np.cos(a) * n + np.sin(a) * b
        verts.append(inicio + r_base * dir_r)
        uvs.append([k / lados, 0.0])
    for k, a in enumerate(angulos):
        dir_r = np.cos(a) * n + np.sin(a) * b
        verts.append(fin + r_tope * dir_r)
        uvs.append([k / lados, 1.0])

    caras = []
    for i in range(lados):
        sig = (i + 1) % lados
        caras.append([i, sig, lados + i])
        caras.append([sig, lados + sig, lados + i])

    return Malla(
        vertices=np.array(verts, dtype=np.float64),
        caras=np.array(caras, dtype=np.int32),
        uvs=np.array(uvs, dtype=np.float64),
        color=color,
        textura_id=None,
        nombre="rama_fractal",
    )


def _rotar_rodrigues(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)


def arbol_fractal(
    profundidad: int = 4,
    longitud_inicial: float = 0.42,
    radio_inicial: float = 0.032,
    angulo_grados: float = 28.0,
    factor_escala: float = 0.72,
    ramas_por_nodo: int = 3,
) -> Malla:
    """
    Construye un árbol fractal 3D recursivo: cada rama se divide en `ramas_por_nodo`
    sub-ramas escaladas por `factor_escala` e inclinadas `angulo_grados`.
    """
    piezas: list[Malla] = []
    ang_rad = np.radians(angulo_grados)

    def _recursivo(ini: np.ndarray, dir_u: np.ndarray, largo: float, radio: float, nivel: int) -> None:
        if nivel <= 0:
            return
        fin = ini + dir_u * largo
        r_sig = radio * factor_escala
        es_hoja = nivel == 1
        color = (52, 150, 68) if es_hoja else (115, 78, 46)
        lados = 4 if es_hoja else 5
        piezas.append(_prisma_rama(ini, fin, radio, r_sig * (0.25 if es_hoja else 1.0), lados, color))

        if nivel > 1:
            ref = np.array([1.0, 0.0, 0.0]) if abs(dir_u[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
            p1 = np.cross(dir_u, ref)
            p1 /= np.linalg.norm(p1)
            p2 = np.cross(dir_u, p1)

            for idx in range(ramas_por_nodo):
                az = (2.0 * np.pi * idx / ramas_por_nodo) + nivel * 0.65
                eje_rot = np.cos(az) * p1 + np.sin(az) * p2
                nueva_dir = _rotar_rodrigues(dir_u, eje_rot, ang_rad)
                nueva_dir /= np.linalg.norm(nueva_dir)
                _recursivo(fin, nueva_dir, largo * factor_escala, r_sig, nivel - 1)

    _recursivo(
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        longitud_inicial,
        radio_inicial,
        profundidad,
    )
    return Malla.combinar(piezas, nombre=f"arbol_fractal_n{profundidad}")
