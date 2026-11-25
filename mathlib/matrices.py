"""Matrices homogéneas: Mat3 (transformaciones 2D) y Mat4 (transformaciones 3D).

Convención: vectores columna, P' = M · P. Por eso en T · R · S se aplica primero
el escalamiento, luego la rotación y al final la traslación.
Los ángulos se dan en grados; la rotación positiva es antihoraria en un sistema
con el eje Y hacia arriba (el del plano de la vivienda, en metros).
"""
import math
import numpy as np
from .vectores import Vec2, Vec3


def _cs(grados):
    a = math.radians(grados)
    return math.cos(a), math.sin(a)


class Mat3:
    """Matriz 3×3 en coordenadas homogéneas para el plano."""

    def __init__(self, m=None):
        self.m = np.eye(3) if m is None else np.array(m, dtype=float).reshape(3, 3)

    # ---- constructores de transformaciones básicas ----
    @staticmethod
    def identidad():
        return Mat3()

    @staticmethod
    def traslacion(tx, ty):
        m = np.eye(3)
        m[0, 2], m[1, 2] = tx, ty
        return Mat3(m)

    @staticmethod
    def escalamiento(sx, sy=None):
        sy = sx if sy is None else sy
        return Mat3(np.diag([sx, sy, 1.0]))

    @staticmethod
    def rotacion(grados):
        c, s = _cs(grados)
        return Mat3([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @staticmethod
    def reflexion_x():
        """Reflexión respecto al eje X: (x, y) → (x, −y)."""
        return Mat3(np.diag([1.0, -1.0, 1.0]))

    @staticmethod
    def reflexion_y():
        """Reflexión respecto al eje Y: (x, y) → (−x, y)."""
        return Mat3(np.diag([-1.0, 1.0, 1.0]))

    @staticmethod
    def cizallamiento(shx=0.0, shy=0.0):
        return Mat3([[1, shx, 0], [shy, 1, 0], [0, 0, 1]])

    # ---- transformaciones respecto a un punto (trasladar–transformar–volver) ----
    @staticmethod
    def rotacion_en(grados, cx, cy):
        return (Mat3.traslacion(cx, cy) @ Mat3.rotacion(grados) @ Mat3.traslacion(-cx, -cy))

    @staticmethod
    def escalamiento_en(sx, sy, cx, cy):
        return (Mat3.traslacion(cx, cy) @ Mat3.escalamiento(sx, sy) @ Mat3.traslacion(-cx, -cy))

    # ---- operaciones ----
    def __matmul__(self, otro):
        if isinstance(otro, Mat3):
            return Mat3(self.m @ otro.m)
        if isinstance(otro, Vec2):
            return self.aplicar(otro)
        return NotImplemented

    def aplicar(self, v):
        r = self.m @ np.array([v.x, v.y, 1.0])
        return Vec2(r[0], r[1])

    def aplicar_puntos(self, pts):
        """Aplica la matriz a un arreglo (N, 2) de puntos de una sola vez."""
        pts = np.asarray(pts, dtype=float)
        return pts @ self.m[:2, :2].T + self.m[:2, 2]

    def inversa(self):
        return Mat3(np.linalg.inv(self.m))

    def __repr__(self):
        return f"Mat3(\n{np.array2string(self.m, precision=4, suppress_small=True)})"


class Mat4:
    """Matriz 4×4 en coordenadas homogéneas para el espacio."""

    def __init__(self, m=None):
        self.m = np.eye(4) if m is None else np.array(m, dtype=float).reshape(4, 4)

    @staticmethod
    def identidad():
        return Mat4()

    @staticmethod
    def traslacion(tx, ty, tz):
        m = np.eye(4)
        m[0, 3], m[1, 3], m[2, 3] = tx, ty, tz
        return Mat4(m)

    @staticmethod
    def escalamiento(sx, sy=None, sz=None):
        sy = sx if sy is None else sy
        sz = sx if sz is None else sz
        return Mat4(np.diag([sx, sy, sz, 1.0]))

    @staticmethod
    def rotacion_x(grados):
        c, s = _cs(grados)
        return Mat4([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    @staticmethod
    def rotacion_y(grados):
        """Giro sobre el eje vertical: es el que se usa para girar muebles sobre el piso."""
        c, s = _cs(grados)
        return Mat4([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    @staticmethod
    def rotacion_z(grados):
        c, s = _cs(grados)
        return Mat4([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def __matmul__(self, otro):
        if isinstance(otro, Mat4):
            return Mat4(self.m @ otro.m)
        if isinstance(otro, Vec3):
            return self.aplicar(otro)
        return NotImplemented

    def aplicar(self, v):
        r = self.m @ np.array([v.x, v.y, v.z, 1.0])
        return Vec3(r[0], r[1], r[2])

    def aplicar_puntos(self, pts):
        """Aplica la matriz a un arreglo (N, 3) de vértices (sin división por w)."""
        pts = np.asarray(pts, dtype=float)
        return pts @ self.m[:3, :3].T + self.m[:3, 3]

    def inversa(self):
        return Mat4(np.linalg.inv(self.m))

    def __repr__(self):
        return f"Mat4(\n{np.array2string(self.m, precision=4, suppress_small=True)})"
