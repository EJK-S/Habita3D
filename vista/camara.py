"""Cámara sintética 3D y construcción de la matriz de vista usando Vec3 y Mat4."""
import math
import numpy as np
from mathlib import Vec3, Mat4


class Camara:
    """
    Cámara virtual definida por posición (eye), punto de interés (target) y vector arriba (up).
    Usa sistema de coordenadas de mano derecha (+X derecha, +Y arriba, cámara mira hacia -Z local).
    """

    def __init__(
        self,
        eye: Vec3 | tuple = (3.0, 2.2, 3.5),
        target: Vec3 | tuple = (3.0, 1.0, -2.2),
        up: Vec3 | tuple = (0.0, 1.0, 0.0),
    ):
        self.eye = eye if isinstance(eye, Vec3) else Vec3(*eye)
        self.target = target if isinstance(target, Vec3) else Vec3(*target)
        self.up = up if isinstance(up, Vec3) else Vec3(*up)

        # Calcular yaw y pitch iniciales desde eye -> target para navegación interactiva
        f = (self.target - self.eye).normalizado()
        self.pitch = math.degrees(math.asin(max(-0.999, min(0.999, f.y))))
        self.yaw = math.degrees(math.atan2(f.z, f.x))

    def mirar_a(self, eye: Vec3 | tuple, target: Vec3 | tuple, up: Vec3 | tuple = (0.0, 1.0, 0.0)) -> None:
        self.eye = eye if isinstance(eye, Vec3) else Vec3(*eye)
        self.target = target if isinstance(target, Vec3) else Vec3(*target)
        self.up = up if isinstance(up, Vec3) else Vec3(*up)
        f = (self.target - self.eye).normalizado()
        self.pitch = math.degrees(math.asin(max(-0.999, min(0.999, f.y))))
        self.yaw = math.degrees(math.atan2(f.z, f.x))

    def ejes(self) -> tuple[Vec3, Vec3, Vec3]:
        """Devuelve la base ortonormal (derecha u, arriba v, atrás n) de la cámara."""
        n = (self.eye - self.target).normalizado()
        if n.norma() < 1e-9:
            n = Vec3(0.0, 0.0, 1.0)
        u = self.up.cruz(n).normalizado()
        if u.norma() < 1e-9:
            u = Vec3(1.0, 0.0, 0.0)
        v = n.cruz(u).normalizado()
        return u, v, n

    def matriz_vista(self) -> Mat4:
        """
        Construye la matriz de vista V = R · T en convención de vectores columna (P' = V · P).
        Transforma coordenadas de mundo a coordenadas de cámara.
        """
        u, v, n = self.ejes()
        e = self.eye
        m = np.array([
            [u.x, u.y, u.z, -u.punto(e)],
            [v.x, v.y, v.z, -v.punto(e)],
            [n.x, n.y, n.z, -n.punto(e)],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        return Mat4(m)

    def mover(self, adelante: float = 0.0, derecha: float = 0.0, arriba: float = 0.0) -> None:
        """Desplaza eye y target manteniendo la orientación."""
        u, _, n = self.ejes()
        frente = Vec3(-n.x, 0.0, -n.z).normalizado()
        if frente.norma() < 1e-6:
            frente = -n
        delta = frente * adelante + u * derecha + Vec3(0.0, 1.0, 0.0) * arriba
        self.eye = self.eye + delta
        self.target = self.target + delta

    def rotar_vista(self, delta_yaw: float, delta_pitch: float) -> None:
        """Rota la dirección de mirada (yaw/pitch en grados) manteniendo fija la posición eye."""
        self.yaw += delta_yaw
        self.pitch = max(-85.0, min(85.0, self.pitch + delta_pitch))
        yr = math.radians(self.yaw)
        pr = math.radians(self.pitch)
        dir_f = Vec3(
            math.cos(pr) * math.cos(yr),
            math.sin(pr),
            math.cos(pr) * math.sin(yr),
        ).normalizado()
        dist = max((self.target - self.eye).norma(), 1.0)
        self.target = self.eye + dir_f * dist
