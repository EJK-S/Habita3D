"""Vectores 2D y 3D propios de Habita3D (sin depender de librerías externas de gráficos)."""
import math


class Vec2:
    """Punto o vector en el plano. Se usa para el editor 2D y el plano de la vivienda."""
    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, o):
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Vec2(self.x - o.x, self.y - o.y)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __mul__(self, k):
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __truediv__(self, k):
        return Vec2(self.x / k, self.y / k)

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return f"Vec2({self.x:.4g}, {self.y:.4g})"

    def punto(self, o):
        return self.x * o.x + self.y * o.y

    def norma(self):
        return math.hypot(self.x, self.y)

    def normalizado(self):
        n = self.norma()
        return Vec2(self.x / n, self.y / n) if n > 0 else Vec2(0, 0)

    def perpendicular(self):
        """Vector girado 90° en sentido antihorario."""
        return Vec2(-self.y, self.x)

    def casi_igual(self, o, tol=1e-9):
        return abs(self.x - o.x) <= tol and abs(self.y - o.y) <= tol


class Vec3:
    """Punto o vector en el espacio, base de mallas, cámara e iluminación."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, o):
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, k):
        return Vec3(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def __truediv__(self, k):
        return Vec3(self.x / k, self.y / k, self.z / k)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __repr__(self):
        return f"Vec3({self.x:.4g}, {self.y:.4g}, {self.z:.4g})"

    def punto(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cruz(self, o):
        return Vec3(self.y * o.z - self.z * o.y,
                    self.z * o.x - self.x * o.z,
                    self.x * o.y - self.y * o.x)

    def norma(self):
        return math.sqrt(self.punto(self))

    def normalizado(self):
        n = self.norma()
        return Vec3(self.x / n, self.y / n, self.z / n) if n > 0 else Vec3(0, 0, 0)

    def casi_igual(self, o, tol=1e-9):
        return (abs(self.x - o.x) <= tol and abs(self.y - o.y) <= tol
                and abs(self.z - o.z) <= tol)
