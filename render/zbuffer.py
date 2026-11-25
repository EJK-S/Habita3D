"""Buffer de profundidad (z-buffer) propio para visibilidad de superficies 3D."""
import numpy as np


class ZBuffer:
    """Almacena la profundidad mínima (más cercana a la cámara) por píxel en NDC [-1, 1]."""

    def __init__(self, ancho: int, alto: int):
        self.ancho = int(ancho)
        self.alto = int(alto)
        self.profundidad = np.full((self.alto, self.ancho), np.inf, dtype=np.float64)

    def limpiar(self) -> None:
        self.profundidad.fill(np.inf)

    def probar_y_actualizar(self, x: int, y: int, z: float) -> bool:
        if 0 <= x < self.ancho and 0 <= y < self.alto:
            if z < self.profundidad[y, x]:
                self.profundidad[y, x] = z
                return True
        return False
