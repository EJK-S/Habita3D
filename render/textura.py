"""Carga de mapas de bits con Pillow y generación de texturas procedurales por defecto."""
import pathlib
import numpy as np
from PIL import Image


class Textura:
    """Mapa de bits RGB (alto, ancho, 3) normalizado en [0, 1] con muestreo repetido (wrap)."""

    def __init__(self, datos: np.ndarray, nombre: str = "textura"):
        arr = np.asarray(datos, dtype=np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0
        self.datos = np.clip(arr, 0.0, 1.0)
        self.alto, self.ancho = self.datos.shape[:2]
        self.nombre = nombre

    @classmethod
    def desde_archivo(cls, ruta: str | pathlib.Path) -> "Textura":
        img = Image.open(ruta).convert("RGB")
        return cls(np.asarray(img, dtype=np.float64) / 255.0, nombre=pathlib.Path(ruta).stem)

    def guardar(self, ruta: str | pathlib.Path) -> None:
        p = pathlib.Path(ruta)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((self.datos * 255.0).astype(np.uint8)).save(p)

    def muestrear_vectorizado(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Muestrea colores RGB (N, 3) para arreglos de coordenadas (u, v) con repetición."""
        u_wrap = np.mod(u, 1.0)
        v_wrap = np.mod(1.0 - v, 1.0)  # origen arriba-izquierda en imagen
        tx = np.minimum((u_wrap * (self.ancho - 1)).astype(np.int32), self.ancho - 1)
        ty = np.minimum((v_wrap * (self.alto - 1)).astype(np.int32), self.alto - 1)
        return self.datos[ty, tx]


def generar_texturas_defecto(carpeta: str | pathlib.Path | None = None) -> dict[str, Textura]:
    """
    Genera texturas procedurales realistas (piso de madera/parquet, pared estucada,
    madera de muebles y cerámica) y opcionalmente las guarda en `assets/texturas/`.
    """
    tam = 128
    yy, xx = np.mgrid[0:tam, 0:tam].astype(np.float64)

    # 1. Piso de parquet / tablones de madera
    tabla_y = (yy // 16).astype(int)
    desfase = (tabla_y % 2) * 32
    junta_y = (yy % 16) < 1.2
    junta_x = ((xx + desfase) % 64) < 1.2
    veta = 0.06 * np.sin(yy * 0.8 + 2.5 * np.sin(xx * 0.08 + tabla_y))
    tono_tabla = ((tabla_y * 37) % 11) / 110.0
    piso = np.zeros((tam, tam, 3), dtype=np.float64)
    piso[..., 0] = 0.74 + veta + tono_tabla
    piso[..., 1] = 0.56 + veta * 0.8 + tono_tabla * 0.8
    piso[..., 2] = 0.36 + veta * 0.6 + tono_tabla * 0.5
    piso[junta_y | junta_x] *= 0.55

    # 2. Pared estucada / ladrillo suave
    ruido = 0.04 * np.sin(xx * 0.35) * np.cos(yy * 0.35) + 0.02 * np.sin((xx + yy) * 0.7)
    pared = np.zeros((tam, tam, 3), dtype=np.float64)
    pared[..., 0] = 0.91 + ruido
    pared[..., 1] = 0.88 + ruido
    pared[..., 2] = 0.82 + ruido

    # 3. Madera vetada para muebles
    veta_m = 0.08 * np.sin(xx * 0.25 + 3.0 * np.sin(yy * 0.06))
    madera = np.zeros((tam, tam, 3), dtype=np.float64)
    madera[..., 0] = 0.62 + veta_m
    madera[..., 1] = 0.42 + veta_m * 0.75
    madera[..., 2] = 0.24 + veta_m * 0.5

    # 4. Cerámica decorativa con bandas geométricas
    banda = 0.08 * np.cos(yy * 2.0 * np.pi / 32.0)
    ceramica = np.zeros((tam, tam, 3), dtype=np.float64)
    ceramica[..., 0] = 0.80 + banda
    ceramica[..., 1] = 0.48 + banda * 0.6
    ceramica[..., 2] = 0.34 + banda * 0.4

    texturas = {
        "piso": Textura(piso, "piso"),
        "pared": Textura(pared, "pared"),
        "madera": Textura(madera, "madera"),
        "ceramica": Textura(ceramica, "ceramica"),
    }

    if carpeta is not None:
        base = pathlib.Path(carpeta)
        for nombre, tex in texturas.items():
            ruta = base / f"{nombre}.png"
            if not ruta.exists():
                tex.guardar(ruta)

    return texturas
