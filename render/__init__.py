from .zbuffer import ZBuffer
from .textura import Textura, generar_texturas_defecto
from .iluminacion import (
    Luz,
    calcular_phong_vectorizado,
    sombreado_plano,
    sombreado_gouraud,
)
from .raster3d import rasterizar_triangulo, dibujar_triangulo_alambre

__all__ = [
    "ZBuffer",
    "Textura",
    "generar_texturas_defecto",
    "Luz",
    "calcular_phong_vectorizado",
    "sombreado_plano",
    "sombreado_gouraud",
    "rasterizar_triangulo",
    "dibujar_triangulo_alambre",
]
