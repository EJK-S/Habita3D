"""Pruebas unitarias del paquete render (ZBuffer, iluminación de Phong y rasterizador 3D)."""
import numpy as np
from raster2d import Framebuffer
from render import ZBuffer, Luz, sombreado_plano, sombreado_gouraud, rasterizar_triangulo


def test_zbuffer_oculta_superficie_trasera():
    fb = Framebuffer(40, 40, fondo=(0, 0, 0))
    zb = ZBuffer(40, 40)
    pts_lejos = np.array([[5.0, 5.0, 0.8], [35.0, 5.0, 0.8], [20.0, 35.0, 0.8]])
    pts_cerca = np.array([[5.0, 5.0, 0.2], [35.0, 5.0, 0.2], [20.0, 35.0, 0.2]])
    uvs = np.zeros((3, 2))
    inv_w = np.ones(3)

    rasterizar_triangulo(fb, zb, pts_cerca, inv_w, uvs, np.tile([1.0, 0.0, 0.0], (3, 1)), usar_textura=False)
    rasterizar_triangulo(fb, zb, pts_lejos, inv_w, uvs, np.tile([0.0, 0.0, 1.0], (3, 1)), usar_textura=False)

    assert tuple(fb.datos[15, 20]) == (255, 0, 0)


def test_iluminacion_phong_intensidad():
    luz = Luz(posicion=(0.0, 5.0, 0.0), intensidad_ambiental=0.2, intensidad_luz=0.8)
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    caras = np.array([[0, 1, 2]])
    normales = np.array([[0.0, 1.0, 0.0]])
    c_plano = sombreado_plano(verts, caras, normales, (200, 200, 200), np.array([0.0, 5.0, 0.0]), luz)
    c_gouraud = sombreado_gouraud(verts, np.tile([0.0, 1.0, 0.0], (3, 1)), (200, 200, 200), np.array([0.0, 5.0, 0.0]), luz)
    assert np.all(c_plano > 0.7)
    # En el vértice (0, 0, 0) directamente bajo la luz, Gouraud captura el pico especular completo
    assert np.all(c_gouraud[0] > 0.95)
