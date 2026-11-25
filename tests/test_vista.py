"""Pruebas unitarias del paquete vista (cámara, proyecciones y volumen de visión)."""
import math
import numpy as np
from mathlib import Vec3
from vista import Camara, VolumenVision, proy_ortografica, proy_perspectiva


def test_camara_matriz_vista_ortonormal():
    cam = Camara(eye=(0.0, 2.0, 5.0), target=(0.0, 2.0, 0.0), up=(0.0, 1.0, 0.0))
    V = cam.matriz_vista()
    # El punto target está a 5 unidades frente a la cámara -> en espacio cámara es (0, 0, -5)
    p_cam = V.aplicar(Vec3(0.0, 2.0, 0.0))
    assert p_cam.casi_igual(Vec3(0.0, 0.0, -5.0), tol=1e-6)


def test_proyeccion_ortografica_preserva_escala():
    P = proy_ortografica(-5.0, 5.0, -5.0, 5.0, cercano=0.1, lejano=50.0)
    # Dos puntos con mismo X pero distinta profundidad Z deben proyectar al mismo X en NDC
    p_cerca = P.m @ np.array([2.5, 0.0, -2.0, 1.0])
    p_lejos = P.m @ np.array([2.5, 0.0, -20.0, 1.0])
    assert math.isclose(p_cerca[0] / p_cerca[3], p_lejos[0] / p_lejos[3], rel_tol=1e-7)
    assert math.isclose(p_cerca[0] / p_cerca[3], 0.5, rel_tol=1e-7)


def test_proyeccion_perspectiva_achica_con_profundidad():
    P = proy_perspectiva(fov_grados=60.0, aspecto=1.0, cercano=0.1, lejano=50.0)
    c_cerca = P.m @ np.array([1.0, 0.0, -2.0, 1.0])
    c_lejos = P.m @ np.array([1.0, 0.0, -4.0, 1.0])
    x_ndc_cerca = c_cerca[0] / c_cerca[3]
    x_ndc_lejos = c_lejos[0] / c_lejos[3]
    assert math.isclose(x_ndc_cerca, 2.0 * x_ndc_lejos, rel_tol=1e-7)


def test_recorte_plano_cercano():
    vol = VolumenVision(cercano=1.0, lejano=20.0)
    P = proy_perspectiva(60.0, 1.0, 1.0, 20.0)
    # Triángulo que cruza z = -1.0 (un vértice detrás en z = 0.5)
    tri = np.array([[0.0, 0.0, 0.5], [-1.0, 0.0, -2.0], [1.0, 0.0, -2.0]])
    uvs = np.array([[0.5, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cols = np.ones((3, 3))
    res = vol.recortar_y_proyectar_triangulo(tri, uvs, cols, P, 200, 200)
    assert len(res) == 2  # un vértice fuera -> genera 2 triángulos recortados
