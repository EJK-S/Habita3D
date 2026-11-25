import math
import random
import numpy as np
from raster2d import (Framebuffer, puntos_dda, puntos_bresenham, linea_wu, linea,
                      puntos_circulo, circulo, rellenar_poligono)


def _recta_ideal_error(puntos, p1, p2):
    """Distancia máxima (en píxeles) de los puntos a la recta ideal, medida sobre el eje menor."""
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    peor = 0.0
    for x, y in puntos:
        if abs(dx) >= abs(dy):
            y_ideal = y1 + dy * (x - x1) / dx if dx else y1
            peor = max(peor, abs(y - y_ideal))
        else:
            x_ideal = x1 + dx * (y - y1) / dy
            peor = max(peor, abs(x - x_ideal))
    return peor


def test_bresenham_incluye_extremos_y_cantidad_de_pixeles():
    for p1, p2 in [((0, 0), (10, 3)), ((5, 5), (5, 20)), ((9, 2), (1, 12)), ((3, 3), (3, 3))]:
        pts = puntos_bresenham(*p1, *p2)
        assert pts[0] == p1 and pts[-1] == p2
        assert len(pts) == max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) + 1


def test_bresenham_esta_a_media_unidad_de_la_recta_ideal_en_los_8_octantes():
    random.seed(1)
    for _ in range(300):
        p1 = (random.randint(0, 80), random.randint(0, 80))
        p2 = (random.randint(0, 80), random.randint(0, 80))
        if p1 == p2:
            continue
        assert _recta_ideal_error(puntos_bresenham(*p1, *p2), p1, p2) <= 0.5 + 1e-9


def test_dda_y_bresenham_difieren_como_mucho_un_pixel():
    random.seed(2)
    for _ in range(200):
        p1 = (random.randint(0, 80), random.randint(0, 80))
        p2 = (random.randint(0, 80), random.randint(0, 80))
        a, b = puntos_dda(*p1, *p2), puntos_bresenham(*p1, *p2)
        assert len(a) == len(b)
        assert _recta_ideal_error(a, p1, p2) <= 0.5 + 1e-9


def test_lineas_horizontal_vertical_y_diagonal_exactas():
    assert puntos_bresenham(2, 5, 6, 5) == [(2, 5), (3, 5), (4, 5), (5, 5), (6, 5)]
    assert puntos_dda(4, 1, 4, 3) == [(4, 1), (4, 2), (4, 3)]
    assert puntos_bresenham(0, 0, 3, 3) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_wu_horizontal_en_entero_no_mezcla_vecinos():
    fb = Framebuffer(20, 10)
    linea_wu(fb, 2, 5, 15, 5, (0, 0, 0))
    assert fb.leer_pixel(8, 5) == (0, 0, 0)
    assert fb.leer_pixel(8, 4) == (255, 255, 255) and fb.leer_pixel(8, 6) == (255, 255, 255)


def test_wu_reparte_intensidad_entre_dos_pixeles():
    fb = Framebuffer(40, 20)
    linea_wu(fb, 2, 2, 35, 14, (0, 0, 0))
    oscuridad = 255 - fb.datos[:, :, 0].astype(float)
    # en una columna interior la suma de intensidades debe valer ~1 (255)
    for x in (10, 20, 30):
        assert abs(oscuridad[:, x].sum() - 255) <= 2
    # y hay píxeles de gris intermedio (eso es el antialiasing)
    intermedios = ((fb.datos[:, :, 0] > 10) & (fb.datos[:, :, 0] < 245)).sum()
    assert intermedios > 20


def test_circulo_cumple_la_ecuacion_con_error_menor_a_medio_pixel():
    for r in (1, 5, 17, 40):
        pts = puntos_circulo(0, 0, r)
        assert all(abs(math.hypot(x, y) - r) <= 0.5 + 1e-9 for x, y in pts)
        assert set(pts) == {(-x, y) for x, y in pts} == {(x, -y) for x, y in pts}


def test_circulo_relleno_pinta_el_centro_y_no_sale():
    fb = Framebuffer(40, 40)
    circulo(fb, 20, 20, 10, (0, 0, 0), relleno=(255, 0, 0))
    assert fb.leer_pixel(20, 20) == (255, 0, 0)
    assert fb.leer_pixel(20, 5) == (255, 255, 255)


def test_relleno_de_rectangulo_con_borde_cubre_todo_el_rectangulo():
    fb = Framebuffer(30, 20)
    rellenar_poligono(fb, [(2, 2), (12, 2), (12, 7), (2, 7)], (255, 0, 0), (0, 0, 0))
    zona = fb.datos[2:8, 2:13]
    assert not (zona == 255).all(axis=2).any()          # no queda ningún píxel blanco adentro
    assert fb.leer_pixel(13, 4) == (255, 255, 255)      # y nada se sale por fuera


def test_relleno_regla_par_impar_en_poligono_concavo():
    fb = Framebuffer(30, 30)
    # forma de "U": el hueco central no debe pintarse
    u = [(2, 2), (8, 2), (8, 18), (20, 18), (20, 2), (26, 2), (26, 26), (2, 26)]
    rellenar_poligono(fb, u, (0, 255, 0))
    assert fb.leer_pixel(14, 10) == (255, 255, 255)     # dentro del hueco de la U
    assert fb.leer_pixel(14, 22) == (0, 255, 0)         # en la base de la U


def test_triangulo_area_aproximada():
    fb = Framebuffer(60, 60)
    rellenar_poligono(fb, [(5, 5), (45, 5), (5, 45)], (1, 2, 3))
    pintados = (fb.datos[:, :, 0] == 1).sum()
    # área ideal = 40·40/2 = 800; al contar los píxeles del borde salen ~860 (error < 10 %)
    assert abs(pintados - 800) < 80


def test_grosor_ensancha_la_linea():
    fb1, fb3 = Framebuffer(30, 30), Framebuffer(30, 30)
    linea(fb1, 3, 15, 26, 15, (0, 0, 0), 1)
    linea(fb3, 3, 15, 26, 15, (0, 0, 0), 5)
    assert (fb3.datos[:, :, 0] == 0).sum() > 4 * (fb1.datos[:, :, 0] == 0).sum()


def test_primitivas_fuera_de_pantalla_no_fallan():
    fb = Framebuffer(20, 20)
    linea(fb, -50, -50, 100, 100, (0, 0, 0), 3)
    linea(fb, -10, 5, 40, 8, (0, 0, 0), algoritmo="wu")
    circulo(fb, -5, -5, 30, (0, 0, 0), relleno=(1, 1, 1))
    rellenar_poligono(fb, [(-10, -10), (50, 0), (10, 60)], (9, 9, 9))
