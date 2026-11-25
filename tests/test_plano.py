import numpy as np
from plano import Plano, Muro, MuebleEnPlanta
from raster2d import Framebuffer


def test_muro_horizontal_tiene_el_espesor_y_largo_esperados():
    p = Muro((0, 0), (4, 0), 0.2).poligono()
    assert abs(p[:, 1].max() - p[:, 1].min() - 0.2) < 1e-12
    assert abs(p[:, 0].max() - p[:, 0].min() - 4.2) < 1e-12      # +medio espesor por cada extremo


def test_mueble_rotado_90_intercambia_ancho_y_largo():
    m = MuebleEnPlanta("cama", ancho=1.4, largo=2.0, x=3, y=2, angulo=90)
    p = m.poligono()
    assert abs(np.ptp(p[:, 0]) - 2.0) < 1e-9 and abs(np.ptp(p[:, 1]) - 1.4) < 1e-9
    assert np.allclose(p.mean(axis=0), (3, 2))                   # gira sobre su propio centro


def test_dibujar_plano_pinta_muros_y_muebles():
    pl = Plano()
    s, e, n, o = pl.agregar_habitacion(0, 0, 4, 3)
    pl.agregar_abertura(s, 0.5, 0.9, "puerta")
    pl.agregar_abertura(n, 0.5, 1.2, "ventana")
    pl.agregar_mueble(MuebleEnPlanta("sofá", 1.8, 0.8, 2, 1.5, 0, (120, 140, 200)))
    fb = Framebuffer(320, 240)
    pl.dibujar(fb, escala=60, origen=(30, 210))
    assert fb.leer_pixel(30, 210 - 90) == (70, 70, 80)           # muro oeste (x=0, y=1.5 m)
    assert fb.leer_pixel(30 + 120, 210 - 90) == (120, 140, 200)  # centro del sofá
