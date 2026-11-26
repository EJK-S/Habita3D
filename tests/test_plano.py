import numpy as np
from plano import Plano, Muro, MuebleEnPlanta, Figura
from mathlib import Mat3
from raster2d import puntos_dda, puntos_bresenham
from plano.plano import punto_en_poligono
from raster2d import Framebuffer


def _plano_ejemplo():
    pl = Plano()
    s, e, n, o = pl.agregar_habitacion(0, 0, 4, 3)
    pl.agregar_abertura(s, 0.5, 0.9, "puerta")      # muro 0 (sur)
    pl.agregar_abertura(n, 0.5, 1.2, "ventana")     # muro 2 (norte)
    pl.agregar_mueble(MuebleEnPlanta("mesa", 1.0, 1.0, 2, 1.5))
    return pl


def test_punto_en_poligono_concavo():
    L = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)]     # polígono en forma de L
    assert punto_en_poligono(0.5, 2.5, L)
    assert punto_en_poligono(3.5, 0.5, L)
    assert not punto_en_poligono(2.5, 2.0, L)                # hueco de la L
    assert not punto_en_poligono(5.0, 0.5, L)


def test_elemento_en_prioriza_mueble_abertura_y_muro():
    pl = _plano_ejemplo()
    assert pl.elemento_en(2.0, 1.5) == ("mueble", 0)
    assert pl.elemento_en(2.0, 0.0) == ("abertura", 0)       # centro de la puerta del muro sur
    assert pl.elemento_en(0.5, 0.0) == ("muro", 0)
    assert pl.elemento_en(0.5, 0.12) == ("muro", 0)          # junto al muro, dentro del margen
    assert pl.elemento_en(1.0, 2.0) is None                  # piso vacío


def test_mueble_en_devuelve_el_de_arriba_y_respeta_la_rotacion():
    pl = _plano_ejemplo()
    pl.agregar_mueble(MuebleEnPlanta("jarron", 0.4, 0.4, 2, 1.5))   # encima de la mesa
    assert pl.mueble_en(2.0, 1.5) == 1                               # el jarrón, no la mesa
    assert pl.mueble_en(2.4, 1.5) == 0                               # borde de la mesa
    cama = MuebleEnPlanta("cama", 0.5, 2.0, 3.0, 1.5, angulo=90)     # girada: ahora es horizontal
    pl.agregar_mueble(cama)
    assert pl.mueble_en(3.9, 1.5) == 2
    assert pl.mueble_en(3.0, 2.3) is None


def test_escalamiento_en_ejes_locales_compuesto_con_rotacion():
    # M = T · R · S: primero se escala en los ejes del mueble, luego se gira 90° y se traslada
    m = MuebleEnPlanta("mesa", ancho=1.0, largo=0.5, x=2, y=1, angulo=90, escala_x=2.0, escala_y=1.5)
    p = m.poligono()
    assert np.isclose(np.ptp(p[:, 0]), 0.75)       # largo 0.5 × 1.5, ahora a lo largo de x
    assert np.isclose(np.ptp(p[:, 1]), 2.0)        # ancho 1.0 × 2.0, ahora a lo largo de y
    assert np.allclose(p.mean(axis=0), (2, 1))     # escala respecto a su propio centro


def test_posicion_en_muro_proyecta_y_respeta_las_esquinas():
    pl = Plano()
    i = pl.agregar_muro((0, 0), (4, 0), 0.2)
    assert np.isclose(pl.posicion_en_muro(i, 1.0, 0.3, 0.9), 0.25)     # se proyecta sobre el muro
    # Cerca del extremo se limita: medio vano + medio espesor + 5 cm desde la esquina
    assert np.isclose(pl.posicion_en_muro(i, 0.0, 0.0, 0.9) * 4, 0.45 + 0.10 + 0.05)
    assert np.isclose(pl.posicion_en_muro(i, 9.0, 0.0, 0.9) * 4, 4 - 0.60)
    assert pl.posicion_en_muro(pl.agregar_muro((0, 0), (0.8, 0)), 0.4, 0, 0.9) is None   # no cabe


def test_abertura_se_superpone_solo_en_el_mismo_muro():
    pl = Plano()
    s, e, n, o = pl.agregar_habitacion(0, 0, 4, 3)
    pl.agregar_abertura(s, 0.5, 1.0, "puerta")                         # ocupa x de 1.5 a 2.5
    assert pl.abertura_se_superpone(s, 0.5 + 0.8 / 4, 0.9)             # centro a 0.8 m: choca
    assert not pl.abertura_se_superpone(s, 0.5 + 1.2 / 4, 0.9)         # centro a 1.2 m: libre
    assert not pl.abertura_se_superpone(n, 0.5, 1.0)                   # otro muro


def test_dos_ventanas_pueden_quedar_juntas():
    pl = Plano()
    i = pl.agregar_muro((0, 0), (6, 0), 0.15)
    pl.agregar_abertura(i, 2.0 / 6, 1.2, "ventana")                     # ocupa x de 1.4 a 2.6
    assert not pl.abertura_se_superpone(i, 3.2 / 6, 1.2)               # de 2.6 a 3.8: solo se tocan
    # Imán: con el cursor a 3.3 m (deja 10 cm de hueco) o a 3.0 m (se pisa) queda pegada en 3.2 m
    assert np.isclose(pl.ajustar_junto_a_vecinos(i, 3.3 / 6, 1.2) * 6, 3.2)
    assert np.isclose(pl.ajustar_junto_a_vecinos(i, 3.0 / 6, 1.2) * 6, 3.2)
    assert np.isclose(pl.ajustar_junto_a_vecinos(i, 1.0 / 6, 1.2) * 6, 0.8)   # se pega por la izquierda
    # Lejos del vecino no se mueve
    assert np.isclose(pl.ajustar_junto_a_vecinos(i, 4.5 / 6, 1.2) * 6, 4.5)


def test_iman_no_salta_lejos_del_cursor():
    pl = Plano()
    i = pl.agregar_muro((0, 0), (6, 0), 0.15)
    pl.agregar_abertura(i, 2.0 / 6, 1.2, "ventana")                     # de 1.4 a 2.6
    pl.agregar_abertura(i, 3.2 / 6, 1.2, "ventana")                     # de 2.6 a 3.8
    # Cursor en la unión de las dos: el lugar libre más cercano está a 1.8 m (demasiado lejos)
    assert pl.ajustar_junto_a_vecinos(i, 2.6 / 6, 1.2) is None


def test_iman_no_encuentra_lugar_si_no_cabe():
    pl = Plano()
    i = pl.agregar_muro((0, 0), (3, 0), 0.15)
    pl.agregar_abertura(i, 1.5 / 3, 2.0, "ventana")                     # ocupa x de 0.5 a 2.5
    assert pl.ajustar_junto_a_vecinos(i, 1.5 / 3, 1.0) is None          # a ningún lado cabe 1 m


def test_figura_linea_usa_el_algoritmo_elegido():
    # Con escala 1 px/m (eje y hacia abajo) los vértices de la línea son píxeles directamente
    a_pant = Mat3.traslacion(0, 40) @ Mat3.escalamiento(1, -1)
    for algoritmo, puntos in (("dda", puntos_dda), ("bresenham", puntos_bresenham)):
        fb = Framebuffer(60, 50, fondo=(255, 255, 255))
        Figura("linea", [(2, 28), (38, 37)], color=(0, 0, 0), grosor=1, algoritmo=algoritmo).dibujar(fb, a_pant)
        negros = {(x, y) for y in range(50) for x in range(60) if fb.leer_pixel(x, y) == (0, 0, 0)}
        assert negros == set(puntos(2, 12, 38, 3))


def test_figuras_con_relleno_y_borde():
    a_pant = Mat3.traslacion(0, 100) @ Mat3.escalamiento(10, -10)      # 10 px por metro
    fb = Framebuffer(120, 110, fondo=(255, 255, 255))
    Figura("circulo", [(3, 3)], color=(0, 0, 0), relleno=(200, 40, 40), grosor=2, radio=2).dibujar(fb, a_pant)
    assert fb.leer_pixel(30, 70) == (200, 40, 40)                       # centro relleno
    assert fb.leer_pixel(30, 50) == (0, 0, 0)                           # borde (radio 20 px)
    Figura("poligono", [(7, 1), (11, 1), (9, 5)], color=(0, 0, 0), relleno=(40, 110, 200)).dibujar(fb, a_pant)
    assert fb.leer_pixel(90, 80) == (40, 110, 200)                     # dentro del triángulo


def test_figura_se_selecciona_y_se_borra_sin_tocar_el_3d():
    pl = _plano_ejemplo()
    pl.agregar_figura(Figura("linea", [(0.5, 0.5), (3.5, 0.8)]))
    pl.agregar_figura(Figura("circulo", [(2.0, 1.5)], radio=0.3))
    assert pl.elemento_en(2.0, 1.5) == ("figura", 1)                    # encima de la mesa: la figura
    assert pl.elemento_en(1.25, 0.6) == ("figura", 0)                   # sobre la línea (y = 0.575)
    pl.eliminar("figura", 1)
    assert pl.elemento_en(2.0, 1.5) == ("mueble", 0)
    from escena import Escena
    assert [m.nombre for m in Escena(pl).mallas].count("mesa") == 1     # las figuras no generan 3D


def test_eliminar_muro_borra_sus_aberturas_y_reindexa():
    pl = _plano_ejemplo()
    pl.eliminar("muro", 0)                                   # se va el muro sur y su puerta
    assert len(pl.muros) == 3
    assert len(pl.aberturas) == 1
    ventana = pl.aberturas[0]
    assert ventana.tipo == "ventana" and ventana.muro == 1   # el muro norte pasó del índice 2 al 1
    assert pl.muros[ventana.muro].p1 == (4, 3)


def test_eliminar_mueble_y_abertura():
    pl = _plano_ejemplo()
    pl.eliminar("mueble", 0)
    pl.eliminar("abertura", 0)
    assert pl.muebles == [] and len(pl.aberturas) == 1 and pl.aberturas[0].tipo == "ventana"


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
