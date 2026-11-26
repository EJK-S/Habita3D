"""Pruebas de la Escena: objetos huecos por revolución y apoyo de objetos sobre muebles."""
import numpy as np
from plano import Plano, MuebleEnPlanta
from modelado import Bezier, revolucion, construir_mueble_3d
from escena import Escena


def _altura_minima(escena, nombre):
    malla = [m for m in escena.mallas if m.nombre == nombre][0]
    return float(malla.vertices[:, 1].min())


def test_jarron_hueco_se_ve_por_dentro_desde_arriba():
    # Con perfil cerrado (grosor), mirando por la boca se ve el interior y el fondo;
    # la superficie abierta de una sola cara deja ver el fondo de la escena (cielo).
    curva = Bezier([(0.0, 0.0), (0.14, 0.02), (0.20, 0.18), (0.08, 0.34), (0.13, 0.46)])
    centros = {}
    for etiqueta, grosor in (("abierto", None), ("hueco", 0.012)):
        e = Escena(Plano())
        e.camara.mirar_a((0.0, 1.2, 0.05), (0.0, 0.0, 0.0))
        e.agregar_malla_extra(revolucion(curva, 24, 20, grosor=grosor))
        fb = e.renderizar_3d(120, 120)
        centros[etiqueta] = fb.leer_pixel(60, 60)
    assert centros["abierto"] == e.color_cielo
    assert centros["hueco"] != e.color_cielo


def test_perfil_con_grosor_tiene_cara_interior_hacia_el_eje():
    curva = Bezier([(0.0, 0.0), (0.14, 0.02), (0.20, 0.18), (0.08, 0.34), (0.13, 0.46)])
    m = revolucion(curva, 16, 10, grosor=0.012)
    centros = m.vertices[m.caras].mean(axis=1)
    radial = centros.copy()
    radial[:, 1] = 0.0
    hacia_eje = np.einsum("ij,ij->i", m.normales_cara, radial) < 0
    assert hacia_eje.sum() > len(m.caras) // 4            # existe la pared interior
    assert np.isclose(m.vertices[:, 1].min(), 0.0)         # sigue apoyado en y = 0


def test_jarron_sobre_la_mesa_se_apoya_en_su_superficie():
    pl = Plano()
    pl.agregar_mueble(MuebleEnPlanta("mesa", 1.1, 1.1, 1.6, 1.8, 20))
    pl.agregar_mueble(MuebleEnPlanta("jarron", 0.4, 0.4, 1.6, 1.8))      # encima de la mesa
    pl.agregar_mueble(MuebleEnPlanta("planta", 0.45, 0.45, 4.0, 4.0))    # en el piso
    e = Escena(pl)
    assert np.isclose(_altura_minima(e, "jarron"), 0.75)
    assert np.isclose(_altura_minima(e, "planta"), 0.0)
    assert np.isclose(_altura_minima(e, "mesa"), 0.0)


def test_al_mover_el_jarron_fuera_de_la_mesa_vuelve_al_piso():
    pl = Plano()
    pl.agregar_mueble(MuebleEnPlanta("velador", 0.5, 0.5, 1.0, 1.0))
    pl.agregar_mueble(MuebleEnPlanta("lampara", 0.45, 0.45, 1.0, 1.0))
    e = Escena(pl)
    assert np.isclose(_altura_minima(e, "lampara"), 0.55)
    pl.muebles[1].x = 3.0
    e.reconstruir_3d()
    assert np.isclose(_altura_minima(e, "lampara"), 0.0)


def test_cualquier_mueble_sirve_de_apoyo_a_su_altura_real():
    # Sofá: asiento a 0.40 m. Cama: colchón a 0.55 m (la altura se mide sobre la malla 3D)
    pl = Plano()
    pl.agregar_mueble(MuebleEnPlanta("sofá", 2.0, 0.9, 1.0, 1.0))
    pl.agregar_mueble(MuebleEnPlanta("cama", 1.5, 2.0, 4.0, 1.5))
    pl.agregar_mueble(MuebleEnPlanta("jarron", 0.4, 0.4, 1.0, 1.0))
    pl.agregar_mueble(MuebleEnPlanta("planta", 0.45, 0.45, 4.0, 1.5))
    e = Escena(pl)
    assert np.isclose(_altura_minima(e, "jarron"), 0.40)
    assert np.isclose(_altura_minima(e, "planta"), 0.55)


def test_objeto_en_el_borde_no_atraviesa_el_mueble():
    # El centro del jarrón queda 5 cm fuera de la cama, pero el jarrón la pisa: se apoya encima
    pl = Plano()
    pl.agregar_mueble(MuebleEnPlanta("cama", 1.5, 2.0, 4.0, 1.5))         # x de 3.25 a 4.75
    pl.agregar_mueble(MuebleEnPlanta("jarron", 0.4, 0.4, 3.22, 1.5))
    assert np.isclose(_altura_minima(Escena(pl), "jarron"), 0.55)


def test_mueble_escalado_tiene_la_misma_planta_en_2d_y_3d():
    pl = Plano()
    pl.agregar_mueble(MuebleEnPlanta("mesa", 1.0, 0.8, 2.0, 1.0, angulo=30, escala_x=1.5, escala_y=0.5))
    e = Escena(pl)
    malla = [m for m in e.mallas if m.nombre == "mesa"][0]
    poligono = pl.muebles[0].poligono()
    # Extensión en x y en y del plano (en 3D, y del plano = −Z)
    assert np.isclose(np.ptp(malla.vertices[:, 0]), np.ptp(poligono[:, 0]), atol=1e-9)
    assert np.isclose(np.ptp(-malla.vertices[:, 2]), np.ptp(poligono[:, 1]), atol=1e-9)
    assert np.isclose(malla.vertices[:, 1].max(), 0.75)   # la altura no se escala


def test_escala_proporcional_respeta_limites():
    from ui.app import escala_proporcional, ESCALA_MIN, ESCALA_MAX
    assert escala_proporcional(1.0, 2.0, 1.5) == (1.5, 3.0)
    assert escala_proporcional(1.0, 2.0, 10.0) == (2.0, ESCALA_MAX)        # el eje mayor topa
    assert escala_proporcional(0.5, 1.0, 0.1) == (ESCALA_MIN, 0.5)         # el eje menor topa


def test_render_por_lotes_es_identico_al_triangulo_a_triangulo():
    # Mismo algoritmo vectorizado: la imagen debe coincidir píxel por píxel, incluso con
    # texturas, Gouraud/plano y triángulos recortados por el plano cercano.
    pl = Plano()
    s = pl.agregar_habitacion(0, 0, 4, 3)
    pl.agregar_abertura(s[0], 0.5, 0.9, "puerta")
    pl.agregar_mueble(MuebleEnPlanta("mesa", 1.0, 0.8, 2.0, 1.5, angulo=25))
    pl.agregar_mueble(MuebleEnPlanta("jarron", 0.4, 0.4, 2.0, 1.5))
    pl.agregar_mueble(MuebleEnPlanta("planta", 0.45, 0.45, 0.6, 2.4))
    e = Escena(pl)
    camaras = [((0.5, 1.6, -0.4), (3.0, 0.8, -2.6)),      # vista general
               ((2.0, 1.4, -0.1), (2.0, 1.2, 2.0))]       # pegada al muro: recorte cercano
    for ojo, objetivo in camaras:
        e.camara.mirar_a(ojo, objetivo)
        for sombreado in ("gouraud", "plano"):
            e.sombreado = sombreado
            e.por_lotes = False
            ref = e.renderizar_3d(160, 120).datos.copy()
            e.por_lotes = True
            assert np.array_equal(e.renderizar_3d(160, 120).datos, ref)


def test_proyectar_punto_lleva_el_objetivo_al_centro():
    e = Escena(Plano())
    e.camara.mirar_a((1.0, 1.5, 2.0), (1.0, 1.0, -1.0))
    x, y = e.proyectar_punto((1.0, 1.0, -1.0), 101, 81)
    assert np.isclose(x, 50.0) and np.isclose(y, 40.0)
    assert e.proyectar_punto((1.0, 1.5, 5.0), 101, 81) is None          # detrás de la cámara


def test_intensidad_y_ambiente_de_la_luz_cambian_el_brillo():
    pl = Plano()
    pl.agregar_habitacion(0, 0, 4, 3)
    e = Escena(pl)
    e.camara.mirar_a((2.0, 1.5, -0.3), (2.0, 1.0, -3.0))
    brillo = lambda: e.renderizar_3d(80, 60).datos.astype(float).mean()
    base = brillo()
    e.luz.il = 0.2
    assert brillo() < base                                              # menos intensidad: más oscuro
    e.luz.il, e.luz.ia = 0.85, 0.05
    assert brillo() < base                                              # menos luz ambiente: más oscuro
    e.luz.ia = 0.32
    e.luz.posicion.y = 0.3                                              # luz casi en el piso
    assert not np.isclose(brillo(), base)                               # la posición también cuenta


def test_altura_en_rayo_vertical_sobre_caja():
    from modelado import crear_prisma
    caja = crear_prisma(0.0, 0.5, 0.0, 1.0, 1.0, 1.0)                      # y de 0 a 1
    assert np.isclose(caja.altura_en(0.2, -0.3), 1.0)
    assert caja.altura_en(2.0, 0.0) is None


def test_camara_no_atraviesa_muros_pero_cruza_puertas():
    pl = Plano()
    s = pl.agregar_habitacion(0, 0, 4, 3)
    pl.agregar_abertura(s[0], 0.5, 1.0, "puerta")      # puerta sur en x = 2
    e = Escena(pl)
    e.camara.mirar_a((1.0, 1.6, -1.0), (1.0, 1.6, 3.0))     # mira hacia el muro sur (y = 0)
    e.mover_camara(adelante=5.0)
    assert -e.camara.eye.z >= 0.075 + e.RADIO_CAMARA - 1e-6     # se detuvo antes del muro
    e.camara.mirar_a((2.0, 1.6, -1.0), (2.0, 1.6, 3.0))     # ahora de frente a la puerta
    e.mover_camara(adelante=2.0)
    assert -e.camara.eye.z < 0.0                                # salió por la puerta
    e.reiniciar_camara()
    assert (e.camara.eye.x, e.camara.eye.y, e.camara.eye.z) == Escena.CAMARA_INICIAL[0]


def test_escena_usa_el_perfil_del_jarron_bezier():
    from plano import MuebleEnPlanta
    pl = Plano()
    pl.agregar_mueble(MuebleEnPlanta("jarrón Bézier", 0.8, 0.8, 2.0, 2.0, 0, (110, 150, 190),
                                     perfil=((0.0, 0.0), (0.40, 0.05), (0.40, 0.30), (0.10, 0.50))))
    e = Escena(pl)
    v = e.mallas[-1].vertices
    assert v[:, 0].max() - v[:, 0].min() > 0.5        # diámetro ~0.6 m, no los ~0.28 del jarrón base


def test_escena_viste_los_vanos():
    pl = Plano()
    s = pl.agregar_habitacion(0, 0, 4, 3)
    pl.agregar_abertura(s[0], 0.5, 1.0, "puerta")
    pl.agregar_abertura(s[2], 0.5, 1.2, "ventana")
    nombres = [m.nombre for m in Escena(pl).mallas]
    assert "hoja_puerta" in nombres and "vidrio" in nombres
