"""Pruebas de la CSG general por árboles BSP: se comprueban volúmenes exactos."""
import numpy as np
from modelado.csg import caja, esfera, cilindro, volumen, HojaCSG, OperacionCSG, SolidoCSG
from modelado import crear_prisma


def _vol(solido):
    return volumen(solido.a_malla())


def test_union_interseccion_y_diferencia_de_dos_cubos():
    # Dos cubos de 2 m desplazados 1 m en x: se superponen en un prisma de 1 × 2 × 2 = 4 m³
    a, b = caja(0, 0, 0, 2, 2, 2), caja(1, 0, 0, 2, 2, 2)
    assert np.isclose(_vol(a), 8.0)
    assert np.isclose(_vol(a | b), 12.0)
    assert np.isclose(_vol(a & b), 4.0)
    assert np.isclose(_vol(a - b), 4.0)
    assert np.isclose(_vol(b - a), 4.0)


def test_cubos_separados_y_contenidos():
    a, lejos = caja(0, 0, 0, 1, 1, 1), caja(5, 0, 0, 1, 1, 1)
    assert np.isclose(_vol(a | lejos), 2.0)
    assert np.isclose(_vol(a & lejos), 0.0)
    assert np.isclose(_vol(a - lejos), 1.0)
    grande, chico = caja(0, 0, 0, 3, 3, 3), caja(0, 0, 0, 1, 1, 1)
    assert np.isclose(_vol(grande - chico), 26.0)                 # cubo hueco
    assert np.isclose(_vol(grande & chico), 1.0)


def test_caja_menos_cilindro_coincide_con_el_prisma_inscrito():
    c = caja(0, 0, 0, 1.6, 1.6, 1.6)
    cil = cilindro(0.4, -1, 0.4, 0.4, 2, sectores=24)            # atraviesa la caja de lado a lado
    area_poligono = 0.5 * 24 * 0.4 ** 2 * np.sin(2 * np.pi / 24)
    assert np.isclose(_vol(c - cil), 1.6 ** 3 - area_poligono * 1.6, atol=1e-6)


def test_interseccion_con_esfera_redondea_las_esquinas():
    c = caja(0, 0, 0, 1.6, 1.6, 1.6)
    e = esfera(0, 0, 0, 1.0, sectores=24, anillos=16)
    r = (c & e).a_malla()
    assert _vol(c & e) < min(_vol(c), _vol(e))                   # más chico que ambos
    assert np.max(np.abs(r.vertices)) <= 0.8 + 1e-9              # dentro de la caja
    assert np.max(np.linalg.norm(r.vertices, axis=1)) <= 1.0 + 1e-9   # y dentro de la esfera


def test_resultado_cerrado_con_normales_hacia_afuera():
    m = (caja(0, 0, 0, 2, 2, 2) | caja(1, 0, 0, 2, 2, 2)).a_malla()
    centros = m.vertices[m.caras].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", m.normales_cara, centros - (0.5, 0, 0)) > 0)
    assert volumen(m) > 0


def test_arbol_csg_se_evalua_y_se_describe():
    arbol = OperacionCSG(
        "diferencia",
        OperacionCSG("union", HojaCSG("A", caja(0, 0, 0, 2, 2, 2)), HojaCSG("B", caja(1, 0, 0, 2, 2, 2))),
        HojaCSG("C", caja(0.5, 0, 0, 1, 4, 1)),                   # atraviesa la unión: quita 1·2·1
    )
    assert arbol.describir() == "((A ∪ B) − C)"
    assert np.isclose(_vol(arbol.evaluar()), 10.0)


def test_estante_del_catalogo_es_la_union_de_sus_tablas():
    from modelado import construir_mueble_3d
    m = construir_mueble_3d("estante", ancho=1.4, largo=0.45)
    t, alto, interior = 0.03, 1.75, 1.4 - 2 * 0.03
    laterales = 2 * t * alto * 0.45
    fondo = interior * alto * 0.02                       # sin la parte metida en los laterales
    repisas = 5 * interior * 0.025 * (0.44 - 0.01)       # sin la parte metida en laterales y fondo
    assert np.isclose(volumen(m), laterales + fondo + repisas, rtol=1e-6)


def test_conserva_color_de_cada_operando_y_acepta_mallas():
    a = SolidoCSG.desde_malla(crear_prisma(0, 0, 0, 2, 2, 2, color=(200, 0, 0), textura_id=None))
    b = caja(1, 0, 0, 2, 2, 2, color=(0, 0, 200))
    m = (a | b).a_malla()
    colores = {tuple(c) for c in m.colores_vertice.astype(int)}
    assert colores == {(200, 0, 0), (0, 0, 200)}
    assert np.isclose(volumen(m), 12.0)
