"""Pruebas unitarias del paquete modelado (B-Rep, CSG, Bézier, revolución y fractales)."""
import numpy as np
from modelado import (extruir_muro, csg_diferencia, Bezier, revolucion, arbol_fractal,
                      crear_prisma, Malla)


def test_extruir_muro_brep_cerrado():
    m = extruir_muro((0.0, 0.0), (4.0, 0.0), espesor=0.2, altura=2.5)
    # El prisma tiene 8 esquinas distintas y 12 triángulos (6 caras)
    assert len(np.unique(np.round(m.vertices, 9), axis=0)) == 8
    assert len(m.caras) == 12
    # Normales unitarias
    normas = np.linalg.norm(m.normales_cara, axis=1)
    assert np.allclose(normas, 1.0, atol=1e-6)


def test_caja_tiene_normales_de_vertice_iguales_a_su_cara():
    # Sin vértices compartidos entre caras, Gouraud no redondea las aristas de la caja
    m = extruir_muro((0.0, 0.0), (4.0, 0.0), espesor=0.2, altura=2.5)
    for k in range(3):
        assert np.allclose(m.normales_vertice[m.caras[:, k]], m.normales_cara, atol=1e-9)


def test_combinar_conserva_color_y_textura_de_cada_pieza():
    a = crear_prisma(0, 0, 0, 1, 1, 1, color=(200, 0, 0), textura_id="madera")
    b = crear_prisma(2, 0, 0, 1, 1, 1, color=(0, 0, 200), textura_id=None)
    m = Malla.combinar([a, b])
    assert np.allclose(m.colores_vertice[: len(a.vertices)], (200, 0, 0))
    assert np.allclose(m.colores_vertice[len(a.vertices):], (0, 0, 200))
    assert m.texturas_cara == ["madera"] * len(a.caras) + [None] * len(b.caras)
    # Al trasladar el objeto combinado se mantienen los colores por pieza
    from mathlib import Mat4
    t = m.aplicar_transformacion(Mat4.traslacion(1, 0, 0))
    assert np.allclose(t.colores_vertice, m.colores_vertice)
    assert t.texturas_cara == m.texturas_cara


def test_arbol_fractal_tiene_hojas_verdes():
    f = arbol_fractal(profundidad=3)
    colores = {tuple(c) for c in f.colores_vertice.astype(int)}
    assert (52, 150, 68) in colores and (115, 78, 46) in colores


def test_csg_diferencia_puerta_y_ventana():
    m_puerta = csg_diferencia((0.0, 0.0), (4.0, 0.0), 0.15, 2.6, [{"pos": 0.5, "ancho": 1.0, "tipo": "puerta"}])
    # Puerta genera 3 bloques: tramo izq, dintel superior, tramo der -> 36 triángulos
    assert len(m_puerta.caras) == 36

    m_ventana = csg_diferencia((0.0, 0.0), (4.0, 0.0), 0.15, 2.6, [{"pos": 0.5, "ancho": 1.2, "tipo": "ventana"}])
    # Ventana genera 4 bloques: tramo izq, antepecho, dintel, tramo der -> 48 triángulos
    assert len(m_ventana.caras) == 48


def test_bezier_de_casteljau_extremos_y_punto_medio():
    b = Bezier([(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)])
    assert np.allclose(b.evaluar(0.0), [0.0, 0.0])
    assert np.allclose(b.evaluar(1.0), [2.0, 0.0])
    # Para parábola cuadrática B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2 = (1.0, 1.0)
    assert np.allclose(b.evaluar(0.5), [1.0, 1.0])


def test_niveles_de_casteljau_terminan_en_b_de_t():
    b = Bezier([(0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0)])
    for t in (0.0, 0.3, 0.5, 1.0):
        niveles = b.niveles_de_casteljau(t)
        assert [len(n) for n in niveles] == [4, 3, 2, 1]                # cúbica: 4 → 3 → 2 → 1
        assert np.allclose(niveles[-1][0], b.evaluar(t))
    # Primer nivel en t = 0.5: puntos medios de cada tramo del polígono de control
    assert np.allclose(b.niveles_de_casteljau(0.5)[1], [(0.5, 1.0), (2.0, 2.0), (3.5, 1.0)])


def test_revolucion_y_fractal():
    b = Bezier([(0.1, 0.0), (0.2, 0.5), (0.1, 1.0)])
    m = revolucion(b, segmentos_radiales=12, pasos_perfil=8)
    assert len(m.caras) == 12 * 8 * 2

    f1 = arbol_fractal(profundidad=2, ramas_por_nodo=3)
    f2 = arbol_fractal(profundidad=4, ramas_por_nodo=3)
    assert len(f2.caras) > len(f1.caras)


def test_jarron_con_perfil_bezier_propio():
    from modelado import construir_mueble_3d
    base = construir_mueble_3d("jarrón Bézier", 0.4, 0.4)
    ancho = construir_mueble_3d("jarrón Bézier", 0.4, 0.4,
                                perfil=((0.0, 0.0), (0.40, 0.05), (0.40, 0.30), (0.10, 0.50)))
    assert ancho.vertices[:, 0].max() > base.vertices[:, 0].max() + 0.1     # la curva propia manda
    assert abs(ancho.vertices[:, 1].max() - 0.50) < 0.02                    # su altura es la del perfil


def test_marco_y_hoja_de_puerta_y_ventana():
    from modelado import marco_y_hoja
    puerta = marco_y_hoja((0, 0), (6, 0), 0.15, 2.6, {"pos": 0.5, "ancho": 1.0, "tipo": "puerta"})
    assert [m.nombre for m in puerta] == ["marco_jamba_izq", "marco_jamba_der", "marco_dintel", "hoja_puerta"]
    assert max(m.vertices[:, 1].max() for m in puerta) <= 2.10 + 1e-9        # nada sobre el dintel
    hoja = puerta[-1].vertices
    assert abs(hoja[:, 0].min() - 3.0 + 0.5 - 0.05) < 1e-6                   # junto a la jamba de la bisagra (x = 2.5 + 0.05)
    ventana = marco_y_hoja((0, 0), (6, 0), 0.15, 2.6, {"pos": 0.5, "ancho": 1.4, "tipo": "ventana"})
    nombres = [m.nombre for m in ventana]
    assert "vidrio" in nombres and "hoja_puerta" not in nombres
    assert min(m.vertices[:, 1].min() for m in ventana) >= 0.90 - 0.03       # el alféizar apenas sobresale del vano

