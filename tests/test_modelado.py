"""Pruebas unitarias del paquete modelado (B-Rep, CSG, Bézier, revolución y fractales)."""
import numpy as np
from modelado import extruir_muro, csg_diferencia, Bezier, revolucion, arbol_fractal


def test_extruir_muro_brep_cerrado():
    m = extruir_muro((0.0, 0.0), (4.0, 0.0), espesor=0.2, altura=2.5)
    assert len(m.vertices) == 8
    assert len(m.caras) == 12
    # Normales unitarias
    normas = np.linalg.norm(m.normales_cara, axis=1)
    assert np.allclose(normas, 1.0, atol=1e-6)


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


def test_revolucion_y_fractal():
    b = Bezier([(0.1, 0.0), (0.2, 0.5), (0.1, 1.0)])
    m = revolucion(b, segmentos_radiales=12, pasos_perfil=8)
    assert len(m.caras) == 12 * 8 * 2

    f1 = arbol_fractal(profundidad=2, ramas_por_nodo=3)
    f2 = arbol_fractal(profundidad=4, ramas_por_nodo=3)
    assert len(f2.caras) > len(f1.caras)
