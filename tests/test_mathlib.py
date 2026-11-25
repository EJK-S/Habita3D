import numpy as np
from mathlib import Vec2, Vec3, Mat3, Mat4


def test_traslacion_y_escalamiento():
    assert (Mat3.traslacion(3, -2) @ Vec2(1, 1)).casi_igual(Vec2(4, -1))
    assert (Mat3.escalamiento(2, 3) @ Vec2(1, 1)).casi_igual(Vec2(2, 3))


def test_rotacion_90_grados():
    assert (Mat3.rotacion(90) @ Vec2(1, 0)).casi_igual(Vec2(0, 1), 1e-12)


def test_composicion_aplica_primero_la_de_la_derecha():
    # T · R · S: primero escala, luego rota, al final traslada
    m = Mat3.traslacion(10, 0) @ Mat3.rotacion(90) @ Mat3.escalamiento(2)
    assert (m @ Vec2(1, 0)).casi_igual(Vec2(10, 2), 1e-12)


def test_rotacion_sobre_un_centro_lo_deja_fijo():
    m = Mat3.rotacion_en(37, 5, 4)
    assert (m @ Vec2(5, 4)).casi_igual(Vec2(5, 4), 1e-12)
    assert (Mat3.rotacion_en(180, 5, 4) @ Vec2(6, 4)).casi_igual(Vec2(4, 4), 1e-12)


def test_reflexiones_y_cizallamiento():
    assert (Mat3.reflexion_x() @ Vec2(2, 3)).casi_igual(Vec2(2, -3))
    assert (Mat3.reflexion_y() @ Vec2(2, 3)).casi_igual(Vec2(-2, 3))
    assert (Mat3.cizallamiento(shx=0.5) @ Vec2(2, 4)).casi_igual(Vec2(4, 4))


def test_inversa_recupera_el_punto():
    m = Mat3.traslacion(3, 1) @ Mat3.rotacion(25) @ Mat3.escalamiento(2, 0.5)
    p = Vec2(1.7, -2.2)
    assert (m.inversa() @ (m @ p)).casi_igual(p, 1e-9)


def test_aplicar_puntos_igual_que_punto_a_punto():
    m = Mat3.traslacion(1, 2) @ Mat3.rotacion(40)
    pts = np.array([(0, 0), (1, 0), (2, 3)])
    out = m.aplicar_puntos(pts)
    for fila, p in zip(out, pts):
        assert Vec2(*fila).casi_igual(m @ Vec2(*p), 1e-12)


def test_mat4_rotacion_y_90():
    assert (Mat4.rotacion_y(90) @ Vec3(1, 0, 0)).casi_igual(Vec3(0, 0, -1), 1e-12)


def test_mat4_rotaciones_x_y_z():
    assert (Mat4.rotacion_x(90) @ Vec3(0, 1, 0)).casi_igual(Vec3(0, 0, 1), 1e-12)
    assert (Mat4.rotacion_z(90) @ Vec3(1, 0, 0)).casi_igual(Vec3(0, 1, 0), 1e-12)


def test_mat4_composicion_e_inversa():
    m = Mat4.traslacion(1, 2, 3) @ Mat4.rotacion_y(30) @ Mat4.escalamiento(2)
    p = Vec3(1, -1, 0.5)
    assert (m.inversa() @ (m @ p)).casi_igual(p, 1e-9)


def test_producto_cruz_y_punto():
    x, y, z = Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)
    assert x.cruz(y).casi_igual(z)
    assert x.punto(y) == 0
    assert abs(Vec3(3, 4, 12).normalizado().norma() - 1) < 1e-12
