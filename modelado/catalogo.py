"""Catálogo de muebles 3D parametrizados (B-Rep, CSG, Bézier y fractales) sincronizados con el plano."""
from functools import lru_cache
import numpy as np
from mathlib import Mat4
from .malla import Malla
from .curvas import Bezier, revolucion
from .fractales import arbol_fractal
from .csg import caja, esfera, HojaCSG, OperacionCSG


def crear_prisma(
    cx: float,
    cy: float,
    cz: float,
    ancho: float,
    alto: float,
    largo: float,
    color: tuple[int, int, int] = (180, 140, 100),
    textura_id: str | None = "madera",
) -> Malla:
    """Crea una caja B-Rep centrada en (cx, cy, cz) con dimensiones (ancho, alto, largo)."""
    hx, hy, hz = ancho / 2.0, alto / 2.0, largo / 2.0
    v = np.array([
        [cx - hx, cy - hy, cz + hz],  # 0: frente-inf-izq (+Z)
        [cx + hx, cy - hy, cz + hz],  # 1: frente-inf-der
        [cx + hx, cy + hy, cz + hz],  # 2: frente-sup-der
        [cx - hx, cy + hy, cz + hz],  # 3: frente-sup-izq
        [cx + hx, cy - hy, cz - hz],  # 4: atras-inf-der (-Z)
        [cx - hx, cy - hy, cz - hz],  # 5: atras-inf-izq
        [cx - hx, cy + hy, cz - hz],  # 6: atras-sup-izq
        [cx + hx, cy + hy, cz - hz],  # 7: atras-sup-der
    ], dtype=np.float64)

    caras = np.array([
        [0, 1, 2], [0, 2, 3],  # Frente (+Z)
        [4, 5, 6], [4, 6, 7],  # Atrás (-Z)
        [5, 0, 3], [5, 3, 6],  # Izquierda (-X)
        [1, 4, 7], [1, 7, 2],  # Derecha (+X)
        [3, 2, 7], [3, 7, 6],  # Arriba (+Y)
        [5, 4, 1], [5, 1, 0],  # Abajo (-Y)
    ], dtype=np.int32)

    # Cada cara con sus propios vértices: aristas vivas en la luz y textura sin deformar.
    m = Malla(vertices=v, caras=caras, color=color, textura_id=textura_id, nombre="prisma")
    m = m.separar_vertices()
    m.aplicar_mapeado_uv(tipo="cubico", escala=1.0)
    return m


# Objetos pequeños que se apoyan sobre otros muebles si se colocan encima de ellos.
# La altura de apoyo no se fija a mano: se mide sobre la malla 3D del mueble de abajo
# (ver Malla.altura_en y Escena.altura_apoyo).
DECORATIVOS = ("jarron", "jarrón", "lampara", "lámpara", "planta")


@lru_cache(maxsize=64)
def _estante_csg(ancho: float, largo: float, color: tuple) -> Malla:
    """Estante como árbol CSG de unión: laterales ∪ repisas ∪ fondo. Las repisas entran
    medio grosor en los laterales y en el fondo, así la unión elimina las caras internas.
    Se guarda en caché por medidas: la escena se reconstruye a menudo (al mover muebles)."""
    alto, t = 1.75, 0.03
    hojas = [HojaCSG("lateral izq.", caja(-ancho / 2 + t / 2, alto / 2, 0, t, alto, largo, color, "madera")),
             HojaCSG("lateral der.", caja(ancho / 2 - t / 2, alto / 2, 0, t, alto, largo, color, "madera")),
             HojaCSG("fondo", caja(0, alto / 2, -largo / 2 + 0.01, ancho - t, alto, 0.02, color, "madera"))]
    for k, y in enumerate((0.06, 0.46, 0.86, 1.26, alto - 0.0125)):
        hojas.append(HojaCSG(f"repisa {k + 1}", caja(0, y, 0.005, ancho - t, 0.025, largo - 0.01, color, "madera")))
    return OperacionCSG("union", *hojas).evaluar().a_malla(nombre="estante")


@lru_cache(maxsize=64)
def _velador_csg(ancho: float, largo: float, color: tuple) -> Malla:
    """Velador como intersección CSG: caja ∩ esfera. La esfera es algo menor que la diagonal
    de la caja, así que recorta las esquinas y aristas (queda una caja redondeada)."""
    alto = 0.55
    semidiagonal = float(np.linalg.norm([ancho / 2, alto / 2, largo / 2]))
    cuerpo = HojaCSG("caja", caja(0, alto / 2, 0, ancho, alto, largo, color, "madera"))
    redondeo = HojaCSG("esfera", esfera(0, alto / 2, 0, 0.88 * semidiagonal, color, "madera",
                                        sectores=24, anillos=16))
    return OperacionCSG("interseccion", cuerpo, redondeo).evaluar().a_malla(nombre="velador")


def es_decorativo(nombre: str) -> bool:
    clave = nombre.lower().strip()
    return any(p in clave for p in DECORATIVOS)


def construir_mueble_3d(
    nombre: str,
    ancho: float = 1.0,
    largo: float = 1.0,
    x: float = 0.0,
    y_plano: float = 0.0,
    angulo: float = 0.0,
    color: tuple[int, int, int] = (190, 150, 110),
    elevacion: float = 0.0,
    escala_x: float = 1.0,
    escala_z: float = 1.0,
    perfil: tuple | None = None,
) -> Malla:
    """
    Genera la malla 3D de un mueble del catálogo (mesa, silla, sofá, cama, ropero,
    estante, velador, lámpara, jarrón, planta) y la ubica en (x, elevacion, -y_plano)
    rotada `angulo`°. `elevacion` es la altura de apoyo (0 = piso; 0.75 = sobre una mesa).
    `escala_x` y `escala_z` son el escalamiento del plano (ancho y largo); la altura no cambia.
    `perfil` (opcional): puntos de control (r, y) de la curva Bézier de un jarrón personalizado.
    """
    clave = nombre.lower().strip()
    piezas: list[Malla] = []

    if "mesa" in clave:
        alto_mesa = 0.75
        grosor_tablero = 0.06
        piezas.append(crear_prisma(0, alto_mesa - grosor_tablero / 2, 0, ancho, grosor_tablero, largo, color, "madera"))
        pata_w = min(0.07, ancho * 0.12, largo * 0.12)
        for sx in (-1, 1):
            for sz in (-1, 1):
                px = sx * (ancho / 2 - pata_w)
                pz = sz * (largo / 2 - pata_w)
                piezas.append(crear_prisma(px, (alto_mesa - grosor_tablero) / 2, pz, pata_w, alto_mesa - grosor_tablero, pata_w, (120, 85, 55), "madera"))

    elif "silla" in clave:
        alto_asiento = 0.45
        piezas.append(crear_prisma(0, alto_asiento, 0, ancho, 0.05, largo, color, "madera"))
        piezas.append(crear_prisma(0, alto_asiento + 0.25, -largo / 2 + 0.03, ancho, 0.50, 0.05, color, "madera"))
        pw = 0.045
        for sx in (-1, 1):
            for sz in (-1, 1):
                piezas.append(crear_prisma(sx * (ancho / 2 - pw), alto_asiento / 2, sz * (largo / 2 - pw), pw, alto_asiento, pw, (110, 80, 50), "madera"))

    elif "sof" in clave:
        # Base, asiento acolchado, respaldo y apoyabrazos
        piezas.append(crear_prisma(0, 0.20, 0, ancho, 0.40, largo, color, None))
        piezas.append(crear_prisma(0, 0.55, -largo * 0.35, ancho, 0.45, largo * 0.28, tuple(max(0, c - 20) for c in color), None))
        brazo_w = min(0.16, ancho * 0.12)
        piezas.append(crear_prisma(-ancho / 2 + brazo_w / 2, 0.35, 0, brazo_w, 0.55, largo, color, None))
        piezas.append(crear_prisma(ancho / 2 - brazo_w / 2, 0.35, 0, brazo_w, 0.55, largo, color, None))

    elif "cama" in clave:
        # Colchón, base y cabecera
        piezas.append(crear_prisma(0, 0.18, 0, ancho, 0.36, largo, (140, 100, 65), "madera"))
        piezas.append(crear_prisma(0, 0.44, 0.04, ancho * 0.96, 0.22, largo * 0.94, color, None))
        piezas.append(crear_prisma(0, 0.55, -largo / 2 + 0.05, ancho, 1.10, 0.08, (130, 90, 55), "madera"))
        piezas.append(crear_prisma(0, 0.58, -largo * 0.32, ancho * 0.72, 0.12, 0.35, (245, 245, 248), None))

    elif "estante" in clave:
        # CSG: unión de dos laterales, cinco repisas y el fondo (ver _estante_csg)
        piezas.append(_estante_csg(round(ancho, 3), round(largo, 3), tuple(color)))

    elif "ropero" in clave:
        piezas.append(crear_prisma(0, 1.00, 0, ancho, 2.00, largo, color, "madera"))

    elif "velador" in clave:
        # CSG: intersección de una caja con una esfera → caja con aristas redondeadas
        piezas.append(_velador_csg(round(ancho, 3), round(largo, 3), tuple(color)))

    elif "jarron" in clave or "jarrón" in clave:
        # Perfil cerrado con pared de 1.2 cm: sólido hueco con boca, interior y fondo
        # Si el mueble trae `perfil` (curva dibujada en el editor Bézier), se usa esa curva.
        curva = Bezier(perfil if perfil else [(0.0, 0.0), (0.14, 0.02), (0.20, 0.18), (0.08, 0.34), (0.13, 0.46)])
        # Resolución moderada: con grosor la malla tiene cara exterior e interior (doble de
        # triángulos) y el render por software recorre los triángulos uno a uno.
        piezas.append(revolucion(curva, segmentos_radiales=16, pasos_perfil=12, color=color,
                                 textura_id="ceramica", nombre="jarron", grosor=0.012))

    elif "lampara" in clave or "lámpara" in clave:
        # Base y pantalla por revolución de Bézier (ambas huecas, con grosor)
        base_c = Bezier([(0.0, 0.0), (0.15, 0.02), (0.04, 0.12), (0.03, 0.45)])
        pantalla_c = Bezier([(0.22, 0.42), (0.18, 0.58), (0.11, 0.72)])
        piezas.append(revolucion(base_c, 14, 8, color=(150, 120, 70), textura_id=None,
                                 nombre="pie_lampara", grosor=0.01))
        piezas.append(revolucion(pantalla_c, 18, 4, color=(248, 232, 185), textura_id=None,
                                 nombre="pantalla_lampara", grosor=0.006))

    elif "planta" in clave:
        # Maceta hueca por revolución + tierra + árbol fractal
        maceta = Bezier([(0.0, 0.0), (0.14, 0.0), (0.18, 0.24), (0.16, 0.26)])
        piezas.append(revolucion(maceta, 16, 6, color=(175, 95, 65), textura_id="ceramica",
                                 nombre="maceta", grosor=0.012))
        # Disco de tierra: perfil horizontal recorrido hacia el eje, así su normal mira hacia arriba
        tierra = Bezier([(0.163, 0.21), (0.0, 0.21)])
        piezas.append(revolucion(tierra, 16, 1, color=(92, 62, 40), textura_id=None, nombre="tierra"))
        arbol = arbol_fractal(profundidad=4, longitud_inicial=0.32, radio_inicial=0.026)
        arbol_subido = arbol.aplicar_transformacion(Mat4.traslacion(0.0, 0.22, 0.0))
        piezas.append(arbol_subido)

    else:
        piezas.append(crear_prisma(0, 0.35, 0, ancho, 0.70, largo, color, "madera"))

    combinada = Malla.combinar(piezas, nombre=nombre)
    # Matriz de modelo M = T(x, elevación, −y) · Ry(ángulo) · S(sx, 1, sz): la misma
    # composición T · R · S del plano 2D, llevada a 4×4 (el eje y del plano es −Z en 3D)
    mat_modelo = (Mat4.traslacion(x, elevacion, -y_plano) @ Mat4.rotacion_y(angulo)
                  @ Mat4.escalamiento(escala_x, 1.0, escala_z))
    return combinada.aplicar_transformacion(mat_modelo)
