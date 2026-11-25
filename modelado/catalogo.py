"""Catálogo de muebles 3D parametrizados (B-Rep, Bézier y fractales) sincronizados con el plano."""
import numpy as np
from mathlib import Mat4
from .malla import Malla
from .curvas import Bezier, revolucion
from .fractales import arbol_fractal


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

    m = Malla(vertices=v, caras=caras, color=color, textura_id=textura_id, nombre="prisma")
    m.aplicar_mapeado_uv(tipo="cubico", escala=1.0)
    return m


def construir_mueble_3d(
    nombre: str,
    ancho: float = 1.0,
    largo: float = 1.0,
    x: float = 0.0,
    y_plano: float = 0.0,
    angulo: float = 0.0,
    color: tuple[int, int, int] = (190, 150, 110),
) -> Malla:
    """
    Genera la malla 3D de un mueble del catálogo (mesa, silla, sofá, cama, ropero,
    estante, velador, lámpara, jarrón, planta) y la ubica en (x, 0, -y_plano) rotada `angulo`°.
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

    elif "ropero" in clave or "estante" in clave:
        alto_r = 2.00 if "ropero" in clave else 1.75
        piezas.append(crear_prisma(0, alto_r / 2, 0, ancho, alto_r, largo, color, "madera"))

    elif "velador" in clave:
        alto_v = 0.55
        piezas.append(crear_prisma(0, alto_v / 2, 0, ancho, alto_v, largo, color, "madera"))

    elif "jarron" in clave or "jarrón" in clave:
        curva = Bezier([(0.0, 0.0), (0.14, 0.02), (0.20, 0.18), (0.08, 0.34), (0.13, 0.46)])
        piezas.append(revolucion(curva, segmentos_radiales=24, pasos_perfil=20, color=color, textura_id="ceramica", nombre="jarron"))

    elif "lampara" in clave or "lámpara" in clave:
        # Base y pantalla por revolución de Bézier
        base_c = Bezier([(0.0, 0.0), (0.15, 0.02), (0.04, 0.12), (0.03, 0.45)])
        pantalla_c = Bezier([(0.22, 0.42), (0.18, 0.58), (0.11, 0.72)])
        piezas.append(revolucion(base_c, 20, 14, color=(150, 120, 70), textura_id=None, nombre="pie_lampara"))
        piezas.append(revolucion(pantalla_c, 24, 12, color=(248, 232, 185), textura_id=None, nombre="pantalla_lampara"))

    elif "planta" in clave:
        # Maceta por revolución + árbol fractal
        maceta = Bezier([(0.0, 0.0), (0.14, 0.0), (0.18, 0.24), (0.16, 0.26)])
        piezas.append(revolucion(maceta, 20, 12, color=(175, 95, 65), textura_id="ceramica", nombre="maceta"))
        arbol = arbol_fractal(profundidad=4, longitud_inicial=0.32, radio_inicial=0.026)
        arbol_subido = arbol.aplicar_transformacion(Mat4.traslacion(0.0, 0.22, 0.0))
        piezas.append(arbol_subido)

    else:
        piezas.append(crear_prisma(0, 0.35, 0, ancho, 0.70, largo, color, "madera"))

    combinada = Malla.combinar(piezas, nombre=nombre)
    mat_modelo = Mat4.traslacion(x, 0.0, -y_plano) @ Mat4.rotacion_y(angulo)
    return combinada.aplicar_transformacion(mat_modelo)
