"""Genera las figuras de las Fases 3 a 5 (3D, proyecciones, modelado, fractales, texturas,
iluminación y rendimiento) en docs/figuras/.
Uso:  python demos/fase3_5.py"""
import pathlib
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

raiz = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(raiz))

from plano import Plano, MuebleEnPlanta
from modelado import extruir_muro, csg_diferencia, Bezier, revolucion, arbol_fractal
from escena import Escena

SALIDA = raiz / "docs" / "figuras"
SALIDA.mkdir(parents=True, exist_ok=True)


def fuente(t=14):
    for f in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(f, t)
        except OSError:
            pass
    return ImageFont.load_default()


def fuente_simbolos(t=14):
    """Arial no tiene el símbolo ∪: para títulos con símbolos de conjuntos (CSG)."""
    for f in ("DejaVuSans.ttf", "seguisym.ttf", "cambria.ttc"):
        try:
            return ImageFont.truetype(f, t)
        except OSError:
            pass
    return fuente(t)


def con_titulo(img: Image.Image, texto: str) -> Image.Image:
    f = fuente_simbolos(14) if any(s in texto for s in "∪∩") else fuente(14)
    barra = Image.new("RGB", (img.width, 28), (245, 245, 245))
    ImageDraw.Draw(barra).text((8, 6), texto, fill=(30, 30, 30), font=f)
    out = Image.new("RGB", (img.width, img.height + 28), "white")
    out.paste(barra, (0, 0))
    out.paste(img, (0, 28))
    return out


def lado_a_lado(imagenes: list[Image.Image], sep: int = 6) -> Image.Image:
    w = sum(i.width for i in imagenes) + sep * (len(imagenes) - 1)
    h = max(i.height for i in imagenes)
    out = Image.new("RGB", (w, h), (120, 120, 120))
    x = 0
    for i in imagenes:
        out.paste(i, (x, 0))
        x += i.width + sep
    return out


def crear_plano_demo() -> Plano:
    pl = Plano()
    sala = pl.agregar_habitacion(0, 0, 6.0, 4.5)
    div = pl.agregar_muro((3.6, 0), (3.6, 4.5))
    pl.agregar_abertura(sala[0], 0.22, 1.0, "puerta")
    pl.agregar_abertura(sala[2], 0.28, 1.4, "ventana")
    pl.agregar_abertura(sala[2], 0.82, 1.1, "ventana")
    pl.agregar_abertura(div, 0.70, 0.9, "puerta")
    pl.agregar_mueble(MuebleEnPlanta("sofá", 2.0, 0.9, 1.5, 3.7, 0, (130, 150, 210)))
    pl.agregar_mueble(MuebleEnPlanta("mesa", 1.1, 1.1, 1.6, 1.8, 20, (205, 170, 120)))
    pl.agregar_mueble(MuebleEnPlanta("cama", 1.5, 2.0, 5.0, 3.0, 0, (230, 140, 150)))
    pl.agregar_mueble(MuebleEnPlanta("planta", 0.45, 0.45, 0.55, 0.75, 0, (70, 155, 80)))
    pl.agregar_mueble(MuebleEnPlanta("jarron", 0.40, 0.40, 1.6, 1.8, 0, (195, 115, 85)))
    return pl


# ------------------------------------------------ Figura 6: Ortográfica vs Perspectiva (wireframe / vista 3D)
esc = Escena(crear_plano_demo(), carpeta_texturas=raiz / "assets" / "texturas")
esc.camara.mirar_a((3.0, 6.2, 4.2), (3.0, 0.6, -2.2))
esc.modo_alambre = True
esc.proyeccion = "ortografica"
img_orto = con_titulo(esc.renderizar_3d(420, 300).a_imagen(), "Proyección Ortográfica (rayos paralelos)")

esc.proyeccion = "perspectiva"
img_persp = con_titulo(esc.renderizar_3d(420, 300).a_imagen(), "Proyección en Perspectiva (convergencia focal)")
lado_a_lado([img_orto, img_persp]).save(SALIDA / "fig_06_ortografica_vs_perspectiva.png")
esc.modo_alambre = False
print("Generada fig_06_ortografica_vs_perspectiva.png")

# ------------------------------------------------ Figura 7: Modelado B-Rep, CSG y Revolución Bézier
p_brep = Escena(Plano())
p_brep.camara.mirar_a((2.2, 1.8, 2.6), (1.5, 1.1, 0.0))
p_brep.luz.posicion.x, p_brep.luz.posicion.y, p_brep.luz.posicion.z = 2.0, 3.0, 3.0
p_brep.texturas_activas = False
p_brep.agregar_malla_extra(extruir_muro((0.0, 0.0), (3.0, 0.0), 0.2, 2.2, color=(215, 200, 180)))
i1 = con_titulo(p_brep.renderizar_3d(300, 260).a_imagen(), "1. Muro B-Rep extruido")

p_csg = Escena(Plano())
p_csg.camara.mirar_a((2.2, 1.8, 2.6), (1.5, 1.1, 0.0))
p_csg.luz.posicion.x, p_csg.luz.posicion.y, p_csg.luz.posicion.z = 2.0, 3.0, 3.0
p_csg.texturas_activas = False
p_csg.agregar_malla_extra(
    csg_diferencia(
        (0.0, 0.0), (3.0, 0.0), 0.2, 2.2,
        [{"pos": 0.25, "ancho": 0.75, "tipo": "puerta"}, {"pos": 0.72, "ancho": 0.90, "tipo": "ventana"}],
        color=(215, 200, 180),
    )
)
i2 = con_titulo(p_csg.renderizar_3d(300, 260).a_imagen(), "2. CSG: vanos de puerta y ventana")

p_rev = Escena(Plano())
p_rev.camara.mirar_a((0.0, 0.42, 0.95), (0.0, 0.24, 0.0))
p_rev.luz.posicion.x, p_rev.luz.posicion.y, p_rev.luz.posicion.z = 0.8, 1.2, 1.2
p_rev.texturas_activas = False
curva_jarron = Bezier([(0.0, 0.0), (0.14, 0.02), (0.21, 0.18), (0.07, 0.34), (0.13, 0.46)])
p_rev.agregar_malla_extra(revolucion(curva_jarron, segmentos_radiales=28, pasos_perfil=22, color=(205, 110, 75)))
i3 = con_titulo(p_rev.renderizar_3d(300, 260).a_imagen(), "3. Bézier + Superficie de revolución")
lado_a_lado([i1, i2, i3]).save(SALIDA / "fig_07_modelado_brep_csg_bezier.png")
print("Generada fig_07_modelado_brep_csg_bezier.png")

# ------------------------------------------------ Figura 8: Fractales y niveles de recursión
paneles_frac = []
for nivel in (2, 3, 5):
    ef = Escena(Plano())
    ef.camara.mirar_a((0.0, 0.65, 1.50), (0.0, 0.52, 0.0))
    ef.luz.posicion.x, ef.luz.posicion.y, ef.luz.posicion.z = 1.2, 1.8, 1.6
    ef.texturas_activas = False
    m_f = arbol_fractal(profundidad=nivel, longitud_inicial=0.42, radio_inicial=0.032)
    ef.agregar_malla_extra(m_f)
    img_f = ef.renderizar_3d(290, 270).a_imagen()
    paneles_frac.append(con_titulo(img_f, f"Recursión nivel {nivel} ({len(m_f.caras)} triángulos)"))
lado_a_lado(paneles_frac).save(SALIDA / "fig_08_fractales_recursion.png")
print("Generada fig_08_fractales_recursion.png")

# ------------------------------------------------ Figura 9: Sin textura vs Con textura
esc_tex = Escena(crear_plano_demo(), carpeta_texturas=raiz / "assets" / "texturas")
# Cámara dentro de la sala (esquina suroeste) mirando hacia el dormitorio: se ven el piso
# (parquet), las paredes, la madera de mesa y cama, y la cerámica del jarrón.
esc_tex.camara.mirar_a((0.6, 1.6, -0.5), (3.3, 0.9, -4.1))
esc_tex.sombreado = "gouraud"
esc_tex.iluminacion_activa = True
esc_tex.texturas_activas = False
img_sin_tex = con_titulo(esc_tex.renderizar_3d(420, 300).a_imagen(), "Modelo geométrico sin texturas")

esc_tex.texturas_activas = True
img_con_tex = con_titulo(esc_tex.renderizar_3d(420, 300).a_imagen(), "Modelo con texturas (mapeado plano, cúbico y cilíndrico)")
lado_a_lado([img_sin_tex, img_con_tex]).save(SALIDA / "fig_09_sin_vs_con_textura.png")
print("Generada fig_09_sin_vs_con_textura.png")

# ------------------------------------------------ Figura 10: Sin luz vs Sombreado Plano vs Sombreado Gouraud
paneles_luz = []
curva_lampara = Bezier([(0.0, 0.0), (0.24, 0.04), (0.30, 0.25), (0.18, 0.48), (0.04, 0.62)])
malla_esfera = revolucion(curva_lampara, segmentos_radiales=20, pasos_perfil=14, color=(95, 155, 220))

for titulo, ilum, modo_s in [
    ("Sin iluminación (color plano)", False, "plano"),
    ("Iluminación Phong + Sombreado Plano", True, "plano"),
    ("Iluminación Phong + Sombreado Gouraud", True, "gouraud"),
]:
    el = Escena(Plano())
    el.camara.mirar_a((0.0, 0.45, 1.15), (0.0, 0.30, 0.0))
    el.luz.posicion.x, el.luz.posicion.y, el.luz.posicion.z = 0.75, 0.95, 1.05
    el.texturas_activas = False
    el.iluminacion_activa = ilum
    el.sombreado = modo_s
    el.agregar_malla_extra(malla_esfera)
    paneles_luz.append(con_titulo(el.renderizar_3d(300, 270).a_imagen(), titulo))
lado_a_lado(paneles_luz).save(SALIDA / "fig_10_iluminacion_plano_vs_gouraud.png")
print("Generada fig_10_iluminacion_plano_vs_gouraud.png")

# ------------------------------------------------ Figura 11: Escena completa 3D integrada
esc_full = Escena(crear_plano_demo(), carpeta_texturas=raiz / "assets" / "texturas")
# Cámara dentro de la sala (esquina sureste, altura de los ojos) mirando hacia la esquina
# noroeste: se ven el sofá, la mesa con el jarrón, la planta fractal y las ventanas.
esc_full.camara.mirar_a((3.3, 1.7, -0.4), (0.4, 0.6, -3.6))
esc_full.proyeccion = "perspectiva"
esc_full.sombreado = "gouraud"
esc_full.iluminacion_activa = True
esc_full.texturas_activas = True
con_titulo(
    esc_full.renderizar_3d(680, 440).a_imagen(),
    "Habita3D: Vista 3D integrada (B-Rep, CSG, Bézier, Fractales, Texturas e Iluminación Gouraud)",
).save(SALIDA / "fig_11_escena_completa_3d.png")
print("Generada fig_11_escena_completa_3d.png")

# ------------------------------------------------ Figura 12: CSG general (unión, intersección, diferencia)
from modelado.csg import caja, esfera, HojaCSG, OperacionCSG

a_csg = HojaCSG("A", caja(0, 0.5, 0, 1.0, 1.0, 1.0, (215, 95, 70)))
b_csg = HojaCSG("B", esfera(0.45, 0.85, 0.40, 0.55, (70, 125, 215), sectores=24, anillos=16))
paneles_csg = []
for titulo, arbol, azimut in (
    ("A y B (sin operar)", None, 40),
    ("Unión  A ∪ B", OperacionCSG("union", a_csg, b_csg), 40),
    ("Intersección  A ∩ B", OperacionCSG("interseccion", a_csg, b_csg), 220),
    ("Diferencia  A − B", OperacionCSG("diferencia", a_csg, b_csg), 40),
):
    e_csg = Escena(Plano())
    e_csg.sombreado = "plano"
    az, el = np.radians(azimut), np.radians(28)
    objetivo = np.array([0.15, 0.5, 0.15])
    ojo = objetivo + 3.2 * np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    e_csg.camara.mirar_a(tuple(ojo), tuple(objetivo))
    luz = objetivo + 3.0 * np.array([np.cos(az + 0.6), 0.9, np.sin(az + 0.6)])
    e_csg.luz.posicion.x, e_csg.luz.posicion.y, e_csg.luz.posicion.z = luz
    if arbol is None:
        e_csg.mallas_extra = [a_csg.solido.a_malla(), b_csg.solido.a_malla()]
    else:
        e_csg.mallas_extra = [arbol.evaluar().a_malla()]
    paneles_csg.append(con_titulo(e_csg.renderizar_3d(260, 230).a_imagen(), titulo))
lado_a_lado(paneles_csg).save(SALIDA / "fig_12_csg_union_interseccion_diferencia.png")
print("Generada fig_12_csg_union_interseccion_diferencia.png")

# ------------------------------------------------ Benchmark de rendimiento (Tabla 6.3)
print("\n=== TABLA DE RENDIMIENTO (480x320, render por software NumPy) ===")
escenarios = []

# Escena 1: Habitación vacía (piso + 4 muros)
p1 = Plano()
p1.agregar_habitacion(0, 0, 5, 4)
e1 = Escena(p1, carpeta_texturas=raiz / "assets" / "texturas")
e1.sombreado = "plano"
escenarios.append(("Habitación vacía (muros B-Rep)", e1, "Sombreado plano, sin muebles"))

# Escena 2: Vivienda con puertas y ventanas CSG
p2 = Plano()
s2 = p2.agregar_habitacion(0, 0, 6, 4.5)
p2.agregar_abertura(s2[0], 0.25, 1.0, "puerta")
p2.agregar_abertura(s2[2], 0.50, 1.4, "ventana")
e2 = Escena(p2, carpeta_texturas=raiz / "assets" / "texturas")
e2.sombreado = "plano"
escenarios.append(("Vivienda con vanos CSG", e2, "CSG puerta + ventana, texturas ON"))

# Escena 3: Vivienda amoblada básica (sin fractales densos)
p3 = Plano()
s3 = p3.agregar_habitacion(0, 0, 6, 4.5)
p3.agregar_abertura(s3[0], 0.25, 1.0, "puerta")
p3.agregar_abertura(s3[2], 0.50, 1.4, "ventana")
p3.agregar_mueble(MuebleEnPlanta("sofá", 2.0, 0.9, 1.5, 3.7, 0))
p3.agregar_mueble(MuebleEnPlanta("mesa", 1.1, 1.1, 1.6, 1.8, 20))
p3.agregar_mueble(MuebleEnPlanta("cama", 1.5, 2.0, 5.0, 3.0, 0))
e3 = Escena(p3, carpeta_texturas=raiz / "assets" / "texturas")
e3.sombreado = "gouraud"
escenarios.append(("Vivienda amoblada (Gouraud)", e3, "Muebles B-Rep + iluminación Gouraud"))

# Escena 4: Escena completa (muebles + jarrón Bézier + planta fractal)
escenarios.append(("Escena completa (Bézier + Fractal)", esc_full, "Jarrón hueco por revolución 16x12, maceta y fractal n=4"))

print("| Escena | N.º de triángulos | Tiempo/frame (ms) | FPS promedio | Observaciones |")
print("|---|---|---|---|---|")
for nombre_esc, esc_obj, obs in escenarios:
    # Calentamiento
    esc_obj.renderizar_3d(480, 320)
    repeticiones = 5
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        esc_obj.renderizar_3d(480, 320)
    dt_ms = ((time.perf_counter() - t0) / repeticiones) * 1000.0
    fps = 1000.0 / max(dt_ms, 1e-6)
    print(f"| {nombre_esc} | {esc_obj.total_triangulos()} | {dt_ms:.1f} ms | {fps:.1f} FPS | {obs} |")
