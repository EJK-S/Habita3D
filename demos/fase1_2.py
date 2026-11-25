"""Genera las figuras de las Fases 1 y 2 (base 2D, transformaciones y plano) en docs/figuras/.
Uso:  python demos/fase1_2.py"""
import pathlib, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

raiz = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(raiz))
from mathlib import Mat3
from raster2d import (Framebuffer, puntos_dda, puntos_bresenham, linea, linea_wu, circulo,
                      dibujar_poligono, rellenar_poligono)
from plano import Plano, MuebleEnPlanta

SALIDA = raiz / "docs" / "figuras"
SALIDA.mkdir(parents=True, exist_ok=True)


def fuente(t=14):
    for f in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(f, t)
        except OSError:
            pass
    return ImageFont.load_default()


def con_titulo(img, texto):
    f = fuente(15)
    barra = Image.new("RGB", (img.width, 28), (245, 245, 245))
    ImageDraw.Draw(barra).text((8, 5), texto, fill=(30, 30, 30), font=f)
    out = Image.new("RGB", (img.width, img.height + 28), "white")
    out.paste(barra, (0, 0))
    out.paste(img, (0, 28))
    return out


def apilar(imagenes, sep=6):
    w = max(i.width for i in imagenes)
    h = sum(i.height for i in imagenes) + sep * (len(imagenes) - 1)
    out = Image.new("RGB", (w, h), (120, 120, 120))
    y = 0
    for i in imagenes:
        out.paste(i, (0, y))
        y += i.height + sep
    return out


# ---------------------------------------------------------------- Figura 1: DDA vs Bresenham
W, H, Z = 48, 18, 14
random.seed(7)
mejor, difs = None, -1
for _ in range(400):                       # busca una recta donde los dos algoritmos discrepen
    p1 = (random.randint(1, 8), random.randint(1, H - 2))
    p2 = (random.randint(38, W - 2), random.randint(1, H - 2))
    d = len(set(puntos_dda(*p1, *p2)) ^ set(puntos_bresenham(*p1, *p2)))
    if d > difs:
        mejor, difs = (p1, p2), d
p1, p2 = mejor
A, B = set(puntos_dda(*p1, *p2)), set(puntos_bresenham(*p1, *p2))
paneles = []
for nombre, algo in (("DDA (usa reales y redondea)", "dda"), ("Bresenham (solo enteros)", "bresenham")):
    fb = Framebuffer(W, H)
    linea(fb, *p1, *p2, (0, 0, 0), algoritmo=algo)
    paneles.append(con_titulo(fb.a_imagen(Z, True), nombre))
fb = Framebuffer(W, H)
for x, y in A | B:
    fb.poner_pixel(x, y, (150, 150, 150) if (x, y) in A and (x, y) in B else
                   ((220, 40, 40) if (x, y) in A else (40, 80, 220)))
paneles.append(con_titulo(fb.a_imagen(Z, True), f"Diferencias: rojo = solo DDA, azul = solo Bresenham ({len(A ^ B)} píxeles)"))
apilar(paneles).save(SALIDA / "fig_01_dda_vs_bresenham.png")
print(f"Fig 1: recta {p1}->{p2}, píxeles distintos DDA/Bresenham: {len(A ^ B)} de {len(B)}")

# ---------------------------------------------------------------- Figura 2: aliasing vs Wu
fb1, fb2 = Framebuffer(W, H), Framebuffer(W, H)
linea(fb1, 2, 3, 45, 14, (0, 0, 0), algoritmo="bresenham")
linea(fb2, 2, 3, 45, 14, (0, 0, 0), algoritmo="wu")
apilar([con_titulo(fb1.a_imagen(Z, True), "Bresenham: bordes en escalera (aliasing)"),
        con_titulo(fb2.a_imagen(Z, True), "Wu: intensidad repartida entre píxeles vecinos (antialiasing)")]
       ).save(SALIDA / "fig_02_aliasing_vs_wu.png")

# ---------------------------------------------------------------- Figura 3: primitivas
fb = Framebuffer(560, 340)
circulo(fb, 80, 80, 55, (30, 30, 30), 1, relleno=(255, 214, 150))
circulo(fb, 80, 80, 30, (200, 30, 30), 3)
for i, g in enumerate((1, 3, 6, 10)):
    linea(fb, 160, 40 + 22 * i, 300, 40 + 22 * i + 10, (20, 90, 160), g)
linea(fb, 160, 150, 300, 150, (0, 0, 0), 1, "dda")
rellenar_poligono(fb, [(340, 120), (400, 30), (430, 70), (520, 40), (500, 130), (450, 100), (410, 140)],
                  (170, 220, 170), (20, 100, 20), 2)
estrella = []
for k in range(10):
    r = 70 if k % 2 == 0 else 30
    a = -np.pi / 2 + k * np.pi / 5
    estrella.append((110 + r * np.cos(a), 245 + r * np.sin(a)))
rellenar_poligono(fb, estrella, (250, 220, 70), (140, 90, 0), 2)
u = [(220, 190), (250, 190), (250, 290), (310, 290), (310, 190), (340, 190), (340, 320), (220, 320)]
rellenar_poligono(fb, u, (200, 180, 240), (80, 50, 140), 2)
dibujar_poligono(fb, [(380, 200), (480, 200), (480, 300), (380, 300)], (180, 40, 40), 4)
con_titulo(fb.a_imagen(), "Primitivas 2D: círculo (punto medio), líneas con grosor, relleno par-impar").save(
    SALIDA / "fig_03_primitivas.png")

# ---------------------------------------------------------------- Figura 4: transformaciones
rect = np.array([(-1.5, -0.8), (1.5, -0.8), (1.5, 0.8), (-1.5, 0.8)])
casos = [
    ("Traslación (2, 1)", Mat3.traslacion(2, 1)),
    ("Rotación 35° sobre el origen", Mat3.rotacion(35)),
    ("Rotación 35° sobre su propio centro", Mat3.rotacion_en(35, 0.5, 1.0)),
    ("Escalamiento (1.5, 0.6)", Mat3.escalamiento(1.5, 0.6)),
    ("Reflexión respecto al eje X", Mat3.reflexion_x()),
    ("Cizallamiento shx = 0.8", Mat3.cizallamiento(0.8, 0)),
    ("Compuesta T · R · S", Mat3.traslacion(1.5, 1) @ Mat3.rotacion(30) @ Mat3.escalamiento(0.7)),
    ("Reflexión respecto al eje Y", Mat3.reflexion_y()),
]
PW = 300
panel_imgs = []
for titulo, M in casos:
    fb = Framebuffer(PW, PW)
    a_pant = Mat3.traslacion(PW / 2, PW / 2) @ Mat3.escalamiento(36, -36)
    linea(fb, 0, PW / 2, PW, PW / 2, (190, 190, 190))
    linea(fb, PW / 2, 0, PW / 2, PW, (190, 190, 190))
    # el rectángulo original (centro en (0.5, 1)) no es simétrico respecto a los ejes, así cada reflexión y giro se nota
    base = rect + np.array([0.5, 1.0])
    dibujar_poligono(fb, a_pant.aplicar_puntos(base), (120, 120, 120), 1)
    rellenar_poligono(fb, a_pant.aplicar_puntos(M.aplicar_puntos(base)), (255, 170, 80), (180, 70, 0), 2)
    panel_imgs.append(con_titulo(fb.a_imagen(), titulo))
filas = []
for i in range(0, 8, 4):
    fila = Image.new("RGB", (PW * 4 + 18, panel_imgs[0].height), (120, 120, 120))
    for j in range(4):
        fila.paste(panel_imgs[i + j], (j * (PW + 6), 0))
    filas.append(fila)
apilar(filas).save(SALIDA / "fig_04_transformaciones.png")

# ---------------------------------------------------------------- Figura 5: plano de la vivienda
pl = Plano()
sala = pl.agregar_habitacion(0, 0, 6, 4.5)
muro_div = pl.agregar_muro((3.6, 0), (3.6, 4.5))          # divide la vivienda en sala y dormitorio
pl.agregar_abertura(sala[0], 0.2, 1.0, "puerta")          # puerta de entrada (sur)
pl.agregar_abertura(sala[2], 0.28, 1.4, "ventana")        # ventana de la sala (norte)
pl.agregar_abertura(sala[2], 0.82, 1.1, "ventana")        # ventana del dormitorio (norte)
pl.agregar_abertura(muro_div, 0.7, 0.9, "puerta")         # puerta interior
pl.agregar_mueble(MuebleEnPlanta("sofá", 2.0, 0.9, 1.5, 3.7, 0, (130, 150, 210)))
pl.agregar_mueble(MuebleEnPlanta("mesa", 1.1, 1.1, 1.6, 1.8, 20, (205, 170, 120)))
pl.agregar_mueble(MuebleEnPlanta("cama", 1.5, 2.0, 5.0, 3.0, 0, (230, 140, 150)))
pl.agregar_mueble(MuebleEnPlanta("velador", 0.45, 0.45, 4.05, 4.0, 0, (180, 140, 100)))
pl.agregar_mueble(MuebleEnPlanta("ropero", 1.6, 0.6, 5.0, 0.45, 0, (160, 130, 100)))
fb = Framebuffer(440, 340)
pl.dibujar(fb, escala=62, origen=(32, 305))
con_titulo(fb.a_imagen(), "Plano 2D: muros, puertas, ventanas y muebles").save(
    SALIDA / "fig_05_plano.png")
print("Figuras generadas en", SALIDA)
