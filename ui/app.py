"""Interfaz interactiva de Habita3D sobre pygame volcando el Framebuffer propio."""
import sys
import time
import numpy as np

try:
    import pygame
except ImportError as exc:
    raise SystemExit("Instala pygame para ejecutar la interfaz: pip install pygame") from exc

from mathlib import Mat3
from raster2d import Framebuffer, linea, circulo, rellenar_poligono, dibujar_poligono
from plano import Plano, MuebleEnPlanta
from modelado import Bezier, revolucion, arbol_fractal
from escena import Escena


CATALOGO_MUEBLES = [
    ("sofá", 2.0, 0.9, (130, 150, 210)),
    ("mesa", 1.2, 0.8, (205, 170, 120)),
    ("silla", 0.5, 0.5, (185, 145, 95)),
    ("cama", 1.5, 2.0, (230, 140, 150)),
    ("estante", 1.4, 0.45, (160, 130, 100)),
    ("lampara", 0.45, 0.45, (240, 220, 150)),
    ("jarron", 0.40, 0.40, (195, 115, 85)),
    ("planta", 0.45, 0.45, (70, 155, 80)),
]


class App:
    """Ventana principal de Habita3D con pestañas 2D, 3D, Editor de curvas y Fractales."""

    def __init__(self, ancho: int = 1120, alto: int = 700):
        pygame.init()
        pygame.display.set_caption("Habita3D – Diseñador interactivo de interiores 2D y 3D")
        self.ancho = ancho
        self.alto = alto
        self.pantalla = pygame.display.set_mode((ancho, alto))
        self.reloj = pygame.time.Clock()
        self.fuente = pygame.font.SysFont("Arial", 14)
        self.fuente_bold = pygame.font.SysFont("Arial", 15, bold=True)

        # Escena inicial con vivienda de ejemplo
        pl = self._plano_inicial()
        self.escena = Escena(pl, carpeta_texturas="assets/texturas")

        # Estado de interfaz
        self.vista_activa = "Plano 2D"   # "Plano 2D" | "Vista 3D" | "Editor de curvas" | "Fractales"
        self.herramienta = "Mueble"
        self.escala_2d = 68.0
        self.origen_2d = [45.0, 490.0]
        self.mueble_sel: int | None = 0
        self.idx_catalogo = 0
        self.inicio_muro: tuple[float, float] | None = None
        self.mouse_mundo = (0.0, 0.0)
        self.antialiasing_2d = False

        # Estado de editor de curvas Bézier
        self.puntos_bezier = [
            [0.02, 0.0],
            [0.22, 0.05],
            [0.26, 0.25],
            [0.10, 0.42],
            [0.18, 0.58],
        ]
        self.ctrl_bezier_sel: int | None = None

        # Estado de fractales
        self.fractal_profundidad = 4
        self.fractal_angulo = 28.0
        self.fractal_escala = 0.72

        # Métricas de rendimiento
        self.fps_render = 0.0
        self.res_3d = (440, 300)

    @staticmethod
    def _plano_inicial() -> Plano:
        pl = Plano()
        sala = pl.agregar_habitacion(0, 0, 6.0, 4.5)
        div = pl.agregar_muro((3.6, 0), (3.6, 4.5))
        pl.agregar_abertura(sala[0], 0.20, 1.0, "puerta")
        pl.agregar_abertura(sala[2], 0.28, 1.4, "ventana")
        pl.agregar_abertura(sala[2], 0.82, 1.1, "ventana")
        pl.agregar_abertura(div, 0.70, 0.9, "puerta")
        pl.agregar_mueble(MuebleEnPlanta("sofá", 2.0, 0.9, 1.5, 3.7, 0, (130, 150, 210)))
        pl.agregar_mueble(MuebleEnPlanta("mesa", 1.1, 1.1, 1.6, 1.8, 20, (205, 170, 120)))
        pl.agregar_mueble(MuebleEnPlanta("cama", 1.5, 2.0, 5.0, 3.0, 0, (230, 140, 150)))
        pl.agregar_mueble(MuebleEnPlanta("planta", 0.45, 0.45, 0.55, 0.65, 0, (70, 155, 80)))
        pl.agregar_mueble(MuebleEnPlanta("jarron", 0.40, 0.40, 1.6, 1.8, 0, (195, 115, 85)))
        return pl

    def _pantalla_a_mundo_2d(self, px: int, py: int) -> tuple[float, float]:
        lx = px - 180
        ly = py - 84
        mx = (lx - self.origen_2d[0]) / self.escala_2d
        my = (self.origen_2d[1] - ly) / self.escala_2d
        return round(mx, 2), round(my, 2)

    def _volcar_framebuffer(self, fb: Framebuffer, rect_dest: pygame.Rect) -> None:
        # Framebuffer tiene forma (alto, ancho, 3); surfarray espera (ancho, alto, 3)
        arr = np.transpose(fb.datos, (1, 0, 2))
        surf = pygame.surfarray.make_surface(arr)
        if surf.get_size() != rect_dest.size:
            surf = pygame.transform.smoothscale(surf, rect_dest.size)
        self.pantalla.blit(surf, rect_dest.topleft)

    def _renderizar_lienzo_2d(self, w: int, h: int) -> Framebuffer:
        fb = Framebuffer(w, h, fondo=(252, 252, 252))
        self.escena.plano.dibujar(fb, escala=self.escala_2d, origen=tuple(self.origen_2d))

        # Resaltar mueble seleccionado con contorno propio
        if self.mueble_sel is not None and 0 <= self.mueble_sel < len(self.escena.plano.muebles):
            mu = self.escena.plano.muebles[self.mueble_sel]
            a_pant = Mat3.traslacion(*self.origen_2d) @ Mat3.escalamiento(self.escala_2d, -self.escala_2d)
            pts = a_pant.aplicar_puntos(mu.poligono())
            dibujar_poligono(fb, pts, (220, 50, 40), grosor=2)

        # Vista previa de muro en construcción
        if self.herramienta == "Muro" and self.inicio_muro is not None:
            a_pant = Mat3.traslacion(*self.origen_2d) @ Mat3.escalamiento(self.escala_2d, -self.escala_2d)
            p = a_pant.aplicar_puntos([self.inicio_muro, self.mouse_mundo])
            algo = "wu" if self.antialiasing_2d else "bresenham"
            linea(fb, p[0][0], p[0][1], p[1][0], p[1][1], (40, 120, 210), 2, algoritmo=algo)

        return fb

    def _renderizar_editor_curvas(self, w: int, h: int) -> Framebuffer:
        fb = Framebuffer(w, h, fondo=(250, 250, 252))
        # Mitad izquierda: perfil Bézier 2D; mitad derecha: malla 3D por revolución
        eje_x = int(w * 0.22)
        base_y = int(h * 0.82)
        esc = h * 0.95
        linea(fb, eje_x, 30, eje_x, h - 30, (160, 160, 175), 1)
        linea(fb, 30, base_y, int(w * 0.46), base_y, (160, 160, 175), 1)

        pts_pant = [(eje_x + r * esc, base_y - y * esc) for r, y in self.puntos_bezier]
        dibujar_poligono(fb, pts_pant, (180, 180, 195), 1, cerrar=False)

        curva = Bezier(self.puntos_bezier)
        muestras = curva.muestrear(32)
        curva_pant = [(eje_x + r * esc, base_y - y * esc) for r, y in muestras]
        dibujar_poligono(fb, curva_pant, (200, 75, 45), 2, cerrar=False)

        for px, py in pts_pant:
            circulo(fb, px, py, 6, (30, 80, 170), 1, relleno=(90, 160, 240))

        # Previsualización 3D a la derecha
        esc_rev = Escena(Plano())
        esc_rev.camara.mirar_a((0.0, 0.45, 1.15), (0.0, 0.30, 0.0))
        esc_rev.luz.posicion.x, esc_rev.luz.posicion.y, esc_rev.luz.posicion.z = 0.8, 1.0, 1.2
        esc_rev.agregar_malla_extra(revolucion(curva, segmentos_radiales=24, pasos_perfil=20))
        fb_3d = esc_rev.renderizar_3d(w // 2, h)
        fb.datos[:, w // 2 : w // 2 + fb_3d.ancho] = fb_3d.datos
        return fb

    def _renderizar_vista_fractales(self, w: int, h: int) -> Framebuffer:
        esc_f = Escena(Plano())
        esc_f.camara.mirar_a((0.0, 0.65, 1.55), (0.0, 0.50, 0.0))
        esc_f.luz.posicion.x, esc_f.luz.posicion.y, esc_f.luz.posicion.z = 1.0, 1.5, 1.5
        esc_f.agregar_malla_extra(
            arbol_fractal(
                profundidad=self.fractal_profundidad,
                angulo_grados=self.fractal_angulo,
                factor_escala=self.fractal_escala,
            )
        )
        return esc_f.renderizar_3d(w // 2, h // 2)

    def _dibujar_ui(self) -> None:
        self.pantalla.fill((236, 240, 245))
        # Barra superior
        pygame.draw.rect(self.pantalla, (36, 74, 118), (0, 0, self.ancho, 46))
        self.pantalla.blit(self.fuente_bold.render("Habita3D – Diseñador Interactivo 2D y 3D", True, (255, 255, 255)), (16, 14))

        # Pestañas de vista
        vistas = ["Plano 2D", "Vista 3D", "Editor de curvas", "Fractales"]
        for i, v in enumerate(vistas):
            rx = 180 + i * 165
            activo = self.vista_activa == v
            col = (255, 255, 255) if activo else (210, 222, 236)
            pygame.draw.rect(self.pantalla, col, (rx, 50, 160, 32), border_top_left_radius=6, border_top_right_radius=6)
            self.pantalla.blit(self.fuente_bold.render(v, True, (25, 45, 75)), (rx + 18, 58))

        # Panel izquierdo: Herramientas
        pygame.draw.rect(self.pantalla, (244, 246, 249), (0, 46, 175, self.alto - 74))
        self.pantalla.blit(self.fuente_bold.render("Herramientas", True, (30, 45, 70)), (16, 56))
        herramientas = ["Muro", "Puerta", "Ventana", "Mueble", "Planta fractal"]
        for idx, h_nom in enumerate(herramientas):
            ry = 86 + idx * 38
            sel = self.herramienta == h_nom
            pygame.draw.rect(self.pantalla, (52, 110, 175) if sel else (225, 232, 240), (14, ry, 146, 30), border_radius=4)
            self.pantalla.blit(self.fuente.render(h_nom, True, (255, 255, 255) if sel else (30, 40, 55)), (24, ry + 7))

        # Panel derecho: Propiedades y estados
        px = self.ancho - 225
        pygame.draw.rect(self.pantalla, (244, 246, 249), (px, 46, 225, self.alto - 74))
        self.pantalla.blit(self.fuente_bold.render("Propiedades y Atajos", True, (30, 45, 70)), (px + 14, 56))

        cat_nom = CATALOGO_MUEBLES[self.idx_catalogo][0]
        props = [
            f"Catálogo [Tab]: {cat_nom}",
            f"Proyección [P]: {self.escena.proyeccion}",
            f"Sombreado [G]: {self.escena.sombreado}",
            f"Iluminación [L]: {'ON' if self.escena.iluminacion_activa else 'OFF'}",
            f"Texturas [T]: {'ON' if self.escena.texturas_activas else 'OFF'}",
            f"Alambre [K]: {'ON' if self.escena.modo_alambre else 'OFF'}",
            f"Antialiasing [U]: {'Wu' if self.antialiasing_2d else 'OFF'}",
            f"Fractal nivel [+/-]: {self.fractal_profundidad}",
            "Rotar mueble: Q / E",
            "Cámara 3D: W A S D + Mouse",
        ]
        for i, txt in enumerate(props):
            self.pantalla.blit(self.fuente.render(txt, True, (40, 50, 65)), (px + 14, 90 + i * 26))

        # Lienzo central
        rect_lienzo = pygame.Rect(180, 84, self.ancho - 410, self.alto - 116)
        t0 = time.perf_counter()
        if self.vista_activa == "Plano 2D":
            fb = self._renderizar_lienzo_2d(rect_lienzo.width, rect_lienzo.height)
        elif self.vista_activa == "Vista 3D":
            fb = self.escena.renderizar_3d(*self.res_3d)
        elif self.vista_activa == "Editor de curvas":
            fb = self._renderizar_editor_curvas(rect_lienzo.width, rect_lienzo.height)
        else:
            fb = self._renderizar_vista_fractales(rect_lienzo.width, rect_lienzo.height)
        dt = max(time.perf_counter() - t0, 1e-6)
        self.fps_render = 1.0 / dt
        self._volcar_framebuffer(fb, rect_lienzo)

        # Barra inferior de estado
        pygame.draw.rect(self.pantalla, (224, 230, 238), (0, self.alto - 28, self.ancho, 28))
        estado = (
            f"x: {self.mouse_mundo[0]:.2f} m  y: {self.mouse_mundo[1]:.2f} m  |  "
            f"Triángulos 3D: {self.escena.total_triangulos()}  |  "
            f"Render software: {self.fps_render:.1f} FPS"
        )
        self.pantalla.blit(self.fuente.render(estado, True, (30, 40, 55)), (16, self.alto - 22))
        pygame.display.flip()

    def ejecutar(self) -> None:
        arrastrando_3d = False
        ultimo_mouse = (0, 0)
        corriendo = True

        while corriendo:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    corriendo = False

                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        corriendo = False
                    elif ev.key == pygame.K_p:
                        self.escena.proyeccion = "ortografica" if self.escena.proyeccion == "perspectiva" else "perspectiva"
                    elif ev.key == pygame.K_l:
                        self.escena.iluminacion_activa = not self.escena.iluminacion_activa
                    elif ev.key == pygame.K_t:
                        self.escena.texturas_activas = not self.escena.texturas_activas
                    elif ev.key == pygame.K_g:
                        self.escena.sombreado = "plano" if self.escena.sombreado == "gouraud" else "gouraud"
                    elif ev.key == pygame.K_k:
                        self.escena.modo_alambre = not self.escena.modo_alambre
                    elif ev.key == pygame.K_u:
                        self.antialiasing_2d = not self.antialiasing_2d
                    elif ev.key == pygame.K_TAB:
                        self.idx_catalogo = (self.idx_catalogo + 1) % len(CATALOGO_MUEBLES)
                    elif ev.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                        self.fractal_profundidad = min(6, self.fractal_profundidad + 1)
                    elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.fractal_profundidad = max(1, self.fractal_profundidad - 1)
                    elif ev.key in (pygame.K_q, pygame.K_e) and self.mueble_sel is not None:
                        delta = 15.0 if ev.key == pygame.K_q else -15.0
                        self.escena.plano.muebles[self.mueble_sel].angulo += delta
                        self.escena.reconstruir_3d()

                elif ev.type == pygame.MOUSEWHEEL and self.vista_activa == "Plano 2D":
                    self.escala_2d = max(25.0, min(160.0, self.escala_2d + ev.y * 5.0))

                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    mx, my = ev.pos
                    # Cambio de pestañas
                    if 50 <= my <= 82:
                        vistas = ["Plano 2D", "Vista 3D", "Editor de curvas", "Fractales"]
                        for i, v in enumerate(vistas):
                            if 180 + i * 165 <= mx <= 340 + i * 165:
                                self.vista_activa = v
                    # Selección de herramienta izquierda
                    elif mx < 175 and 86 <= my <= 276:
                        idx = (my - 86) // 38
                        herramientas = ["Muro", "Puerta", "Ventana", "Mueble", "Planta fractal"]
                        if 0 <= idx < len(herramientas):
                            self.herramienta = herramientas[idx]
                    # Interacción en lienzo
                    elif 180 <= mx <= self.ancho - 230 and 84 <= my <= self.alto - 32:
                        if self.vista_activa == "Vista 3D":
                            arrastrando_3d = True
                            ultimo_mouse = (mx, my)
                        elif self.vista_activa == "Plano 2D":
                            wx, wy = self._pantalla_a_mundo_2d(mx, my)
                            if self.herramienta == "Muro":
                                if self.inicio_muro is None:
                                    self.inicio_muro = (wx, wy)
                                else:
                                    self.escena.plano.agregar_muro(self.inicio_muro, (wx, wy))
                                    self.inicio_muro = None
                                    self.escena.reconstruir_3d()
                            elif self.herramienta in ("Mueble", "Planta fractal"):
                                if self.herramienta == "Planta fractal":
                                    nom, a, l, col = ("planta", 0.45, 0.45, (70, 155, 80))
                                else:
                                    nom, a, l, col = CATALOGO_MUEBLES[self.idx_catalogo]
                                self.escena.plano.agregar_mueble(MuebleEnPlanta(nom, a, l, wx, wy, 0.0, col))
                                self.mueble_sel = len(self.escena.plano.muebles) - 1
                                self.escena.reconstruir_3d()

                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    arrastrando_3d = False

                elif ev.type == pygame.MOUSEMOTION:
                    mx, my = ev.pos
                    self.mouse_mundo = self._pantalla_a_mundo_2d(mx, my)
                    if arrastrando_3d and self.vista_activa == "Vista 3D":
                        dx = mx - ultimo_mouse[0]
                        dy = my - ultimo_mouse[1]
                        ultimo_mouse = (mx, my)
                        self.escena.camara.rotar_vista(dx * 0.35, -dy * 0.35)

            # Movimiento continuo W A S D en Vista 3D
            teclas = pygame.key.get_pressed()
            if self.vista_activa == "Vista 3D":
                paso = 0.12
                if teclas[pygame.K_w]:
                    self.escena.camara.mover(adelante=paso)
                if teclas[pygame.K_s]:
                    self.escena.camara.mover(adelante=-paso)
                if teclas[pygame.K_a]:
                    self.escena.camara.mover(derecha=-paso)
                if teclas[pygame.K_d]:
                    self.escena.camara.mover(derecha=paso)

            self._dibujar_ui()
            self.reloj.tick(30)

        pygame.quit()
        sys.exit(0)
