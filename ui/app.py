"""Interfaz interactiva de Habita3D sobre pygame volcando el Framebuffer propio."""
import pathlib
import sys
import time
import numpy as np

try:
    import pygame
except ImportError as exc:
    raise SystemExit("Instala pygame para ejecutar la interfaz: pip install pygame") from exc

from mathlib import Mat3
from raster2d import Framebuffer, linea, circulo, rellenar_poligono, dibujar_poligono
from plano import Plano, MuebleEnPlanta, Abertura, Figura
from modelado import Bezier, revolucion, arbol_fractal
from modelado.csg import caja as csg_caja, esfera as csg_esfera, cilindro as csg_cilindro, HojaCSG, OperacionCSG
from escena import Escena


CATALOGO_MUEBLES = [
    ("sofá", 2.0, 0.9, (130, 150, 210)),
    ("mesa", 1.2, 0.8, (205, 170, 120)),
    ("silla", 0.5, 0.5, (185, 145, 95)),
    ("cama", 1.5, 2.0, (230, 140, 150)),
    ("estante", 1.4, 0.45, (160, 130, 100)),
    ("velador", 0.5, 0.45, (175, 135, 95)),
    ("lampara", 0.45, 0.45, (240, 220, 150)),
    ("jarron", 0.40, 0.40, (195, 115, 85)),
    ("planta", 0.45, 0.45, (70, 155, 80)),
]

# Carpeta de los mapas de bits usados como texturas, relativa al proyecto (no a la carpeta
# desde la que se ejecuta main.py). Toda imagen PNG/JPG/BMP que se ponga aquí se carga.
CARPETA_TEXTURAS = pathlib.Path(__file__).resolve().parent.parent / "assets" / "texturas"

HERRAMIENTAS_DISENO = ["Seleccionar", "Muro", "Puerta", "Ventana", "Mueble", "Planta fractal", "Borrar"]
HERRAMIENTAS_DIBUJO = ["Punto", "Línea", "Círculo", "Polígono"]     # primitivas libres (módulo 1)
HERRAMIENTAS = HERRAMIENTAS_DISENO + HERRAMIENTAS_DIBUJO
Y_HERRAMIENTAS, ALTO_BOTON = 86, 34
NOMBRES_ELEMENTO = {"mueble": "mueble", "abertura": "puerta/ventana", "muro": "muro", "figura": "figura"}
NOMBRES_ALGORITMO = {"bresenham": "Bresenham", "dda": "DDA", "wu": "Wu (antialiasing)"}
# Nombre visible de cada figura y concordancia del mensaje ("línea agregada", "punto agregado")
NOMBRES_FIGURA = {"punto": ("Punto", "agregado"), "linea": ("Línea", "agregada"),
                  "circulo": ("Círculo", "agregado"), "poligono": ("Polígono", "agregado")}
# Paleta del editor 2D: clic = color de trazo; Shift+clic o clic derecho = color de relleno
PALETA = [(30, 30, 30), (200, 40, 40), (230, 140, 30), (230, 200, 40),
          (50, 150, 70), (40, 110, 200), (120, 70, 170), (255, 255, 255)]
Y_PALETA = Y_HERRAMIENTAS + len(HERRAMIENTAS) * ALTO_BOTON + 24
LADO_MUESTRA, PASO_MUESTRA = 30, 37
RECT_SIN_RELLENO = (14, Y_PALETA + 58, 146, 22)
RADIO_CIERRE = 9          # píxeles: un clic tan cerca del primer vértice cierra el polígono
RADIO_PUNTO_CONTROL = 6   # píxeles: tamaño de los puntos de control de Bézier en el editor
VISTAS = ["Plano 2D", "Vista 3D", "Editor de curvas", "Fractales", "CSG"]
# Panel de luz de la Vista 3D: (clave, etiqueta). "y" es la coordenada del plano (Z = −y en 3D)
CONTROLES_LUZ = [("x", "Posición X"), ("y", "Posición Y"), ("altura", "Altura"),
                 ("intensidad", "Intensidad"), ("ambiente", "Ambiente")]
X_BARRA, ANCHO_BARRA, ALTO_FILA_LUZ = 104, 110, 24
ANCHO_PESTANA, PASO_PESTANA = 136, 140
# Pestaña CSG: (clave, texto del botón). "diferencia_ba" es B − A.
OPERACIONES_CSG = [("ninguna", "A y B"), ("union", "A ∪ B"), ("interseccion", "A ∩ B"),
                   ("diferencia", "A − B"), ("diferencia_ba", "B − A")]
COLOR_CSG_A, COLOR_CSG_B = (215, 95, 70), (70, 125, 215)
# Ángulo de cámara (azimut, elevación) desde el que mejor se ve cada operación: en A ∩ B y en
# B − A las caras que aporta el otro sólido quedan del lado opuesto al de A ∪ B y A − B
VISTA_CSG = {"ninguna": (40.0, 28.0), "union": (40.0, 28.0), "interseccion": (220.0, 28.0),
             "diferencia": (40.0, 28.0), "diferencia_ba": (220.0, 28.0)}
ESCALA_MIN, ESCALA_MAX = 0.25, 4.0    # límites del escalamiento de un mueble en cada eje
RADIO_MANIJA = 7                      # píxeles: tolerancia para tomar una esquina con el mouse
# Ancho inicial y límites (m) de los vanos que se colocan con las herramientas Puerta y Ventana
ANCHO_ABERTURA = {"Puerta": 0.9, "Ventana": 1.2}
LIMITES_ANCHO = {"Puerta": (0.6, 2.0), "Ventana": (0.4, 3.0)}


def escala_proporcional(sx: float, sy: float, k: float) -> tuple[float, float]:
    """Multiplica (sx, sy) por k conservando la proporción y sin salir de [ESCALA_MIN, ESCALA_MAX]."""
    k = max(ESCALA_MIN / min(sx, sy), min(ESCALA_MAX / max(sx, sy), k))
    return round(float(sx * k), 2), round(float(sy * k), 2)


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
        # Arial no tiene el símbolo ∪: para textos con símbolos de conjuntos (CSG) se usa
        # la primera fuente disponible que sí lo dibuja (Windows: Segoe UI Symbol / Cambria Math)
        self.fuente_simbolos = pygame.font.SysFont("segoeuisymbol,cambriamath,dejavusans", 15)

        # Escena inicial con vivienda de ejemplo
        pl = self._plano_inicial()
        self.escena = Escena(pl, carpeta_texturas=CARPETA_TEXTURAS)

        # Estado de interfaz
        self.vista_activa = "Plano 2D"   # "Plano 2D" | "Vista 3D" | "Editor de curvas" | "Fractales"
        self.herramienta = "Seleccionar"
        self.escala_2d = 68.0
        self.origen_2d = [45.0, 490.0]
        self.mueble_sel: int | None = 0
        self.idx_catalogo = 0
        self.inicio_muro: tuple[float, float] | None = None
        self.mouse_mundo = (0.0, 0.0)
        self.antialiasing_2d = False
        # Atributos del editor 2D (módulo 1) y figura que se está dibujando (vértices en metros)
        self.color_trazo = PALETA[0]
        self.color_relleno: tuple | None = None
        self.grosor_trazo = 2
        self.algoritmo_lineas = "bresenham"     # "bresenham" | "dda"; U activa Wu (antialiasing)
        self.figura_en_curso: list[tuple[float, float]] = []
        self.mensaje = ""   # aviso breve en la barra de estado (p. ej. qué se borró)
        # Arrastre de un mueble con la herramienta Seleccionar: desfase entre el centro
        # del mueble y el punto donde se hizo clic, para que no "salte" al cursor.
        self.desfase_arrastre: tuple[float, float] | None = None
        self.posicion_antes_arrastre: tuple[float, float] | None = None
        # Escalamiento con las manijas de las esquinas: guarda la inversa de T · R del
        # mueble (para llevar el mouse a sus ejes locales) y la escala inicial.
        self.escalado: dict | None = None
        self.ancho_abertura = dict(ANCHO_ABERTURA)   # ancho del próximo vano por herramienta

        # Estado de editor de curvas Bézier
        self.puntos_bezier = [
            [0.02, 0.0],
            [0.22, 0.05],
            [0.26, 0.25],
            [0.10, 0.42],
            [0.18, 0.58],
        ]
        self.ctrl_bezier_sel: int | None = None    # punto de control que se está arrastrando
        self.t_bezier = 0.5                        # parámetro de la construcción de De Casteljau
        self._escena_curvas: Escena | None = None   # escena de la vista previa 3D (se reutiliza)
        self._cache_curvas: tuple | None = None     # (clave de la curva, framebuffer 3D)

        # Panel de luz (Vista 3D): deslizador que se está arrastrando y rectángulos clicables
        self.panel_luz_visible = True
        self.deslizador_luz: str | None = None
        self._deslizadores_luz: list = []           # (rect de la barra, clave)
        self._rect_panel_luz: pygame.Rect | None = None
        self._boton_lampara: pygame.Rect | None = None
        self.idx_lampara = -1

        # Pestaña CSG: operación elegida, forma de B, órbita de la cámara y cachés
        self.csg_operacion = "union"
        self.csg_b = "esfera"
        self.csg_azimut, self.csg_elevacion = 40.0, 28.0
        self._csg_resultados: dict = {}             # (operación, B) → (mallas, árbol, ms, triángulos)
        self._escena_csg: Escena | None = None
        self._cache_csg: tuple | None = None        # (clave, framebuffer)
        self._botones_csg: list = []                # (rect, tipo, valor) dibujados en el lienzo

        # Estado de fractales
        self.fractal_profundidad = 4
        self.fractal_angulo = 28.0
        self.fractal_escala = 0.72

        # Métricas de rendimiento
        self.fps_render = 0.0
        # Alto interno del render 3D; el ancho se calcula con la proporción del lienzo
        # para que la imagen no se deforme al escalarla a la ventana.
        self.alto_3d = 320
        # Resolución dinámica: mientras la cámara se mueve se renderiza a menor resolución
        # (se nota fluido) y al detenerse se vuelve a la completa (se ve nítido).
        self.alto_3d_movimiento = 200
        self.camara_moviendose = False
        self.velocidad_camara = 2.4          # m/s: el paso por cuadro depende del tiempo real
        self._cache_3d: tuple | None = None   # (clave, framebuffer) del último render 3D

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

    # ------------------------------------------------------------ editor 2D (módulo 1)
    @property
    def algoritmo_efectivo(self) -> str:
        """Algoritmo con que se trazarán las líneas: Wu si el antialiasing está activo."""
        return "wu" if self.antialiasing_2d else self.algoritmo_lineas

    def _nueva_figura(self, tipo: str, puntos: list, radio: float = 0.0) -> Figura:
        con_relleno = tipo in ("circulo", "poligono")
        return Figura(tipo, list(puntos), self.color_trazo, self.color_relleno if con_relleno else None,
                      self.grosor_trazo, self.algoritmo_efectivo, radio)

    def _agregar_figura(self, figura: Figura) -> None:
        self.escena.plano.agregar_figura(figura)
        self.figura_en_curso = []
        nombre, agregado = NOMBRES_FIGURA[figura.tipo]
        if figura.tipo in ("linea", "poligono"):
            detalle = f"{NOMBRES_ALGORITMO[figura.algoritmo]}, {figura.grosor} px"
        elif figura.tipo == "circulo":
            detalle = f"punto medio, {figura.grosor} px"     # la circunferencia usa su propio algoritmo
        else:
            detalle = f"{figura.grosor} px"
        if figura.relleno is not None:
            detalle += ", con relleno"
        self.mensaje = f"{nombre} {agregado} ({detalle})"

    def _clic_dibujo(self, wx: float, wy: float, mx: int, my: int) -> None:
        """Clic en el lienzo con una herramienta de dibujo libre."""
        h, p = self.herramienta, (wx, wy)
        en_curso = self.figura_en_curso
        if h == "Punto":
            self._agregar_figura(self._nueva_figura("punto", [p]))
        elif h == "Línea":
            if not en_curso:
                self.figura_en_curso = [p]
                self.mensaje = "Línea: haz clic en el punto final (Esc cancela)"
            else:
                self._agregar_figura(self._nueva_figura("linea", [en_curso[0], p]))
        elif h == "Círculo":
            if not en_curso:
                self.figura_en_curso = [p]
                self.mensaje = "Círculo: haz clic para fijar el radio (Esc cancela)"
            else:
                radio = float(np.hypot(p[0] - en_curso[0][0], p[1] - en_curso[0][1]))
                if radio < 0.02:
                    self.mensaje = "El radio es demasiado pequeño"
                    return
                self._agregar_figura(self._nueva_figura("circulo", [en_curso[0]], radio))
        elif h == "Polígono":
            if len(en_curso) >= 3:
                x0, y0 = self._matriz_pantalla().aplicar_puntos([en_curso[0]])[0]
                if np.hypot(x0 - (mx - 180), y0 - (my - 84)) <= RADIO_CIERRE:
                    self._cerrar_poligono()
                    return
            en_curso.append(p)
            self.mensaje = (f"Polígono: {len(en_curso)} vértices; clic en el primero o Enter para cerrar")

    def _cerrar_poligono(self) -> None:
        if self.herramienta == "Polígono" and len(self.figura_en_curso) >= 3:
            self._agregar_figura(self._nueva_figura("poligono", self.figura_en_curso))
        elif self.herramienta == "Polígono":
            self.mensaje = "Un polígono necesita al menos 3 vértices"

    def _cancelar_dibujo(self) -> bool:
        """Cancela un muro o una figura a medio dibujar; devuelve True si había algo en curso."""
        habia = self.inicio_muro is not None or bool(self.figura_en_curso)
        self.inicio_muro = None
        self.figura_en_curso = []
        return habia

    def _en_paleta(self, mx: int, my: int) -> bool:
        x, y, w, h = RECT_SIN_RELLENO
        return mx < 175 and Y_PALETA <= my <= y + h

    def _clic_paleta(self, mx: int, my: int, relleno: bool) -> None:
        x, y, w, h = RECT_SIN_RELLENO
        if x <= mx <= x + w and y <= my <= y + h:
            self.color_relleno = None
            self.mensaje = "Sin relleno"
            return
        col, fila = (mx - 14) // PASO_MUESTRA, (my - Y_PALETA) // 28
        if (0 <= col < 4 and 0 <= fila < 2 and (mx - 14) % PASO_MUESTRA <= LADO_MUESTRA
                and (my - Y_PALETA) % 28 <= 22):
            color = PALETA[fila * 4 + col]
            if relleno:
                self.color_relleno = color
                self.mensaje = f"Color de relleno: {color}"
            else:
                self.color_trazo = color
                self.mensaje = f"Color de trazo: {color}"

    def _borrar(self, elemento) -> None:
        """Borra del plano el elemento (tipo, índice) y regenera el modelo 3D."""
        if elemento is None:
            self.mensaje = "No hay ningún elemento en ese punto"
            return
        tipo, i = elemento
        if tipo == "mueble":
            nombre = self.escena.plano.muebles[i].nombre
        elif tipo == "abertura":
            nombre = self.escena.plano.aberturas[i].tipo
        elif tipo == "figura":
            nombre = NOMBRES_FIGURA[self.escena.plano.figuras[i].tipo][0].lower()
        else:
            nombre = NOMBRES_ELEMENTO[tipo]
        self.escena.plano.eliminar(tipo, i)
        self.desfase_arrastre = None            # cancela un arrastre o escalamiento en curso
        self.posicion_antes_arrastre = None
        self.escalado = None
        # Mantener la selección apuntando al mismo mueble tras el borrado
        if tipo == "mueble" and self.mueble_sel is not None:
            if self.mueble_sel == i:
                self.mueble_sel = None
            elif self.mueble_sel > i:
                self.mueble_sel -= 1
        self.escena.reconstruir_3d()
        self.mensaje = f"Borrado: {nombre}"

    def _mueble_seleccionado(self):
        muebles = self.escena.plano.muebles
        if self.mueble_sel is not None and 0 <= self.mueble_sel < len(muebles):
            return muebles[self.mueble_sel]
        return None

    def _abertura_candidata(self, x: float, y: float) -> dict | None:
        """Vano que se colocaría con la herramienta Puerta/Ventana en el punto (x, y) del plano.
        Devuelve None si no hay un muro ahí; si lo hay, indica si el vano es válido y por qué no."""
        if self.herramienta not in ANCHO_ABERTURA:
            return None
        plano = self.escena.plano
        i = plano.muro_en(x, y)
        if i is None:
            return None
        tipo = self.herramienta.lower()
        ancho = self.ancho_abertura[self.herramienta]
        pos = plano.posicion_en_muro(i, x, y, ancho)
        if pos is None:
            return {"muro": i, "pos": None, "ancho": ancho, "tipo": tipo, "valida": False,
                    "motivo": f"El muro es muy corto para una {tipo} de {ancho:.2f} m"}
        ajustada = plano.ajustar_junto_a_vecinos(i, pos, ancho)   # imán: se pega a un vano vecino
        if ajustada is None:
            return {"muro": i, "pos": pos, "ancho": ancho, "tipo": tipo, "valida": False,
                    "motivo": "No cabe ahí: se superpone con otra puerta o ventana de ese muro"}
        pegada = abs(ajustada - pos) * plano.muros[i].largo() > 1e-6
        return {"muro": i, "pos": ajustada, "ancho": ancho, "tipo": tipo, "valida": True,
                "motivo": "", "pegada": pegada}

    def _colocar_abertura(self, x: float, y: float) -> None:
        cand = self._abertura_candidata(x, y)
        if cand is None:
            self.mensaje = f"Haz clic sobre un muro para colocar la {self.herramienta.lower()}"
        elif not cand["valida"]:
            self.mensaje = cand["motivo"]
        else:
            # Sin redondear la posición: dos vanos pegados deben tocarse exactamente
            self.escena.plano.agregar_abertura(cand["muro"], cand["pos"], cand["ancho"], cand["tipo"])
            self.escena.reconstruir_3d()
            junto = " junto al vano vecino" if cand.get("pegada") else ""
            self.mensaje = f"{self.herramienta} de {cand['ancho']:.2f} m agregada{junto}"

    def _clave_3d(self, ancho: int, alto: int) -> tuple:
        """Todo lo que cambia la imagen 3D; si la clave no cambia, no hace falta renderizar."""
        e, c, l = self.escena, self.escena.camara, self.escena.luz.posicion
        return (ancho, alto, tuple(c.eye), tuple(c.target), (l.x, l.y, l.z),
                e.proyeccion, e.sombreado, e.iluminacion_activa, e.texturas_activas,
                e.modo_alambre, id(e.mallas), len(e.mallas_extra), id(e.texturas), e.luz.il, e.luz.ia)

    def _matriz_pantalla(self) -> Mat3:
        """Metros del plano → píxeles del lienzo: T(origen) · S(escala, −escala)."""
        return Mat3.traslacion(*self.origen_2d) @ Mat3.escalamiento(self.escala_2d, -self.escala_2d)

    def _manija_en(self, mx: int, my: int) -> bool:
        """¿El clic (en coordenadas de ventana) cae sobre una esquina del mueble seleccionado?"""
        mu = self._mueble_seleccionado()
        if self.herramienta != "Seleccionar" or mu is None:
            return False
        lx, ly = mx - 180, my - 84
        esquinas = self._matriz_pantalla().aplicar_puntos(mu.poligono())
        return any(abs(px - lx) <= RADIO_MANIJA and abs(py - ly) <= RADIO_MANIJA for px, py in esquinas)

    def _iniciar_escalado(self) -> None:
        mu = self._mueble_seleccionado()
        # Inversa de T · R: lleva un punto del plano a los ejes locales (girados) del mueble
        inversa = (Mat3.traslacion(mu.x, mu.y) @ Mat3.rotacion(mu.angulo)).inversa()
        self.escalado = {
            "inversa": inversa,
            "sx0": mu.escala_x,
            "sy0": mu.escala_y,
            "esquina0": float(np.hypot(mu.ancho / 2 * mu.escala_x, mu.largo / 2 * mu.escala_y)),
        }

    def _actualizar_escalado(self, proporcional: bool) -> None:
        """La esquina sigue al mouse: en ejes locales, |x| = (ancho/2)·sx y |y| = (largo/2)·sy."""
        mu = self._mueble_seleccionado()
        e = self.escalado
        lx, ly = e["inversa"].aplicar_puntos([self.mouse_mundo])[0]
        if proporcional:
            k = float(np.hypot(lx, ly)) / max(e["esquina0"], 1e-6)
            mu.escala_x, mu.escala_y = escala_proporcional(e["sx0"], e["sy0"], k)
        else:
            limitar = lambda s: round(float(max(ESCALA_MIN, min(ESCALA_MAX, s))), 2)
            mu.escala_x = limitar(abs(lx) / (mu.ancho / 2))
            mu.escala_y = limitar(abs(ly) / (mu.largo / 2))

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
            # Manijas de escalamiento en las cuatro esquinas
            if self.herramienta == "Seleccionar":
                r = 4
                for px, py in pts:
                    cuadro = [(px - r, py - r), (px + r, py - r), (px + r, py + r), (px - r, py + r)]
                    rellenar_poligono(fb, cuadro, (255, 255, 255), borde=(220, 50, 40), grosor=1)

        # Herramienta Seleccionar: contorno azul en el mueble que se tomaría con un clic
        if self.herramienta == "Seleccionar" and self.desfase_arrastre is None and self.escalado is None:
            i = self.escena.plano.mueble_en(*self.mouse_mundo)
            if i is not None and i != self.mueble_sel:
                a_pant = Mat3.traslacion(*self.origen_2d) @ Mat3.escalamiento(self.escala_2d, -self.escala_2d)
                pts = a_pant.aplicar_puntos(self.escena.plano.muebles[i].poligono())
                dibujar_poligono(fb, pts, (40, 120, 210), grosor=2)

        # Herramientas Puerta/Ventana: vista previa del vano sobre el muro bajo el cursor
        # (verde si se puede colocar, rojo si no cabe o choca con otro vano)
        cand = self._abertura_candidata(*self.mouse_mundo)
        if cand is not None and cand["pos"] is not None:
            vano = Abertura(cand["muro"], cand["pos"], cand["ancho"], cand["tipo"])
            pts = self._matriz_pantalla().aplicar_puntos(self.escena.plano.poligono_abertura(vano))
            relleno, borde = ((150, 215, 150), (30, 140, 60)) if cand["valida"] else ((240, 160, 160), (200, 40, 40))
            rellenar_poligono(fb, pts, relleno, borde=borde, grosor=2)
        elif cand is not None:
            pts = self._matriz_pantalla().aplicar_puntos(self.escena.plano.muros[cand["muro"]].poligono())
            dibujar_poligono(fb, pts, (200, 40, 40), grosor=2)

        # Herramienta Borrar: resaltar en rojo el elemento que se borraría con un clic
        if self.herramienta == "Borrar":
            bajo_cursor = self.escena.plano.elemento_en(*self.mouse_mundo)
            if bajo_cursor is not None:
                a_pant = Mat3.traslacion(*self.origen_2d) @ Mat3.escalamiento(self.escala_2d, -self.escala_2d)
                pts = a_pant.aplicar_puntos(self.escena.plano.poligono_elemento(*bajo_cursor))
                dibujar_poligono(fb, pts, (225, 30, 30), grosor=3)

        # Vista previa de la figura libre que se está dibujando (mismos algoritmos y atributos)
        if self.herramienta in HERRAMIENTAS_DIBUJO and self.figura_en_curso:
            a_pant = self._matriz_pantalla()
            pts = self.figura_en_curso
            if self.herramienta == "Línea":
                self._nueva_figura("linea", [pts[0], self.mouse_mundo]).dibujar(fb, a_pant)
            elif self.herramienta == "Círculo":
                r = float(np.hypot(self.mouse_mundo[0] - pts[0][0], self.mouse_mundo[1] - pts[0][1]))
                if r > 1e-3:
                    self._nueva_figura("circulo", [pts[0]], r).dibujar(fb, a_pant)
            elif self.herramienta == "Polígono":
                p_pant = a_pant.aplicar_puntos(pts + [self.mouse_mundo])
                dibujar_poligono(fb, p_pant, self.color_trazo, self.grosor_trazo, cerrar=False,
                                 algoritmo=self.algoritmo_efectivo)
                if len(pts) >= 3:      # aro en el primer vértice: clic ahí para cerrar
                    circulo(fb, p_pant[0][0], p_pant[0][1], RADIO_CIERRE, (40, 120, 210), 2)

        # Posición de la luz vista en planta (X, −Z): mismo ícono que en la vista 3D
        if self.escena.iluminacion_activa:
            l = self.escena.luz.posicion
            sx, sy = self._matriz_pantalla().aplicar_puntos([(l.x, -l.z)])[0]
            self._dibujar_sol(fb, sx, sy)

        # Vista previa de muro en construcción
        if self.herramienta == "Muro" and self.inicio_muro is not None:
            a_pant = Mat3.traslacion(*self.origen_2d) @ Mat3.escalamiento(self.escala_2d, -self.escala_2d)
            p = a_pant.aplicar_puntos([self.inicio_muro, self.mouse_mundo])
            algo = "wu" if self.antialiasing_2d else "bresenham"
            linea(fb, p[0][0], p[0][1], p[1][0], p[1][1], (40, 120, 210), 2, algoritmo=algo)

        return fb

    # ------------------------------------------------------------ editor de curvas (módulo 5)
    def _rect_lienzo(self) -> pygame.Rect:
        return pygame.Rect(180, 84, self.ancho - 410, self.alto - 116)

    def _geometria_curvas(self) -> tuple[int, int, float]:
        """(eje_x, base_y, escala): el perfil (r, y) en metros se dibuja en (eje_x + r·esc, base_y − y·esc)."""
        r = self._rect_lienzo()
        return int(r.width * 0.22), int(r.height * 0.82), r.height * 0.95

    def _perfil_a_lienzo(self, puntos) -> list[tuple[float, float]]:
        eje_x, base_y, esc = self._geometria_curvas()
        return [(eje_x + r * esc, base_y - y * esc) for r, y in puntos]

    def _punto_bezier_en(self, mx: int, my: int) -> int | None:
        """Índice del punto de control bajo el mouse (coordenadas de ventana), o None."""
        lx, ly = mx - 180, my - 84
        cercanos = [(np.hypot(px - lx, py - ly), i)
                    for i, (px, py) in enumerate(self._perfil_a_lienzo(self.puntos_bezier))]
        d, i = min(cercanos)
        return i if d <= RADIO_PUNTO_CONTROL + 3 else None

    def _mover_punto_bezier(self, mx: int, my: int) -> None:
        """Lleva el punto de control seleccionado a la posición del mouse (radio r ≥ 0)."""
        eje_x, base_y, esc = self._geometria_curvas()
        r = (mx - 180 - eje_x) / esc
        y = (base_y - (my - 84)) / esc
        self.puntos_bezier[self.ctrl_bezier_sel] = [round(min(max(r, 0.0), 0.45), 3),
                                                    round(min(max(y, 0.0), 0.75), 3)]

    def _agregar_o_quitar_punto_bezier(self, mx: int, my: int) -> None:
        """Clic derecho: sobre un punto lo quita; en otro lugar inserta uno nuevo en el tramo
        más cercano del polígono de control (así la curva cambia poco)."""
        i = self._punto_bezier_en(mx, my)
        if i is not None:
            if len(self.puntos_bezier) <= 2:
                self.mensaje = "La curva necesita al menos 2 puntos de control"
            else:
                del self.puntos_bezier[i]
                self.mensaje = f"Punto P{i} quitado (grado {len(self.puntos_bezier) - 1})"
            return
        lx, ly = mx - 180, my - 84
        if lx > self._rect_lienzo().width * 0.48:
            return                                    # la mitad derecha es la vista 3D
        pts = np.array(self._perfil_a_lienzo(self.puntos_bezier))
        mejor, d_min = 1, np.inf
        for k in range(len(pts) - 1):                 # distancia del clic a cada tramo
            a, b = pts[k], pts[k + 1]
            ab = b - a
            t = np.clip(np.dot((lx, ly) - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
            d = np.hypot(*((lx, ly) - (a + t * ab)))
            if d < d_min:
                mejor, d_min = k + 1, d
        self.puntos_bezier.insert(mejor, [0.0, 0.0])
        self.ctrl_bezier_sel = mejor
        self._mover_punto_bezier(mx, my)
        self.ctrl_bezier_sel = None
        self.mensaje = f"Punto agregado como P{mejor} (grado {len(self.puntos_bezier) - 1})"

    def _renderizar_editor_curvas(self, w: int, h: int) -> Framebuffer:
        fb = Framebuffer(w, h, fondo=(250, 250, 252))
        # Mitad izquierda: perfil Bézier 2D; mitad derecha: malla 3D por revolución
        eje_x, base_y, esc = self._geometria_curvas()
        linea(fb, eje_x, 30, eje_x, h - 30, (160, 160, 175), 1)
        linea(fb, 30, base_y, int(w * 0.46), base_y, (160, 160, 175), 1)

        curva = Bezier(self.puntos_bezier)
        pts_pant = self._perfil_a_lienzo(self.puntos_bezier)

        # Construcción de De Casteljau para t: cada nivel interpola el anterior
        niveles = curva.niveles_de_casteljau(self.t_bezier)
        colores_nivel = [(185, 185, 200), (120, 190, 140), (230, 170, 60), (170, 120, 210), (90, 170, 200)]
        for k, nivel in enumerate(niveles[:-1]):
            p = self._perfil_a_lienzo(nivel)
            color = colores_nivel[min(k, len(colores_nivel) - 1)]
            if len(p) > 1:
                dibujar_poligono(fb, p, color, 1, cerrar=False)
            if k > 0:
                for px, py in p:
                    circulo(fb, px, py, 3, color, 1, relleno=color)

        muestras = curva.muestrear(48)
        dibujar_poligono(fb, self._perfil_a_lienzo(muestras), (200, 75, 45), 2, cerrar=False)

        bx, by = self._perfil_a_lienzo(niveles[-1])[0]          # B(t)
        circulo(fb, bx, by, 5, (200, 30, 30), 1, relleno=(235, 60, 50))

        hover = self._punto_bezier_en(*pygame.mouse.get_pos()) if self.ctrl_bezier_sel is None else None
        for i, (px, py) in enumerate(pts_pant):
            activo = i == self.ctrl_bezier_sel or i == hover
            circulo(fb, px, py, RADIO_PUNTO_CONTROL + (2 if activo else 0), (30, 80, 170), 1,
                    relleno=(255, 200, 60) if activo else (90, 160, 240))

        # Previsualización 3D a la derecha: superficie de revolución con grosor (se ve también
        # por dentro). Solo se vuelve a renderizar si cambió la curva.
        clave = (tuple(map(tuple, self.puntos_bezier)), w, h)
        if self._cache_curvas is None or self._cache_curvas[0] != clave:
            if self._escena_curvas is None:
                self._escena_curvas = Escena(Plano())
                self._escena_curvas.camara.mirar_a((0.0, 0.45, 1.15), (0.0, 0.30, 0.0))
                l = self._escena_curvas.luz.posicion
                l.x, l.y, l.z = 0.8, 1.0, 1.2
            self._escena_curvas.mallas_extra = [revolucion(curva, segmentos_radiales=24, pasos_perfil=20,
                                                           grosor=0.01)]
            self._cache_curvas = (clave, self._escena_curvas.renderizar_3d(w // 2, h))
        fb_3d = self._cache_curvas[1]
        fb.datos[:, w // 2 : w // 2 + fb_3d.ancho] = fb_3d.datos
        return fb

    # ------------------------------------------------------------ luz configurable (módulo 9)
    def _rangos_luz(self) -> dict:
        """Rango de cada control: la posición abarca la vivienda con medio metro de margen."""
        muros = self.escena.plano.muros
        xs = [p[0] for m in muros for p in (m.p1, m.p2)] or [0.0, 6.0]
        ys = [p[1] for m in muros for p in (m.p1, m.p2)] or [0.0, 4.5]
        return {"x": (min(xs) - 0.5, max(xs) + 0.5), "y": (min(ys) - 0.5, max(ys) + 0.5),
                "altura": (0.2, 3.5), "intensidad": (0.0, 1.5), "ambiente": (0.0, 0.8)}

    def _valor_luz(self, clave: str) -> float:
        l = self.escena.luz
        return {"x": l.posicion.x, "y": -l.posicion.z, "altura": l.posicion.y,
                "intensidad": l.il, "ambiente": l.ia}[clave]

    def _fijar_luz(self, clave: str, valor: float) -> None:
        minimo, maximo = self._rangos_luz()[clave]
        valor = round(min(maximo, max(minimo, valor)), 2)
        l = self.escena.luz
        if clave == "x":
            l.posicion.x = valor
        elif clave == "y":
            l.posicion.z = -valor
        elif clave == "altura":
            l.posicion.y = valor
        elif clave == "intensidad":
            l.il = valor
        else:
            l.ia = valor

    def _mover_deslizador_luz(self, mx: int) -> None:
        for rect, clave in self._deslizadores_luz:
            if clave == self.deslizador_luz:
                minimo, maximo = self._rangos_luz()[clave]
                t = min(1.0, max(0.0, (mx - rect.x) / rect.width))
                self._fijar_luz(clave, minimo + t * (maximo - minimo))

    def _luz_en_lampara(self) -> None:
        """Pone la luz dentro de la pantalla de la siguiente lámpara del plano."""
        plano = self.escena.plano
        lamparas = [i for i, mu in enumerate(plano.muebles) if "lampara" in mu.nombre or "lámpara" in mu.nombre]
        if not lamparas:
            self.mensaje = "No hay lámparas en el plano: agrega una con la herramienta Mueble (Tab)"
            return
        self.idx_lampara = (self.idx_lampara + 1) % len(lamparas)
        i = lamparas[self.idx_lampara]
        primera_mueble = 1 + len(plano.muros) if plano.muros else 0     # piso + muros van antes
        malla = self.escena.mallas[primera_mueble + i]
        mu = plano.muebles[i]
        l = self.escena.luz.posicion
        l.x, l.y, l.z = mu.x, float(malla.vertices[:, 1].max()) - 0.12, -mu.y
        self.mensaje = f"Luz en la lámpara {self.idx_lampara + 1} de {len(lamparas)} ({mu.x:.2f}, {mu.y:.2f})"

    def _dibujar_panel_luz(self, rect: pygame.Rect) -> None:
        """Panel semitransparente con los deslizadores de la luz (texto y controles de interfaz)."""
        if not self.panel_luz_visible:
            self._deslizadores_luz, self._rect_panel_luz, self._boton_lampara = [], None, None
            return
        x0, y0 = rect.x + 10, rect.y + 10
        alto = 30 + len(CONTROLES_LUZ) * ALTO_FILA_LUZ + 36
        panel = pygame.Surface((262, alto), pygame.SRCALPHA)
        panel.fill((255, 255, 255, 215))
        self.pantalla.blit(panel, (x0, y0))
        self._rect_panel_luz = pygame.Rect(x0, y0, 262, alto)
        self.pantalla.blit(self.fuente_bold.render("Luz puntual  (ocultar: O)", True, (30, 45, 70)), (x0 + 10, y0 + 7))
        rangos = self._rangos_luz()
        self._deslizadores_luz = []
        for k, (clave, etiqueta) in enumerate(CONTROLES_LUZ):
            fy = y0 + 32 + k * ALTO_FILA_LUZ
            self.pantalla.blit(self.fuente.render(etiqueta, True, (40, 50, 65)), (x0 + 10, fy))
            barra = pygame.Rect(x0 + X_BARRA, fy + 7, ANCHO_BARRA, 4)
            pygame.draw.rect(self.pantalla, (190, 198, 210), barra, border_radius=2)
            minimo, maximo = rangos[clave]
            valor = self._valor_luz(clave)
            t = (valor - minimo) / (maximo - minimo)
            activo = clave == self.deslizador_luz
            pygame.draw.circle(self.pantalla, (52, 110, 175) if activo else (90, 140, 200),
                               (int(barra.x + t * barra.width), barra.centery), 7 if activo else 6)
            unidad = " m" if clave in ("x", "y", "altura") else ""
            self.pantalla.blit(self.fuente.render(f"{valor:.2f}{unidad}", True, (40, 50, 65)),
                               (barra.right + 12, fy))
            self._deslizadores_luz.append((barra, clave))
        self._boton_lampara = pygame.Rect(x0 + 10, y0 + alto - 32, 242, 24)
        pygame.draw.rect(self.pantalla, (225, 232, 240), self._boton_lampara, border_radius=4)
        self.pantalla.blit(self.fuente.render("Luz en lámpara (siguiente)", True, (30, 40, 55)),
                           (self._boton_lampara.x + 48, self._boton_lampara.y + 4))

    def _clic_panel_luz(self, mx: int, my: int) -> bool:
        """Clic en el panel de luz: empieza a arrastrar un deslizador o usa el botón."""
        if self._rect_panel_luz is None or not self._rect_panel_luz.collidepoint(mx, my):
            return False
        if self._boton_lampara is not None and self._boton_lampara.collidepoint(mx, my):
            self._luz_en_lampara()
            return True
        for barra, clave in self._deslizadores_luz:
            if barra.inflate(16, 16).collidepoint(mx, my):      # zona de clic más amplia que la barra
                self.deslizador_luz = clave
                self._mover_deslizador_luz(mx)
        return True

    @staticmethod
    def _dibujar_sol(fb: Framebuffer, x: float, y: float, r: int = 6) -> None:
        """Ícono de la luz con los algoritmos propios: círculo de punto medio y rayos (Bresenham)."""
        for k in range(8):
            a = k * np.pi / 4
            linea(fb, x + (r + 3) * np.cos(a), y + (r + 3) * np.sin(a),
                  x + (r + 8) * np.cos(a), y + (r + 8) * np.sin(a), (230, 140, 20), 2)
        circulo(fb, x, y, r, (230, 140, 20), 2, relleno=(255, 215, 60))

    # ------------------------------------------------------------ pestaña CSG (módulo 6)
    def _resultado_csg(self, operacion: str, forma_b: str) -> tuple:
        """Evalúa (con caché) el árbol CSG: A = caja, B = esfera o cilindro."""
        clave = (operacion, forma_b)
        if clave not in self._csg_resultados:
            a = HojaCSG("Caja A", csg_caja(0, 0.5, 0, 1.0, 1.0, 1.0, COLOR_CSG_A))
            if forma_b == "esfera":
                b = HojaCSG("Esfera B", csg_esfera(0.45, 0.85, 0.40, 0.55, COLOR_CSG_B, sectores=24, anillos=16))
            else:
                b = HojaCSG("Cilindro B", csg_cilindro(0.30, -0.15, 0.30, 0.32, 1.3, COLOR_CSG_B, sectores=28))
            t0 = time.perf_counter()
            if operacion == "ninguna":
                mallas = [a.solido.a_malla("A"), b.solido.a_malla("B")]
                texto = f"{a.describir()} y {b.describir()} sin operar"
            else:
                arbol = (OperacionCSG("diferencia", b, a) if operacion == "diferencia_ba"
                         else OperacionCSG(operacion, a, b))
                mallas = [arbol.evaluar().a_malla("resultado_csg")]
                texto = f"Árbol CSG: {arbol.describir()}"
            ms = (time.perf_counter() - t0) * 1000
            self._csg_resultados[clave] = (mallas, texto, ms, sum(len(m.caras) for m in mallas))
        return self._csg_resultados[clave]

    def _renderizar_vista_csg(self, w: int, h: int) -> Framebuffer:
        mallas = self._resultado_csg(self.csg_operacion, self.csg_b)[0]
        clave = (self.csg_operacion, self.csg_b, self.csg_azimut, self.csg_elevacion, w, h)
        if self._cache_csg is None or self._cache_csg[0] != clave:
            if self._escena_csg is None:
                self._escena_csg = Escena(Plano())
                self._escena_csg.sombreado = "plano"        # caras planas: se ven bien los cortes
            e = self._escena_csg
            az, el = np.radians(self.csg_azimut), np.radians(self.csg_elevacion)
            objetivo = np.array([0.15, 0.5, 0.15])
            direccion = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
            ojo = objetivo + 3.2 * direccion
            e.camara.mirar_a(tuple(ojo), tuple(objetivo))
            luz = objetivo + 3.0 * np.array([np.cos(az + 0.6), 0.9, np.sin(az + 0.6)])
            e.luz.posicion.x, e.luz.posicion.y, e.luz.posicion.z = luz
            e.mallas_extra = list(mallas)
            self._cache_csg = (clave, e.renderizar_3d(w, h))
        return self._cache_csg[1]

    def _dibujar_controles_csg(self, rect) -> None:
        """Botones de operación y de forma de B, y el árbol evaluado (texto de la interfaz)."""
        self._botones_csg = []
        x = rect.x + 10
        for clave, texto in OPERACIONES_CSG:
            r = pygame.Rect(x, rect.y + 10, 82, 28)
            activo = clave == self.csg_operacion
            pygame.draw.rect(self.pantalla, (52, 110, 175) if activo else (225, 232, 240), r, border_radius=4)
            self.pantalla.blit(self.fuente_simbolos.render(texto, True, (255, 255, 255) if activo else (30, 40, 55)),
                               (r.x + 20, r.y + 4))
            self._botones_csg.append((r, "operacion", clave))
            x += 88
        x += 12
        for forma in ("esfera", "cilindro"):
            r = pygame.Rect(x, rect.y + 10, 104, 28)
            activo = forma == self.csg_b
            pygame.draw.rect(self.pantalla, (120, 70, 170) if activo else (232, 228, 242), r, border_radius=4)
            self.pantalla.blit(self.fuente.render(f"B = {forma}", True, (255, 255, 255) if activo else (30, 40, 55)),
                               (r.x + 14, r.y + 6))
            self._botones_csg.append((r, "forma", forma))
            x += 110
        _, texto, ms, n_tri = self._resultado_csg(self.csg_operacion, self.csg_b)
        info = f"{texto}   ·   {n_tri} triángulos   ·   calculado en {ms:.0f} ms   ·   arrastra para girar"
        self.pantalla.blit(self.fuente_simbolos.render(info, True, (40, 50, 65)), (rect.x + 10, rect.bottom - 26))
        leyenda = [("Caras de A (caja)", COLOR_CSG_A), ("Caras de B", COLOR_CSG_B)]
        for k, (txt, col) in enumerate(leyenda):
            pygame.draw.rect(self.pantalla, col, (rect.x + 12, rect.y + 50 + k * 22, 14, 14))
            self.pantalla.blit(self.fuente.render(txt, True, (40, 50, 65)), (rect.x + 32, rect.y + 48 + k * 22))

    def _clic_boton_csg(self, mx: int, my: int) -> bool:
        for r, tipo, valor in self._botones_csg:
            if r.collidepoint(mx, my):
                if tipo == "operacion":
                    self.csg_operacion = valor
                    self.csg_azimut, self.csg_elevacion = VISTA_CSG[valor]
                else:
                    self.csg_b = valor
                _, texto, ms, n_tri = self._resultado_csg(self.csg_operacion, self.csg_b)
                self.mensaje = f"{texto} ({n_tri} triángulos, {ms:.0f} ms)"
                return True
        return False

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
        for i, v in enumerate(VISTAS):
            rx = 180 + i * PASO_PESTANA
            activo = self.vista_activa == v
            col = (255, 255, 255) if activo else (210, 222, 236)
            pygame.draw.rect(self.pantalla, col, (rx, 50, ANCHO_PESTANA, 32), border_top_left_radius=6, border_top_right_radius=6)
            self.pantalla.blit(self.fuente_bold.render(v, True, (25, 45, 75)), (rx + 14, 58))

        # Panel izquierdo: Herramientas
        pygame.draw.rect(self.pantalla, (244, 246, 249), (0, 46, 175, self.alto - 74))
        self.pantalla.blit(self.fuente_bold.render("Herramientas", True, (30, 45, 70)), (16, 56))
        for idx, h_nom in enumerate(HERRAMIENTAS):
            ry = Y_HERRAMIENTAS + idx * ALTO_BOTON
            sel = self.herramienta == h_nom
            dibujo = h_nom in HERRAMIENTAS_DIBUJO
            fondo = (52, 110, 175) if sel else ((232, 228, 242) if dibujo else (225, 232, 240))
            pygame.draw.rect(self.pantalla, fondo, (14, ry, 146, 28), border_radius=4)
            self.pantalla.blit(self.fuente.render(h_nom, True, (255, 255, 255) if sel else (30, 40, 55)), (24, ry + 6))

        # Paleta del editor 2D: el color de trazo lleva marco grueso; el de relleno, un punto
        self.pantalla.blit(self.fuente.render("Color (Shift: relleno)", True, (30, 45, 70)), (14, Y_PALETA - 19))
        for k, color in enumerate(PALETA):
            rx, ry = 14 + (k % 4) * PASO_MUESTRA, Y_PALETA + (k // 4) * 28
            pygame.draw.rect(self.pantalla, color, (rx, ry, LADO_MUESTRA, 22))
            pygame.draw.rect(self.pantalla, (20, 20, 20) if color == self.color_trazo else (170, 175, 185),
                             (rx, ry, LADO_MUESTRA, 22), 3 if color == self.color_trazo else 1)
            if color == self.color_relleno:
                pygame.draw.circle(self.pantalla, (20, 20, 20), (rx + LADO_MUESTRA // 2, ry + 11), 5)
                pygame.draw.circle(self.pantalla, (255, 255, 255), (rx + LADO_MUESTRA // 2, ry + 11), 3)
        x, y, w, h = RECT_SIN_RELLENO
        pygame.draw.rect(self.pantalla, (52, 110, 175) if self.color_relleno is None else (225, 232, 240),
                         (x, y, w, h), border_radius=4)
        self.pantalla.blit(self.fuente.render("Sin relleno", True, (255, 255, 255) if self.color_relleno is None
                                              else (30, 40, 55)), (x + 10, y + 3))

        # Panel derecho: Propiedades y estados
        px = self.ancho - 225
        pygame.draw.rect(self.pantalla, (244, 246, 249), (px, 46, 225, self.alto - 74))
        self.pantalla.blit(self.fuente_bold.render("Propiedades y Atajos", True, (30, 45, 70)), (px + 14, 56))

        cat_nom = CATALOGO_MUEBLES[self.idx_catalogo][0]
        muebles = self.escena.plano.muebles
        if self.mueble_sel is not None and 0 <= self.mueble_sel < len(muebles):
            mu = muebles[self.mueble_sel]
            txt_sel = f"Selección: {mu.nombre} ({mu.x:.2f}, {mu.y:.2f}) {mu.angulo % 360:.0f}°"
            txt_esc = f"Escala [Z/X]: {mu.escala_x:.2f} × {mu.escala_y:.2f}"
        else:
            txt_sel = "Selección: ninguna"
            txt_esc = "Escala [Z/X]: —"
        if self.herramienta in ANCHO_ABERTURA:
            txt_esc = f"Ancho {self.herramienta.lower()} [Z/X]: {self.ancho_abertura[self.herramienta]:.2f} m"
        props = [
            txt_sel,
            txt_esc,
            f"Catálogo [Tab]: {cat_nom}",
            f"Proyección [P]: {self.escena.proyeccion}",
            f"Sombreado [G]: {self.escena.sombreado}",
            f"Iluminación [L]: {'ON' if self.escena.iluminacion_activa else 'OFF'}",
            f"Texturas [T]: {'ON' if self.escena.texturas_activas else 'OFF'} ({len(self.escena.texturas)} imágenes)",
            "Recargar texturas: R",
            f"Alambre [K]: {'ON' if self.escena.modo_alambre else 'OFF'}",
            f"Antialiasing [U]: {'Wu' if self.antialiasing_2d else 'OFF'}",
            f"Algoritmo líneas [B]: {NOMBRES_ALGORITMO[self.algoritmo_lineas]}",
            f"Grosor [1-9]: {self.grosor_trazo} px",
            f"Fractal nivel [+/-]: {self.fractal_profundidad}",
            "Mover: Seleccionar + arrastrar",
            "Escalar: esquina (Shift: proporción)",
            "Rotar mueble: Q / E",
            "Borrar mueble sel.: Supr",
            "Cámara 3D: W A S D + Mouse",
            "Panel de luz (Vista 3D): O",
        ]
        for i, txt in enumerate(props):
            self.pantalla.blit(self.fuente.render(txt, True, (40, 50, 65)), (px + 14, 90 + i * 26))

        # Lienzo central
        rect_lienzo = pygame.Rect(180, 84, self.ancho - 410, self.alto - 116)
        t0 = time.perf_counter()
        if self.vista_activa == "Plano 2D":
            fb = self._renderizar_lienzo_2d(rect_lienzo.width, rect_lienzo.height)
        renderizado = True
        if self.vista_activa == "Plano 2D":
            fb = self._renderizar_lienzo_2d(rect_lienzo.width, rect_lienzo.height)
        elif self.vista_activa == "Vista 3D":
            alto_3d = self.alto_3d_movimiento if self.camara_moviendose else self.alto_3d
            ancho_3d = int(round(alto_3d * rect_lienzo.width / rect_lienzo.height))
            clave = self._clave_3d(ancho_3d, alto_3d)
            if self._cache_3d is not None and self._cache_3d[0] == clave:
                fb, renderizado = self._cache_3d[1], False      # nada cambió: se reutiliza
            else:
                fb = self.escena.renderizar_3d(ancho_3d, alto_3d)
                self._cache_3d = (clave, fb)
            if self.escena.iluminacion_activa:
                # Ícono de la luz donde la proyecta la cámara (sobre una copia: el render queda en caché)
                l = self.escena.luz.posicion
                pos = self.escena.proyectar_punto((l.x, l.y, l.z), fb.ancho, fb.alto)
                if pos is not None:
                    fb_luz = Framebuffer(fb.ancho, fb.alto)
                    fb_luz.datos[:] = fb.datos
                    self._dibujar_sol(fb_luz, pos[0], pos[1], r=4)
                    fb = fb_luz
        elif self.vista_activa == "Editor de curvas":
            fb = self._renderizar_editor_curvas(rect_lienzo.width, rect_lienzo.height)
        elif self.vista_activa == "Fractales":
            fb = self._renderizar_vista_fractales(rect_lienzo.width, rect_lienzo.height)
        else:
            cache_antes = self._cache_csg
            fb = self._renderizar_vista_csg(rect_lienzo.width, rect_lienzo.height)
            renderizado = self._cache_csg is not cache_antes     # si reutilizó el render, no hay FPS que medir
        if renderizado:
            self.fps_render = 1.0 / max(time.perf_counter() - t0, 1e-6)
        self._volcar_framebuffer(fb, rect_lienzo)
        if self.vista_activa == "CSG":
            self._dibujar_controles_csg(rect_lienzo)
        if self.vista_activa == "Vista 3D":
            self._dibujar_panel_luz(rect_lienzo)
        if self.vista_activa == "Editor de curvas":
            # Etiquetas de los puntos de control y ayuda (texto de la interfaz)
            for i, (px, py) in enumerate(self._perfil_a_lienzo(self.puntos_bezier)):
                self.pantalla.blit(self.fuente.render(f"P{i}", True, (30, 60, 130)),
                                   (rect_lienzo.x + px + 9, rect_lienzo.y + py - 18))
            grado = len(self.puntos_bezier) - 1
            ayuda = (f"Bézier de grado {grado}  ·  De Casteljau con t = {self.t_bezier:.2f} (rueda)  ·  "
                     "arrastrar: mover punto  ·  clic derecho: agregar / quitar")
            self.pantalla.blit(self.fuente.render(ayuda, True, (60, 70, 90)), (rect_lienzo.x + 10, rect_lienzo.y + 8))

        # Barra inferior de estado
        pygame.draw.rect(self.pantalla, (224, 230, 238), (0, self.alto - 28, self.ancho, 28))
        estado = (
            f"x: {self.mouse_mundo[0]:.2f} m  y: {self.mouse_mundo[1]:.2f} m  |  "
            f"Triángulos 3D: {self.escena.total_triangulos()}  |  "
            f"Render software: {self.fps_render:.1f} FPS"
        )
        if self.mensaje:
            estado += f"  |  {self.mensaje}"
        fuente = self.fuente_simbolos if any(s in estado for s in "∪∩") else self.fuente
        self.pantalla.blit(fuente.render(estado, True, (30, 40, 55)), (16, self.alto - 22))
        pygame.display.flip()

    def ejecutar(self) -> None:
        arrastrando_3d = False
        arrastrando_csg = False
        ultimo_mouse = (0, 0)
        corriendo = True
        dt_cuadro = 1 / 30

        while corriendo:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    corriendo = False

                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        # Esc primero cancela lo que se está dibujando; si no hay nada, cierra
                        if self._cancelar_dibujo():
                            self.mensaje = "Dibujo cancelado"
                        else:
                            corriendo = False
                    elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self._cerrar_poligono()
                    elif pygame.K_1 <= ev.key <= pygame.K_9:
                        self.grosor_trazo = ev.key - pygame.K_0
                        self.mensaje = f"Grosor del pincel: {self.grosor_trazo} px"
                    elif ev.key == pygame.K_o:
                        self.panel_luz_visible = not self.panel_luz_visible
                        self.mensaje = "Panel de luz " + ("visible" if self.panel_luz_visible else "oculto (O)")
                    elif ev.key == pygame.K_r:
                        # Vuelve a leer assets/texturas (p. ej. tras reemplazar piso.png por una foto)
                        n = self.escena.recargar_texturas()
                        avisos = self.escena.avisos_texturas
                        self.mensaje = (f"Texturas recargadas: {n} imágenes de assets/texturas"
                                        + (f" ({avisos[0]})" if avisos else ""))
                    elif ev.key == pygame.K_b:
                        self.algoritmo_lineas = "dda" if self.algoritmo_lineas == "bresenham" else "bresenham"
                        self.mensaje = f"Algoritmo de líneas: {NOMBRES_ALGORITMO[self.algoritmo_lineas]}"
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
                    elif ev.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                        if self.mueble_sel is not None and 0 <= self.mueble_sel < len(self.escena.plano.muebles):
                            self._borrar(("mueble", self.mueble_sel))
                        else:
                            self.mensaje = "No hay mueble seleccionado (usa la herramienta Borrar)"
                    elif ev.key in (pygame.K_z, pygame.K_x) and self.herramienta in ANCHO_ABERTURA:
                        # Con Puerta/Ventana, Z / X cambian el ancho del próximo vano (±10 cm)
                        h = self.herramienta
                        minimo, maximo = LIMITES_ANCHO[h]
                        delta = 0.1 if ev.key == pygame.K_x else -0.1
                        self.ancho_abertura[h] = round(min(maximo, max(minimo, self.ancho_abertura[h] + delta)), 2)
                        self.mensaje = f"Ancho de {h.lower()}: {self.ancho_abertura[h]:.2f} m"
                    elif ev.key in (pygame.K_z, pygame.K_x) and self._mueble_seleccionado() is not None:
                        # Escalamiento uniforme respecto al centro del mueble: ×1/1.1 (Z) o ×1.1 (X)
                        mu = self._mueble_seleccionado()
                        k = 1.1 if ev.key == pygame.K_x else 1 / 1.1
                        mu.escala_x, mu.escala_y = escala_proporcional(mu.escala_x, mu.escala_y, k)
                        self.escena.reconstruir_3d()
                        self.mensaje = f"Escala de {mu.nombre}: {mu.escala_x:.2f} × {mu.escala_y:.2f}"
                    elif ev.key in (pygame.K_q, pygame.K_e) and self.mueble_sel is not None:
                        delta = 15.0 if ev.key == pygame.K_q else -15.0
                        self.escena.plano.muebles[self.mueble_sel].angulo += delta
                        self.escena.reconstruir_3d()

                elif ev.type == pygame.MOUSEWHEEL and self.vista_activa == "Plano 2D":
                    self.escala_2d = max(25.0, min(160.0, self.escala_2d + ev.y * 5.0))

                elif ev.type == pygame.MOUSEWHEEL and self.vista_activa == "Editor de curvas":
                    # La rueda recorre la curva: cambia t de la construcción de De Casteljau
                    self.t_bezier = round(min(1.0, max(0.0, self.t_bezier + ev.y * 0.05)), 2)

                elif (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3 and self.vista_activa == "Editor de curvas"
                      and self._rect_lienzo().collidepoint(ev.pos)):
                    self._agregar_o_quitar_punto_bezier(*ev.pos)

                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3) and self._en_paleta(*ev.pos):
                    shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    self._clic_paleta(*ev.pos, relleno=ev.button == 3 or shift)

                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    mx, my = ev.pos
                    # Cambio de pestañas
                    if 50 <= my <= 82:
                        for i, v in enumerate(VISTAS):
                            if 180 + i * PASO_PESTANA <= mx <= 180 + i * PASO_PESTANA + ANCHO_PESTANA:
                                self.vista_activa = v
                    # Selección de herramienta izquierda
                    elif mx < 175 and Y_HERRAMIENTAS <= my < Y_HERRAMIENTAS + len(HERRAMIENTAS) * ALTO_BOTON:
                        idx = (my - Y_HERRAMIENTAS) // ALTO_BOTON
                        if 0 <= idx < len(HERRAMIENTAS):
                            self.herramienta = HERRAMIENTAS[idx]
                            self._cancelar_dibujo()
                    # Interacción en lienzo
                    elif 180 <= mx <= self.ancho - 230 and 84 <= my <= self.alto - 32:
                        if self.vista_activa == "Vista 3D" and self._clic_panel_luz(mx, my):
                            pass                                 # deslizador o botón del panel de luz
                        elif self.vista_activa == "Vista 3D":
                            arrastrando_3d = True
                            ultimo_mouse = (mx, my)
                        elif self.vista_activa == "Editor de curvas":
                            self.ctrl_bezier_sel = self._punto_bezier_en(mx, my)
                        elif self.vista_activa == "CSG":
                            if not self._clic_boton_csg(mx, my):
                                arrastrando_csg = True          # girar el objeto
                                ultimo_mouse = (mx, my)
                        elif self.vista_activa == "Plano 2D":
                            wx, wy = self._pantalla_a_mundo_2d(mx, my)
                            if self.herramienta == "Seleccionar" and self._manija_en(mx, my):
                                self._iniciar_escalado()
                            elif self.herramienta == "Seleccionar":
                                i = self.escena.plano.mueble_en(wx, wy)
                                self.mueble_sel = i
                                if i is None:
                                    self.mensaje = "Nada seleccionado"
                                else:
                                    mu = self.escena.plano.muebles[i]
                                    self.desfase_arrastre = (mu.x - wx, mu.y - wy)
                                    self.posicion_antes_arrastre = (mu.x, mu.y)
                                    self.mensaje = f"Seleccionado: {mu.nombre}"
                            elif self.herramienta in ANCHO_ABERTURA:
                                self._colocar_abertura(wx, wy)
                            elif self.herramienta in HERRAMIENTAS_DIBUJO:
                                self._clic_dibujo(wx, wy, mx, my)
                            elif self.herramienta == "Borrar":
                                self._borrar(self.escena.plano.elemento_en(wx, wy))
                            elif self.herramienta == "Muro":
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
                    arrastrando_csg = False
                    if self.deslizador_luz is not None:
                        etiqueta = dict(CONTROLES_LUZ)[self.deslizador_luz]
                        self.mensaje = f"Luz – {etiqueta}: {self._valor_luz(self.deslizador_luz):.2f}"
                        self.deslizador_luz = None
                    if self.ctrl_bezier_sel is not None:
                        r, y = self.puntos_bezier[self.ctrl_bezier_sel]
                        self.mensaje = f"P{self.ctrl_bezier_sel} = (r {r:.3f}, y {y:.3f}) m"
                        self.ctrl_bezier_sel = None
                    if self.escalado is not None:
                        mu = self._mueble_seleccionado()
                        if (mu.escala_x, mu.escala_y) != (self.escalado["sx0"], self.escalado["sy0"]):
                            self.escena.reconstruir_3d()
                            self.mensaje = f"Escala de {mu.nombre}: {mu.escala_x:.2f} × {mu.escala_y:.2f}"
                        self.escalado = None
                    if self.desfase_arrastre is not None:
                        # Fin del arrastre: el modelo 3D se regenera una sola vez, al soltar
                        mu = self.escena.plano.muebles[self.mueble_sel]
                        if (mu.x, mu.y) != self.posicion_antes_arrastre:
                            self.escena.reconstruir_3d()
                            self.mensaje = f"Movido: {mu.nombre} a ({mu.x:.2f}, {mu.y:.2f}) m"
                        self.desfase_arrastre = None
                        self.posicion_antes_arrastre = None

                elif ev.type == pygame.MOUSEMOTION:
                    mx, my = ev.pos
                    self.mouse_mundo = self._pantalla_a_mundo_2d(mx, my)
                    if self.deslizador_luz is not None and self.vista_activa == "Vista 3D":
                        self._mover_deslizador_luz(mx)
                    if self.ctrl_bezier_sel is not None and self.vista_activa == "Editor de curvas":
                        self._mover_punto_bezier(mx, my)
                    if self.escalado is not None and self.vista_activa == "Plano 2D":
                        self._actualizar_escalado(proporcional=bool(pygame.key.get_mods() & pygame.KMOD_SHIFT))
                    if self.desfase_arrastre is not None and self.vista_activa == "Plano 2D":
                        # Traslación del mueble: su centro sigue al cursor (el polígono se
                        # recalcula con su matriz de modelo T · R · S al dibujar)
                        mu = self.escena.plano.muebles[self.mueble_sel]
                        mu.x = round(self.mouse_mundo[0] + self.desfase_arrastre[0], 2)
                        mu.y = round(self.mouse_mundo[1] + self.desfase_arrastre[1], 2)
                    if arrastrando_csg and self.vista_activa == "CSG":
                        # Órbita alrededor del objeto: la imagen sigue al mouse
                        self.csg_azimut = (self.csg_azimut + (mx - ultimo_mouse[0]) * 0.5) % 360
                        self.csg_elevacion = max(-20.0, min(80.0, self.csg_elevacion + (my - ultimo_mouse[1]) * 0.4))
                        ultimo_mouse = (mx, my)
                    if arrastrando_3d and self.vista_activa == "Vista 3D":
                        dx = mx - ultimo_mouse[0]
                        dy = my - ultimo_mouse[1]
                        ultimo_mouse = (mx, my)
                        # Arrastre tipo "agarrar la escena": la imagen sigue al mouse.
                        # Arrastrar a la derecha gira la cámara a la izquierda (yaw −),
                        # arrastrar hacia arriba la inclina hacia abajo (pitch −).
                        self.escena.camara.rotar_vista(-dx * 0.35, dy * 0.35)

            # Movimiento continuo W A S D en Vista 3D. El paso depende del tiempo real del
            # cuadro anterior (velocidad × dt), así la cámara avanza a la misma velocidad
            # en m/s aunque varíen los FPS; dt se limita para evitar saltos grandes.
            teclas = pygame.key.get_pressed()
            wasd = any(teclas[k] for k in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d))
            self.camara_moviendose = self.vista_activa == "Vista 3D" and (wasd or arrastrando_3d)
            if self.vista_activa == "Vista 3D" and wasd:
                paso = self.velocidad_camara * min(dt_cuadro, 0.1)
                if teclas[pygame.K_w]:
                    self.escena.camara.mover(adelante=paso)
                if teclas[pygame.K_s]:
                    self.escena.camara.mover(adelante=-paso)
                if teclas[pygame.K_a]:
                    self.escena.camara.mover(derecha=-paso)
                if teclas[pygame.K_d]:
                    self.escena.camara.mover(derecha=paso)

            self._dibujar_ui()
            dt_cuadro = self.reloj.tick(30) / 1000.0

        pygame.quit()
        sys.exit(0)
