"""Clase Escena: sincroniza el Plano 2D con las mallas 3D y ejecuta el pipeline de render."""
import pathlib
import numpy as np
from mathlib import Vec3
from plano import Plano
from raster2d import Framebuffer
from modelado import (
    Malla,
    extruir_muro,
    csg_diferencia,
    crear_piso,
    construir_mueble_3d,
    es_decorativo,
)
from vista import Camara, VolumenVision, proy_ortografica, proy_perspectiva
from render import (
    ZBuffer,
    Luz,
    cargar_texturas,
    sombreado_plano,
    sombreado_gouraud,
    rasterizar_triangulo,
    rasterizar_lote,
    dibujar_triangulo_alambre,
)


class Escena:
    """
    Punto de unión entre el editor 2D (Plano) y el motor de visualización 3D.
    Garantiza que las vistas 2D y 3D siempre representen el mismo modelo.
    """

    def __init__(
        self,
        plano: Plano | None = None,
        altura_muros: float = 2.60,
        carpeta_texturas: str | pathlib.Path | None = None,
    ):
        self.plano = plano if plano is not None else Plano()
        self.altura_muros = float(altura_muros)
        self.mallas: list[Malla] = []
        self.mallas_extra: list[Malla] = []

        # Cámara dentro de la sala (esquina suroeste, a altura de los ojos) mirando hacia
        # la esquina opuesta. El plano va de z = 0 a z = -4.5 (Z = -y del plano).
        self.camara = Camara(eye=self.CAMARA_INICIAL[0], target=self.CAMARA_INICIAL[1])
        self.volumen = VolumenVision(cercano=0.15, lejano=60.0)
        self.luz = Luz(posicion=(3.0, 2.35, -2.2))
        # Texturas: mapas de bits cargados de `carpeta_texturas` (o generados si no hay carpeta)
        self.carpeta_texturas = carpeta_texturas
        self.texturas, self.avisos_texturas = cargar_texturas(carpeta_texturas)

        # Opciones de visualización para comparativas en tiempo real
        self.proyeccion = "perspectiva"     # "perspectiva" | "ortografica"
        self.sombreado = "gouraud"          # "plano" | "gouraud"
        self.iluminacion_activa = True
        self.texturas_activas = True
        self.modo_alambre = False
        self.color_cielo = (218, 228, 240)
        # True: proyección y rasterización por lotes (vectorizadas, rápidas).
        # False: triángulo a triángulo (mismo algoritmo, sirve de referencia y comparación).
        self.por_lotes = True

        self.reconstruir_3d()

    CAMARA_INICIAL = ((0.6, 1.6, -0.5), (3.3, 0.9, -4.1))
    RADIO_CAMARA = 0.25      # m: holgura entre la cámara y los muros

    def reiniciar_camara(self) -> None:
        self.camara.mirar_a(*self.CAMARA_INICIAL)

    def mover_camara(self, adelante: float = 0.0, derecha: float = 0.0) -> None:
        """Mueve la cámara sin atravesar muros. El desplazamiento se parte en pasos de a lo
        sumo 10 cm para que un paso grande no "salte" un muro."""
        n = max(1, int(np.ceil(max(abs(adelante), abs(derecha)) / 0.10)))
        for _ in range(n):
            self._paso_camara(adelante / n, derecha / n)

    def _paso_camara(self, adelante: float, derecha: float) -> None:
        """Un paso: si choca, se desliza a lo largo del muro (probando cada eje por separado).
        Si la cámara ya empezó dentro de un muro, se la deja salir."""
        cam, plano = self.camara, self.plano
        e0, t0 = cam.eye, cam.target
        libre = lambda e: plano.punto_libre(e.x, -e.z, self.RADIO_CAMARA)
        cam.mover(adelante=adelante, derecha=derecha)
        d = cam.eye - e0
        if libre(cam.eye) or not libre(e0):
            return
        for paso in (Vec3(d.x, 0.0, 0.0), Vec3(0.0, 0.0, d.z)):
            if libre(e0 + paso):
                cam.eye, cam.target = e0 + paso, t0 + paso
                return
        cam.eye, cam.target = e0, t0

    def recargar_texturas(self) -> int:
        """Vuelve a leer las imágenes de la carpeta (p. ej. tras reemplazar piso.png por una foto)."""
        self.texturas, self.avisos_texturas = cargar_texturas(self.carpeta_texturas)
        return len(self.texturas)

    def agregar_malla_extra(self, malla: Malla) -> None:
        self.mallas_extra.append(malla)

    def reconstruir_3d(self) -> None:
        """Regenera las mallas 3D (piso, muros B-Rep/CSG y muebles) desde el Plano 2D."""
        generadas: list[Malla] = []

        # 1. Calcular envolvente del plano para generar el piso
        if self.plano.muros:
            xs = [p[0] for m in self.plano.muros for p in (m.p1, m.p2)]
            ys = [p[1] for m in self.plano.muros for p in (m.p1, m.p2)]
            generadas.append(crear_piso(min(xs) - 0.1, min(ys) - 0.1, max(xs) + 0.1, max(ys) + 0.1))

        # 2. Agrupar aberturas por índice de muro y aplicar extrusión B-Rep o CSG diferencia
        aberturas_por_muro: dict[int, list[dict]] = {}
        for ab in self.plano.aberturas:
            aberturas_por_muro.setdefault(ab.muro, []).append({
                "pos": ab.pos,
                "ancho": ab.ancho,
                "tipo": ab.tipo,
            })

        for idx, muro in enumerate(self.plano.muros):
            vanos = aberturas_por_muro.get(idx, [])
            if vanos:
                malla_muro = csg_diferencia(
                    muro.p1, muro.p2, muro.espesor, self.altura_muros, vanos
                )
            else:
                malla_muro = extruir_muro(
                    muro.p1, muro.p2, muro.espesor, self.altura_muros
                )
            generadas.append(malla_muro)

        # 3. Instanciar muebles del catálogo sincronizados con MuebleEnPlanta.
        #    Primero los muebles grandes (van en el piso) y luego los decorativos, que se
        #    apoyan sobre la malla 3D de lo que tengan debajo. Se conserva el orden del plano.
        def construir(mu, elevacion=0.0):
            return construir_mueble_3d(
                nombre=mu.nombre, ancho=mu.ancho, largo=mu.largo, x=mu.x, y_plano=mu.y,
                angulo=mu.angulo, color=mu.color, elevacion=elevacion,
                escala_x=mu.escala_x, escala_z=mu.escala_y,
            )

        muebles_3d: list[Malla | None] = [None] * len(self.plano.muebles)
        apoyos: list[Malla] = []
        for i, mu in enumerate(self.plano.muebles):
            if not es_decorativo(mu.nombre):
                muebles_3d[i] = construir(mu)
                apoyos.append(muebles_3d[i])
        for i, mu in enumerate(self.plano.muebles):
            if es_decorativo(mu.nombre):
                muebles_3d[i] = construir(mu, self.altura_apoyo(mu, apoyos))
        generadas.extend(muebles_3d)

        self.mallas = generadas

    @staticmethod
    def altura_apoyo(mueble, apoyos: list[Malla]) -> float:
        """
        Altura a la que se apoya un objeto decorativo (jarrón, lámpara, planta).

        Se lanza un rayo vertical en su centro contra las mallas 3D de los demás muebles
        y se toma la cara superior más alta que toca: el tablero de la mesa, el asiento
        del sofá, el colchón de la cama… Si el centro no cae sobre ningún mueble pero el
        objeto lo pisa en parte (está en el borde), se prueban cuatro puntos interiores
        de su planta para que no quede atravesando el mueble. Si nada lo sostiene, va al piso.
        """
        def altura_bajo(x, y_plano):
            alturas = [h for m in apoyos if (h := m.altura_en(x, -y_plano)) is not None]
            return max(alturas) if alturas else None

        h = altura_bajo(mueble.x, mueble.y)
        if h is None:
            a, l = 0.4 * mueble.ancho / 2, 0.4 * mueble.largo / 2
            puntos = mueble.matriz_modelo().aplicar_puntos([(a, 0), (-a, 0), (0, l), (0, -l)])
            alturas = [hb for px, py in puntos if (hb := altura_bajo(px, py)) is not None]
            h = max(alturas) if alturas else None
        return max(h, 0.0) if h is not None else 0.0

    def total_triangulos(self) -> int:
        return sum(len(m.caras) for m in (self.mallas + self.mallas_extra))

    def matrices(self, ancho: int, alto: int):
        """Matriz de vista de la cámara y matriz de proyección (ortográfica o en perspectiva)."""
        mat_vista = self.camara.matriz_vista()
        aspecto = ancho / max(alto, 1)
        if self.proyeccion == "ortografica":
            escala_o = 4.6
            mat_proy = proy_ortografica(
                -escala_o * aspecto, escala_o * aspecto, -escala_o, escala_o,
                self.volumen.cercano, self.volumen.lejano,
            )
        else:
            mat_proy = proy_perspectiva(
                fov_grados=60.0, aspecto=aspecto,
                cercano=self.volumen.cercano, lejano=self.volumen.lejano,
            )
        return mat_vista, mat_proy

    def proyectar_punto(self, punto, ancho: int, alto: int) -> tuple[float, float] | None:
        """Píxel (x, y) donde se ve un punto 3D del mundo, o None si queda detrás de la cámara.
        Mismo camino que los vértices: vista → proyección → división por w → pantalla."""
        mat_vista, mat_proy = self.matrices(ancho, alto)
        p_cam = mat_vista.m @ np.array([*punto, 1.0])
        if p_cam[2] > -self.volumen.cercano:
            return None
        clip = mat_proy.m @ p_cam
        ndc = clip[:3] / clip[3]
        return (ndc[0] + 1.0) * 0.5 * (ancho - 1), (1.0 - ndc[1]) * 0.5 * (alto - 1)

    def renderizar_3d(self, ancho: int = 480, alto: int = 320) -> Framebuffer:
        """Ejecuta el pipeline completo de visualización 3D sobre un Framebuffer NumPy."""
        fb = Framebuffer(ancho, alto, fondo=self.color_cielo)
        zb = ZBuffer(ancho, alto)

        mat_vista, mat_proy = self.matrices(ancho, alto)
        es_orto = self.proyeccion == "ortografica"

        pos_cam = np.array([self.camara.eye.x, self.camara.eye.y, self.camara.eye.z], dtype=np.float64)
        lote = []   # triángulos proyectados de todas las mallas (modo por lotes)

        for malla in self.mallas + self.mallas_extra:
            if len(malla.caras) == 0:
                continue

            verts_cam = mat_vista.aplicar_puntos(malla.vertices)

            # Descarte de caras traseras (backface culling)
            visibles = self.volumen.filtrar_caras_traseras(verts_cam, malla.caras, es_ortografica=es_orto)
            indices_vis = np.where(visibles)[0]
            if len(indices_vis) == 0:
                continue

            # Calcular iluminación en espacio de mundo según modo activo.
            # Cada pieza del objeto conserva su color (colores_vertice) y su textura (texturas_cara).
            base_vert = malla.colores_vertice / 255.0
            if not self.iluminacion_activa:
                colores_vert = base_vert
                colores_cara = None
            elif self.sombreado == "plano":
                colores_cara = sombreado_plano(
                    malla.vertices, malla.caras, malla.normales_cara,
                    malla.colores_vertice[malla.caras[:, 0]], pos_cam, self.luz,
                )
                colores_vert = None
            else:  # gouraud
                colores_vert = sombreado_gouraud(
                    malla.vertices, malla.normales_vertice,
                    malla.colores_vertice, pos_cam, self.luz,
                )
                colores_cara = None

            # Datos de todos los triángulos visibles de la malla a la vez: (T, 3, ...)
            caras_vis = malla.caras[indices_vis]
            p_cam = verts_cam[caras_vis]
            uv_tri = malla.uvs[caras_vis]
            if colores_cara is not None:
                c_tri = np.repeat(colores_cara[indices_vis][:, None, :], 3, axis=1)
            else:
                c_tri = colores_vert[caras_vis]

            texs = [self.texturas.get(malla.texturas_cara[i]) if (self.texturas_activas and malla.texturas_cara[i])
                    else None for i in indices_vis]
            con_tex = np.array([t is not None for t in texs])
            if np.any(con_tex):
                # Con textura se usa solo el factor de luz (ambiental + difusa + especular):
                # se divide entre el color base para no teñir dos veces la imagen.
                base_tri = base_vert[caras_vis[con_tex]]
                safe_base = np.where(base_tri < 1e-4, 1.0, base_tri)
                c_tri = c_tri.copy()
                c_tri[con_tex] = np.clip(c_tri[con_tex] / safe_base, 0.0, 1.3)

            if self.por_lotes:
                pts, inv_w, uv_r, col_r, origen = self.volumen.proyectar_lote(
                    p_cam, uv_tri, c_tri, mat_proy, ancho, alto
                )
                if self.modo_alambre:
                    for pts_pant in pts:
                        dibujar_triangulo_alambre(fb, pts_pant)
                else:
                    # Se acumulan los triángulos de todas las mallas (en orden) y se
                    # rasterizan juntos al final: el color se calcula una vez por píxel
                    lote.append((pts, inv_w, uv_r, col_r, [texs[o] for o in origen]))
                continue

            # Camino triángulo a triángulo (referencia del algoritmo, más lento en Python)
            for k in range(len(indices_vis)):
                recortados = self.volumen.recortar_y_proyectar_triangulo(
                    p_cam[k], uv_tri[k], c_tri[k], mat_proy, ancho, alto
                )
                for pts_pant, inv_w, uv_r, col_r in recortados:
                    if self.modo_alambre:
                        dibujar_triangulo_alambre(fb, pts_pant)
                    else:
                        rasterizar_triangulo(
                            fb, zb, pts_pant, inv_w, uv_r, col_r,
                            textura=texs[k], usar_textura=texs[k] is not None,
                        )

        if lote:
            rasterizar_lote(
                fb, zb,
                np.concatenate([l[0] for l in lote]), np.concatenate([l[1] for l in lote]),
                np.concatenate([l[2] for l in lote]), np.concatenate([l[3] for l in lote]),
                [t for l in lote for t in l[4]],
            )
        return fb
