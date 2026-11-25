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
)
from vista import Camara, VolumenVision, proy_ortografica, proy_perspectiva
from render import (
    ZBuffer,
    Luz,
    generar_texturas_defecto,
    sombreado_plano,
    sombreado_gouraud,
    rasterizar_triangulo,
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

        self.camara = Camara(eye=(3.0, 2.3, 3.2), target=(3.0, 1.0, -2.2))
        self.volumen = VolumenVision(cercano=0.15, lejano=60.0)
        self.luz = Luz(posicion=(3.0, 2.35, -2.2))
        self.texturas = generar_texturas_defecto(carpeta_texturas)

        # Opciones de visualización para comparativas en tiempo real
        self.proyeccion = "perspectiva"     # "perspectiva" | "ortografica"
        self.sombreado = "gouraud"          # "plano" | "gouraud"
        self.iluminacion_activa = True
        self.texturas_activas = True
        self.modo_alambre = False
        self.color_cielo = (218, 228, 240)

        self.reconstruir_3d()

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

        # 3. Instanciar muebles del catálogo sincronizados con MuebleEnPlanta
        for mu in self.plano.muebles:
            generadas.append(
                construir_mueble_3d(
                    nombre=mu.nombre,
                    ancho=mu.ancho,
                    largo=mu.largo,
                    x=mu.x,
                    y_plano=mu.y,
                    angulo=mu.angulo,
                    color=mu.color,
                )
            )

        self.mallas = generadas

    def total_triangulos(self) -> int:
        return sum(len(m.caras) for m in (self.mallas + self.mallas_extra))

    def renderizar_3d(self, ancho: int = 480, alto: int = 320) -> Framebuffer:
        """Ejecuta el pipeline completo de visualización 3D sobre un Framebuffer NumPy."""
        fb = Framebuffer(ancho, alto, fondo=self.color_cielo)
        zb = ZBuffer(ancho, alto)

        mat_vista = self.camara.matriz_vista()
        aspecto = ancho / max(alto, 1)
        es_orto = self.proyeccion == "ortografica"
        if es_orto:
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

        pos_cam = np.array([self.camara.eye.x, self.camara.eye.y, self.camara.eye.z], dtype=np.float64)

        for malla in self.mallas + self.mallas_extra:
            if len(malla.caras) == 0:
                continue

            verts_cam = mat_vista.aplicar_puntos(malla.vertices)

            # Descarte de caras traseras (backface culling)
            visibles = self.volumen.filtrar_caras_traseras(verts_cam, malla.caras, es_ortografica=es_orto)
            indices_vis = np.where(visibles)[0]
            if len(indices_vis) == 0:
                continue

            # Calcular iluminación en espacio de mundo según modo activo
            col_norm_obj = np.asarray(malla.color, dtype=np.float64) / 255.0
            if not self.iluminacion_activa:
                colores_vert = np.tile(col_norm_obj, (len(malla.vertices), 1))
                colores_cara = None
            elif self.sombreado == "plano":
                colores_cara = sombreado_plano(
                    malla.vertices, malla.caras, malla.normales_cara,
                    malla.color, pos_cam, self.luz,
                )
                colores_vert = None
            else:  # gouraud
                colores_vert = sombreado_gouraud(
                    malla.vertices, malla.normales_vertice,
                    malla.color, pos_cam, self.luz,
                )
                colores_cara = None

            tex = self.texturas.get(malla.textura_id) if (self.texturas_activas and malla.textura_id) else None
            # Si hay textura activa y luz activa, usamos intensidad blanca modulada para no teñir doble
            usar_tex = tex is not None

            for idx_cara in indices_vis:
                c0, c1, c2 = malla.caras[idx_cara]
                p_cam = verts_cam[[c0, c1, c2]]
                uv_tri = malla.uvs[[c0, c1, c2]]

                if colores_cara is not None:
                    c_tri = np.repeat(colores_cara[idx_cara][None, :], 3, axis=0)
                else:
                    c_tri = colores_vert[[c0, c1, c2]]

                if usar_tex:
                    # Dividir por color_obj para obtener factor puro de luz (ambiental + difusa + especular)
                    safe_base = np.where(col_norm_obj < 1e-4, 1.0, col_norm_obj)
                    c_tri = np.clip(c_tri / safe_base[None, :], 0.0, 1.3)

                recortados = self.volumen.recortar_y_proyectar_triangulo(
                    p_cam, uv_tri, c_tri, mat_proy, ancho, alto
                )
                for pts_pant, inv_w, uv_r, col_r in recortados:
                    if self.modo_alambre:
                        dibujar_triangulo_alambre(fb, pts_pant)
                    else:
                        rasterizar_triangulo(
                            fb, zb, pts_pant, inv_w, uv_r, col_r,
                            textura=tex, usar_textura=usar_tex,
                        )

        return fb
