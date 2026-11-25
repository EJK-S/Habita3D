"""Rasterización de triángulos 3D con z-buffer, corrección de perspectiva UV y wireframe."""
import numpy as np
from raster2d import Framebuffer, linea
from .zbuffer import ZBuffer
from .textura import Textura


def rasterizar_triangulo(
    fb: Framebuffer,
    zb: ZBuffer,
    pts_pant: np.ndarray,
    inv_w: np.ndarray,
    uvs: np.ndarray,
    colores_vert: np.ndarray,
    textura: Textura | None = None,
    usar_textura: bool = True,
) -> None:
    """
    Rellena un triángulo proyectado en pantalla usando coordenadas baricéntricas vectorizadas
    sobre su caja delimitadora (bounding box), con prueba de z-buffer e interpolación
    de coordenadas de textura (u, v) corregida por perspectiva.
    """
    x0, y0, z0 = pts_pant[0]
    x1, y1, z1 = pts_pant[1]
    x2, y2, z2 = pts_pant[2]

    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if abs(area) < 1e-6:
        return

    min_x = max(0, int(np.floor(min(x0, x1, x2))))
    max_x = min(fb.ancho - 1, int(np.ceil(max(x0, x1, x2))))
    min_y = max(0, int(np.floor(min(y0, y1, y2))))
    max_y = min(fb.alto - 1, int(np.ceil(max(y0, y1, y2))))

    if min_x > max_x or min_y > max_y:
        return

    # Grilla de centros de píxel dentro del bounding box
    xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
    gx, gy = np.meshgrid(xs, ys)

    inv_area = 1.0 / area
    w0 = ((x1 - gx) * (y2 - gy) - (y1 - gy) * (x2 - gx)) * inv_area
    w1 = ((x2 - gx) * (y0 - gy) - (y2 - gy) * (x0 - gx)) * inv_area
    w2 = 1.0 - w0 - w1

    dentro = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    if not np.any(dentro):
        return

    py, px = np.where(dentro)
    fw0 = w0[py, px]
    fw1 = w1[py, px]
    fw2 = w2[py, px]

    z_interp = fw0 * z0 + fw1 * z1 + fw2 * z2
    img_y = min_y + py
    img_x = min_x + px

    z_actual = zb.profundidad[img_y, img_x]
    pasa_z = z_interp < z_actual
    if not np.any(pasa_z):
        return

    img_y = img_y[pasa_z]
    img_x = img_x[pasa_z]
    fw0 = fw0[pasa_z]
    fw1 = fw1[pasa_z]
    fw2 = fw2[pasa_z]
    zb.profundidad[img_y, img_x] = z_interp[pasa_z]

    # Color interpolado (iluminación Gouraud o plana)
    col_interp = (
        fw0[:, None] * colores_vert[0]
        + fw1[:, None] * colores_vert[1]
        + fw2[:, None] * colores_vert[2]
    )

    if usar_textura and textura is not None:
        # Interpolación con corrección de perspectiva: (u/w) / (1/w)
        denom = fw0 * inv_w[0] + fw1 * inv_w[1] + fw2 * inv_w[2]
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        u_corr = (fw0 * uvs[0, 0] * inv_w[0] + fw1 * uvs[1, 0] * inv_w[1] + fw2 * uvs[2, 0] * inv_w[2]) / denom
        v_corr = (fw0 * uvs[0, 1] * inv_w[0] + fw1 * uvs[1, 1] * inv_w[1] + fw2 * uvs[2, 1] * inv_w[2]) / denom
        tex_rgb = textura.muestrear_vectorizado(u_corr, v_corr)
        # Modular textura por la intensidad de iluminación relativa
        color_final = np.clip(tex_rgb * col_interp * 1.25, 0.0, 1.0)
    else:
        color_final = np.clip(col_interp, 0.0, 1.0)

    fb.datos[img_y, img_x] = (color_final * 255.0).astype(np.uint8)


def dibujar_triangulo_alambre(
    fb: Framebuffer,
    pts_pant: np.ndarray,
    color: tuple[int, int, int] = (40, 60, 95),
) -> None:
    """Dibuja las aristas de un triángulo proyectado usando la primitiva propia `linea`."""
    p0, p1, p2 = pts_pant[0], pts_pant[1], pts_pant[2]
    linea(fb, p0[0], p0[1], p1[0], p1[1], color, 1)
    linea(fb, p1[0], p1[1], p2[0], p2[1], color, 1)
    linea(fb, p2[0], p2[1], p0[0], p0[1], color, 1)
