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


def rasterizar_lote(
    fb: Framebuffer,
    zb: ZBuffer,
    pts_pant: np.ndarray,
    inv_w: np.ndarray,
    uvs: np.ndarray,
    colores_vert: np.ndarray,
    texturas: list,
    max_pixeles: int = 250_000,
) -> None:
    """
    Mismo algoritmo que `rasterizar_triangulo` (baricéntricas, z-buffer y UV con
    corrección de perspectiva), pero para K triángulos a la vez: pts_pant (K,3,3),
    inv_w (K,3), uvs (K,3,2), colores_vert (K,3,3) y `texturas`, una Textura o None
    por triángulo. Produce exactamente la misma imagen.

    1. Líneas de barrido: en cada fila del triángulo, las tres funciones de arista son
       lineales en x, así que el tramo donde todas son ≥ 0 es un intervalo [x_ini, x_fin].
       Solo se generan píxeles candidatos dentro de ese tramo (más un píxel de margen),
       no en toda la caja delimitadora, que en un triángulo grande es el doble de área.
    2. Por lotes: llamar a NumPy una vez por triángulo cuesta más que el trabajo en sí
       cuando los triángulos son pequeños; aquí se procesan juntos muchos triángulos.
    3. Visibilidad por píxel: entre los candidatos de un mismo píxel gana el más cercano
       (en empate, el triángulo anterior, como al dibujar en orden) y luego se compara
       con el z-buffer. Los triángulos se agrupan en lotes de hasta `max_pixeles` candidatos.
    """
    K = len(pts_pant)
    if K == 0:
        return
    x, y = pts_pant[:, :, 0], pts_pant[:, :, 1]
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (y[:, 1] - y[:, 0]) * (x[:, 2] - x[:, 0])
    min_x = np.maximum(0, np.floor(x.min(axis=1))).astype(np.int64)
    max_x = np.minimum(fb.ancho - 1, np.ceil(x.max(axis=1))).astype(np.int64)
    min_y = np.maximum(0, np.floor(y.min(axis=1))).astype(np.int64)
    max_y = np.minimum(fb.alto - 1, np.ceil(y.max(axis=1))).astype(np.int64)
    valido = (np.abs(area) >= 1e-6) & (min_x <= max_x) & (min_y <= max_y)

    # --- Tramos por fila (línea de barrido) para todos los triángulos a la vez ---
    alturas = np.where(valido, max_y - min_y + 1, 0)
    inicio_filas = np.concatenate(([0], np.cumsum(alturas)))
    f_tri = np.repeat(np.arange(K), alturas)
    f_y = min_y[f_tri] + (np.arange(inicio_filas[-1]) - inicio_filas[f_tri])
    gy = f_y + 0.5
    X, Y, ar = x[f_tri], y[f_tri], area[f_tri]
    # Numerador de cada función de arista en la fila: c + s·gx (ver w0, w1, w2 abajo)
    c0 = X[:, 1] * (Y[:, 2] - gy) - X[:, 2] * (Y[:, 1] - gy)
    c1 = X[:, 2] * (Y[:, 0] - gy) - X[:, 0] * (Y[:, 2] - gy)
    aristas = ((c0, Y[:, 1] - Y[:, 2]), (c1, Y[:, 2] - Y[:, 0]), (ar - c0 - c1, Y[:, 0] - Y[:, 1]))
    signo, abs_ar = np.sign(ar), np.abs(ar)
    lo = np.full(len(f_tri), -np.inf)
    hi = np.full(len(f_tri), np.inf)
    vacio = np.zeros(len(f_tri), dtype=bool)
    for c, s in aristas:
        C, S = signo * c + 1e-5 * abs_ar, signo * s          # condición: C + S·gx ≥ 0
        sube, baja = S > 1e-12, S < -1e-12
        borde = -C / np.where(sube | baja, S, 1.0)
        lo = np.where(sube, np.maximum(lo, borde), lo)
        hi = np.where(baja, np.minimum(hi, borde), hi)
        vacio |= ~sube & ~baja & (C < 0)
    x_ini = np.maximum(min_x[f_tri], np.ceil(np.clip(lo, -1e9, 1e9) - 0.5) - 1).astype(np.int64)
    x_fin = np.minimum(max_x[f_tri], np.floor(np.clip(hi, -1e9, 1e9) - 0.5) + 1).astype(np.int64)
    f_largo = np.where(vacio | (x_fin < x_ini), 0, x_fin - x_ini + 1)
    filas = (f_tri, f_y, x_ini, f_largo)
    n_pix = np.bincount(f_tri, weights=f_largo, minlength=K).astype(np.int64)

    # Código entero de textura por triángulo (−1 = sin textura) para elegir con NumPy
    unicas: list[Textura] = []
    codigo = np.full(K, -1, dtype=np.int64)
    for i, tx in enumerate(texturas):
        if tx is not None:
            for j, u in enumerate(unicas):
                if u is tx:
                    codigo[i] = j
                    break
            else:
                unicas.append(tx)
                codigo[i] = len(unicas) - 1

    # Búfer de visibilidad: por píxel, qué triángulo ganó y sus coordenadas baricéntricas.
    # El color se calcula al final, una sola vez por píxel visible (sombreado diferido):
    # así no se pinta un píxel que después tapará otro triángulo más cercano.
    n_px = fb.ancho * fb.alto
    gbuf = {"tri": np.full(n_px, -1, dtype=np.int64),
            "w0": np.zeros(n_px), "w1": np.zeros(n_px), "w2": np.zeros(n_px)}

    # Lotes de triángulos consecutivos [a, b) con hasta max_pixeles candidatos
    a, acumulado = 0, 0
    for i in range(K):
        if i > a and acumulado + n_pix[i] > max_pixeles:
            _rasterizar_grupo(fb, zb, gbuf, filas, inicio_filas[a], inicio_filas[i], pts_pant, area)
            a, acumulado = i, 0
        acumulado += n_pix[i]
    _rasterizar_grupo(fb, zb, gbuf, filas, inicio_filas[a], inicio_filas[K], pts_pant, area)

    _sombrear(fb, gbuf, inv_w, uvs, colores_vert, unicas, codigo)


def _rasterizar_grupo(fb, zb, gbuf, filas, fila_a, fila_b, pts_pant, area) -> None:
    """Visibilidad de un lote: deja en `gbuf` el triángulo más cercano de cada píxel."""
    f_tri, f_y, x_ini, f_largo = (arr[fila_a:fila_b] for arr in filas)
    total = int(f_largo.sum())
    if total == 0:
        return
    # Un candidato por píxel de cada tramo: (triángulo, x, y)
    k = np.arange(total) - np.repeat(np.cumsum(f_largo) - f_largo, f_largo)
    tri = np.repeat(f_tri, f_largo)
    px = np.repeat(x_ini, f_largo) + k
    py = np.repeat(f_y, f_largo)
    gx, gy = px + 0.5, py + 0.5

    P = pts_pant[tri]
    x0, y0, z0 = P[:, 0, 0], P[:, 0, 1], P[:, 0, 2]
    x1, y1, z1 = P[:, 1, 0], P[:, 1, 1], P[:, 1, 2]
    x2, y2, z2 = P[:, 2, 0], P[:, 2, 1], P[:, 2, 2]
    inv_area = 1.0 / area[tri]
    w0 = ((x1 - gx) * (y2 - gy) - (y1 - gy) * (x2 - gx)) * inv_area
    w1 = ((x2 - gx) * (y0 - gy) - (y2 - gy) * (x0 - gx)) * inv_area
    w2 = 1.0 - w0 - w1
    dentro = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    if not np.any(dentro):
        return
    tri, px, py = tri[dentro], px[dentro], py[dentro]
    w0, w1, w2 = w0[dentro], w1[dentro], w2[dentro]
    z = w0 * z0[dentro] + w1 * z1[dentro] + w2 * z2[dentro]

    # Visibilidad (z-buffer por lotes): por cada píxel se busca la profundidad mínima entre
    # sus candidatos con np.minimum.at; gana el candidato que la alcanza y además está más
    # cerca que lo ya dibujado. En empate gana el de menor índice, es decir, el triángulo
    # que se habría dibujado antes (igual que dibujando triángulo por triángulo).
    pid = py * fb.ancho + px
    n_px = fb.ancho * fb.alto
    prof = zb.profundidad.reshape(-1)
    z_min = np.full(n_px, np.inf)
    np.minimum.at(z_min, pid, z)
    cand = np.nonzero((z == z_min[pid]) & (z < prof[pid]))[0]
    if len(cand) == 0:
        return
    primero = np.full(n_px, len(z), dtype=np.int64)
    np.minimum.at(primero, pid[cand], cand)
    sel = cand[primero[pid[cand]] == cand]
    prof[pid[sel]] = z[sel]
    destino = pid[sel]
    gbuf["tri"][destino] = tri[sel]
    gbuf["w0"][destino], gbuf["w1"][destino], gbuf["w2"][destino] = w0[sel], w1[sel], w2[sel]


def _sombrear(fb, gbuf, inv_w, uvs, colores_vert, unicas, codigo) -> None:
    """Color (Gouraud o plano) y textura con corrección de perspectiva de cada píxel visible."""
    visibles = np.where(gbuf["tri"] >= 0)[0]
    if len(visibles) == 0:
        return
    t = gbuf["tri"][visibles]
    f0, f1, f2 = gbuf["w0"][visibles], gbuf["w1"][visibles], gbuf["w2"][visibles]
    C = colores_vert[t]
    color = f0[:, None] * C[:, 0] + f1[:, None] * C[:, 1] + f2[:, None] * C[:, 2]

    # Textura con corrección de perspectiva, por grupos de triángulos con la misma imagen
    cod = codigo[t]
    for j, tex in enumerate(unicas):
        m = cod == j
        if not np.any(m):
            continue
        IW, UV = inv_w[t[m]], uvs[t[m]]
        a0, a1, a2 = f0[m], f1[m], f2[m]
        denom = a0 * IW[:, 0] + a1 * IW[:, 1] + a2 * IW[:, 2]
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        u = (a0 * UV[:, 0, 0] * IW[:, 0] + a1 * UV[:, 1, 0] * IW[:, 1] + a2 * UV[:, 2, 0] * IW[:, 2]) / denom
        v = (a0 * UV[:, 0, 1] * IW[:, 0] + a1 * UV[:, 1, 1] * IW[:, 1] + a2 * UV[:, 2, 1] * IW[:, 2]) / denom
        color[m] = tex.muestrear_vectorizado(u, v) * color[m] * 1.25
    color = np.clip(color, 0.0, 1.0)

    fb.datos.reshape(-1, 3)[visibles] = (color * 255.0).astype(np.uint8)


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
