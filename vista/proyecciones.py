"""Proyecciones geométricas (ortográfica y perspectiva), recorte y backface culling."""
import math
import numpy as np
from mathlib import Mat4


def proy_ortografica(
    izquierda: float,
    derecha: float,
    abajo: float,
    arriba: float,
    cercano: float = 0.1,
    lejano: float = 100.0,
) -> Mat4:
    """
    Matriz de proyección ortográfica (volumen en forma de paralelepípedo -> NDC [-1, 1]^3).
    En espacio de cámara, los objetos visibles tienen z en [-lejano, -cercano].
    """
    rl = derecha - izquierda
    tb = arriba - abajo
    fn = lejano - cercano
    m = np.array([
        [2.0 / rl, 0.0, 0.0, -(derecha + izquierda) / rl],
        [0.0, 2.0 / tb, 0.0, -(arriba + abajo) / tb],
        [0.0, 0.0, -2.0 / fn, -(lejano + cercano) / fn],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    return Mat4(m)


def proy_perspectiva(
    fov_grados: float = 60.0,
    aspecto: float = 4.0 / 3.0,
    cercano: float = 0.1,
    lejano: float = 100.0,
) -> Mat4:
    """
    Matriz de proyección en perspectiva (tronco de pirámide -> espacio de recorte homogéneo).
    Convención de vectores columna (P_clip = P · P_cam).
    """
    f = 1.0 / math.tan(math.radians(fov_grados) / 2.0)
    fn = lejano - cercano
    m = np.array([
        [f / aspecto, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, -(lejano + cercano) / fn, -(2.0 * lejano * cercano) / fn],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=float)
    return Mat4(m)


class VolumenVision:
    """
    Gestiona el frustum / volumen de visualización, descarte de caras traseras (backface culling)
    y recorte contra el plano cercano (near-plane clipping) + mapeado a pantalla.
    """

    def __init__(self, cercano: float = 0.1, lejano: float = 100.0):
        self.cercano = cercano
        self.lejano = lejano

    @staticmethod
    def filtrar_caras_traseras(
        verts_cam: np.ndarray,
        caras: np.ndarray,
        es_ortografica: bool = False,
    ) -> np.ndarray:
        """
        Devuelve una máscara booleana de triángulos visibles (front-facing).
        En espacio de cámara, el ojo está en (0, 0, 0) mirando hacia -Z.
        """
        if len(caras) == 0:
            return np.zeros(0, dtype=bool)
        v0 = verts_cam[caras[:, 0]]
        v1 = verts_cam[caras[:, 1]]
        v2 = verts_cam[caras[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        if es_ortografica:
            # Rayos paralelos al eje -Z: visible si n.z > 0
            return n[:, 2] > 1e-9
        # En perspectiva, el vector desde el triángulo hacia la cámara (0,0,0) es -v0
        dot = np.einsum("ij,ij->i", n, -v0)
        return dot > 1e-9

    def recortar_y_proyectar_triangulo(
        self,
        p_cam: np.ndarray,
        uvs: np.ndarray,
        colores: np.ndarray,
        mat_proy: Mat4,
        ancho: int,
        alto: int,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Recorta un triángulo (3 vértices en espacio de cámara) contra el plano z = -cercano,
        proyecta a coordenadas de pantalla y devuelve 0, 1 o 2 triángulos listos para rasterizar.
        Cada elemento retornado es: (pts_pant (3,3) [x, y, z_ndc], inv_w (3,), uvs (3,2), cols (3,3)).
        """
        z_near = -self.cercano
        dentro = p_cam[:, 2] <= z_near
        n_dentro = int(np.sum(dentro))

        if n_dentro == 0:
            return []

        if n_dentro == 3:
            polis = [(p_cam, uvs, colores)]
        else:
            # Recorte Sutherland-Hodgman contra z <= z_near
            verts_out, uvs_out, cols_out = [], [], []
            for i in range(3):
                j = (i + 1) % 3
                p_a, p_b = p_cam[i], p_cam[j]
                in_a, in_b = dentro[i], dentro[j]

                if in_a:
                    verts_out.append(p_a)
                    uvs_out.append(uvs[i])
                    cols_out.append(colores[i])

                if in_a != in_b:
                    dz = p_b[2] - p_a[2]
                    t = (z_near - p_a[2]) / dz if abs(dz) > 1e-12 else 0.0
                    t = max(0.0, min(1.0, float(t)))
                    verts_out.append(p_a + t * (p_b - p_a))
                    uvs_out.append(uvs[i] + t * (uvs[j] - uvs[i]))
                    cols_out.append(colores[i] + t * (colores[j] - colores[i]))

            if len(verts_out) < 3:
                return []
            v_arr = np.array(verts_out, dtype=np.float64)
            u_arr = np.array(uvs_out, dtype=np.float64)
            c_arr = np.array(cols_out, dtype=np.float64)
            polis = [(v_arr[:3], u_arr[:3], c_arr[:3])]
            if len(v_arr) == 4:
                polis.append((v_arr[[0, 2, 3]], u_arr[[0, 2, 3]], c_arr[[0, 2, 3]]))

        resultados = []
        M = mat_proy.m
        resultados.extend(self._proyectar(polis, M, ancho, alto))
        return resultados

    @staticmethod
    def _proyectar(polis, M, ancho, alto):
        """Proyecta triángulos ya recortados (lista de (verts, uvs, cols)) a pantalla."""
        resultados = []
        for v_tri, uv_tri, col_tri in polis:
            hom = np.hstack([v_tri, np.ones((3, 1), dtype=np.float64)])
            clip = hom @ M.T
            w = clip[:, 3]
            if np.any(np.abs(w) < 1e-9):
                continue
            inv_w = 1.0 / w
            ndc = clip[:, :3] * inv_w[:, None]

            # Descarte rápido fuera del frustum en X/Y/Z
            if (np.all(ndc[:, 0] < -1.0) or np.all(ndc[:, 0] > 1.0) or
                np.all(ndc[:, 1] < -1.0) or np.all(ndc[:, 1] > 1.0) or
                np.all(ndc[:, 2] < -1.0) or np.all(ndc[:, 2] > 1.0)):
                continue

            # Mapeado NDC [-1, 1] -> coordenadas de pantalla (Y hacia abajo en framebuffer)
            sx = (ndc[:, 0] + 1.0) * 0.5 * (ancho - 1)
            sy = (1.0 - ndc[:, 1]) * 0.5 * (alto - 1)
            sz = ndc[:, 2]
            pts_pant = np.column_stack((sx, sy, sz))
            resultados.append((pts_pant, inv_w, uv_tri, col_tri))

        return resultados

    def proyectar_lote(
        self,
        p_cam: np.ndarray,
        uvs: np.ndarray,
        colores: np.ndarray,
        mat_proy: Mat4,
        ancho: int,
        alto: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Versión por lotes de `recortar_y_proyectar_triangulo` para T triángulos a la vez
        (p_cam (T, 3, 3), uvs (T, 3, 2), colores (T, 3, 3)).

        Los triángulos que están enteros delante del plano cercano se proyectan y se
        descartan contra el frustum con operaciones vectorizadas; solo los que cruzan el
        plano cercano pasan por el recorte de Sutherland-Hodgman uno a uno.

        Devuelve (pts_pant (K,3,3), inv_w (K,3), uvs (K,3,2), cols (K,3,3), origen (K,)),
        donde `origen` es el índice del triángulo de entrada, en el mismo orden de entrada.
        """
        z_near = -self.cercano
        n_dentro = np.sum(p_cam[:, :, 2] <= z_near, axis=1)
        completos = np.where(n_dentro == 3)[0]
        cruzan = np.where((n_dentro > 0) & (n_dentro < 3))[0]

        # --- triángulos completos: proyección vectorizada ---
        P = p_cam[completos]
        hom = np.concatenate([P, np.ones(P.shape[:2] + (1,))], axis=2)
        clip = hom @ mat_proy.m.T                                  # (K, 3, 4)
        w = clip[:, :, 3]
        valido = np.all(np.abs(w) >= 1e-9, axis=1)
        inv_w = 1.0 / np.where(valido[:, None], w, 1.0)
        ndc = clip[:, :, :3] * inv_w[:, :, None]
        fuera = np.zeros(len(P), dtype=bool)
        for eje in range(3):
            fuera |= np.all(ndc[:, :, eje] < -1.0, axis=1) | np.all(ndc[:, :, eje] > 1.0, axis=1)
        quedan = valido & ~fuera

        ndc, inv_w = ndc[quedan], inv_w[quedan]
        pts = np.stack([
            (ndc[:, :, 0] + 1.0) * 0.5 * (ancho - 1),
            (1.0 - ndc[:, :, 1]) * 0.5 * (alto - 1),
            ndc[:, :, 2],
        ], axis=2)
        lista_pts, lista_iw = [pts], [inv_w]
        lista_uv, lista_col = [uvs[completos][quedan]], [colores[completos][quedan]]
        lista_origen = [completos[quedan]]

        # --- triángulos que cruzan el plano cercano: recorte uno a uno ---
        for i in cruzan:
            for pts_t, iw_t, uv_t, col_t in self.recortar_y_proyectar_triangulo(
                    p_cam[i], uvs[i], colores[i], mat_proy, ancho, alto):
                lista_pts.append(pts_t[None])
                lista_iw.append(iw_t[None])
                lista_uv.append(np.asarray(uv_t)[None])
                lista_col.append(np.asarray(col_t)[None])
                lista_origen.append(np.array([i]))

        origen = np.concatenate(lista_origen)
        orden = np.argsort(origen, kind="stable")          # mismo orden que triángulo a triángulo
        return (np.concatenate(lista_pts)[orden], np.concatenate(lista_iw)[orden],
                np.concatenate(lista_uv)[orden], np.concatenate(lista_col)[orden], origen[orden])
