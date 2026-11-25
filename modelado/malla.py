"""Representación de fronteras (B-Rep) para objetos 3D en Habita3D."""
import numpy as np
from mathlib import Mat4


class Malla:
    """
    Malla poligonal B-Rep definida por vértices, caras triangulares,
    normales (por cara y por vértice) y coordenadas de textura (u, v).
    """

    def __init__(
        self,
        vertices: np.ndarray,
        caras: np.ndarray,
        uvs: np.ndarray | None = None,
        color: tuple[int, int, int] = (200, 200, 200),
        textura_id: str | None = None,
        nombre: str = "malla",
    ):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.caras = np.asarray(caras, dtype=np.int32).reshape(-1, 3)
        self.color = tuple(int(c) for c in color)
        self.textura_id = textura_id
        self.nombre = nombre

        if uvs is not None:
            self.uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        else:
            self.uvs = np.zeros((len(self.vertices), 2), dtype=np.float64)

        self.normales_cara = np.zeros((len(self.caras), 3), dtype=np.float64)
        self.normales_vertice = np.zeros((len(self.vertices), 3), dtype=np.float64)
        if len(self.caras) > 0:
            self.recalcular_normales()

    def recalcular_normales(self) -> None:
        """Calcula normales unitarias por cara (regla mano derecha CCW) y por vértice."""
        v0 = self.vertices[self.caras[:, 0]]
        v1 = self.vertices[self.caras[:, 1]]
        v2 = self.vertices[self.caras[:, 2]]

        normales = np.cross(v1 - v0, v2 - v0)
        longs = np.linalg.norm(normales, axis=1, keepdims=True)
        longs = np.where(longs < 1e-12, 1.0, longs)
        self.normales_cara = normales / longs

        acum = np.zeros_like(self.vertices)
        for k in range(3):
            np.add.at(acum, self.caras[:, k], self.normales_cara)
        norm_v = np.linalg.norm(acum, axis=1, keepdims=True)
        norm_v = np.where(norm_v < 1e-12, 1.0, norm_v)
        self.normales_vertice = acum / norm_v

    def aplicar_transformacion(self, mat: Mat4) -> "Malla":
        """Devuelve una nueva Malla transformada por la matriz homogénea 4×4 `mat`."""
        nuevos_v = mat.aplicar_puntos(self.vertices)
        return Malla(
            vertices=nuevos_v,
            caras=self.caras.copy(),
            uvs=self.uvs.copy(),
            color=self.color,
            textura_id=self.textura_id,
            nombre=self.nombre,
        )

    def aplicar_mapeado_uv(self, tipo: str = "cubico", escala: float = 1.0) -> None:
        """
        Genera coordenadas (u, v) mediante mapeado 'plano', 'cilindrico' o 'cubico'.
        """
        if len(self.vertices) == 0:
            return
        v = self.vertices
        if tipo == "plano":
            self.uvs = np.column_stack((v[:, 0] * escala, -v[:, 2] * escala))
        elif tipo == "cilindrico":
            centro = v.mean(axis=0)
            dx = v[:, 0] - centro[0]
            dz = v[:, 2] - centro[2]
            u = (np.arctan2(dz, dx) + np.pi) / (2.0 * np.pi)
            ymin, ymax = v[:, 1].min(), v[:, 1].max()
            ry = max(ymax - ymin, 1e-6)
            self.uvs = np.column_stack((u * escala, ((v[:, 1] - ymin) / ry) * escala))
        else:  # cubico / triplanar
            abs_n = np.abs(self.normales_vertice)
            dom = np.argmax(abs_n, axis=1)
            uv = np.zeros((len(v), 2), dtype=np.float64)
            uv[dom == 0] = v[dom == 0][:, [2, 1]] * escala
            uv[dom == 1] = v[dom == 1][:, [0, 2]] * escala
            uv[dom == 2] = v[dom == 2][:, [0, 1]] * escala
            self.uvs = uv

    @staticmethod
    def combinar(mallas: list["Malla"], nombre: str = "combinada") -> "Malla":
        """Combina una lista de mallas en una sola estructura B-Rep."""
        vs, cs, uvs = [], [], []
        offset = 0
        color_base = mallas[0].color if mallas else (200, 200, 200)
        tex_base = mallas[0].textura_id if mallas else None

        for m in mallas:
            if len(m.vertices) == 0:
                continue
            vs.append(m.vertices)
            cs.append(m.caras + offset)
            uvs.append(m.uvs)
            offset += len(m.vertices)

        if not vs:
            return Malla(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), nombre=nombre)

        return Malla(
            vertices=np.vstack(vs),
            caras=np.vstack(cs),
            uvs=np.vstack(uvs),
            color=color_base,
            textura_id=tex_base,
            nombre=nombre,
        )
