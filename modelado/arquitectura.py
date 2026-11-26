"""Extrusión B-Rep de muros y CSG de diferencia para puertas y ventanas."""
import numpy as np
from .malla import Malla


def _caja_local_muro(
    s0: float,
    s1: float,
    y0: float,
    y1: float,
    w0: float,
    w1: float,
    p1_2d: np.ndarray,
    dir_2d: np.ndarray,
    norm_2d: np.ndarray,
    color: tuple[int, int, int] = (230, 224, 212),
    textura_id: str | None = "pared",
) -> Malla:
    """
    Construye un prisma cerrado en coordenadas locales del muro (s, y, w)
    y lo proyecta al mundo 3D con convención X = x_2d, Y = altura, Z = -y_2d.
    """
    # 8 vértices en (s, y, w)
    loc = np.array([
        [s0, y0, w0],  # 0
        [s1, y0, w0],  # 1
        [s1, y1, w0],  # 2
        [s0, y1, w0],  # 3
        [s0, y0, w1],  # 4
        [s1, y0, w1],  # 5
        [s1, y1, w1],  # 6
        [s0, y1, w1],  # 7
    ], dtype=np.float64)

    # En 2D: pos_2d = p1 + s*dir + w*norm. En 3D: X = pos_2d[0], Y = y, Z = -pos_2d[1]
    mundo = np.zeros_like(loc)
    mundo[:, 0] = p1_2d[0] + loc[:, 0] * dir_2d[0] + loc[:, 2] * norm_2d[0]
    mundo[:, 1] = loc[:, 1]
    mundo[:, 2] = -(p1_2d[1] + loc[:, 0] * dir_2d[1] + loc[:, 2] * norm_2d[1])

    # Verificar quiralidad de la base (e_s, +Y, e_w) en 3D para garantizar normales salientes
    e_s = np.array([dir_2d[0], 0.0, -dir_2d[1]])
    e_y = np.array([0.0, 1.0, 0.0])
    e_w = np.array([norm_2d[0], 0.0, -norm_2d[1]])
    det = float(np.dot(np.cross(e_s, e_y), e_w))

    if det > 0:
        caras = np.array([
            [4, 5, 6], [4, 6, 7],  # +w
            [1, 0, 3], [1, 3, 2],  # -w
            [0, 4, 7], [0, 7, 3],  # -s
            [5, 1, 2], [5, 2, 6],  # +s
            [7, 6, 2], [7, 2, 3],  # +y
            [0, 1, 5], [0, 5, 4],  # -y
        ], dtype=np.int32)
    else:
        caras = np.array([
            [5, 4, 7], [5, 7, 6],  # +w
            [0, 1, 2], [0, 2, 3],  # -w
            [4, 0, 3], [4, 3, 7],  # -s
            [1, 5, 6], [1, 6, 2],  # +s
            [6, 7, 3], [6, 3, 2],  # +y
            [1, 0, 4], [1, 4, 5],  # -y
        ], dtype=np.int32)

    # B-Rep de 8 vértices y 12 triángulos; para el render cada cara lleva sus propios
    # vértices, así la normal (luz) y el mapeado cúbico (textura) son los de esa cara.
    m = Malla(vertices=mundo, caras=caras, color=color, textura_id=textura_id, nombre="bloque_muro")
    m = m.separar_vertices()
    m.aplicar_mapeado_uv(tipo="cubico", escala=0.5)
    return m


def extruir_muro(
    p1: tuple[float, float],
    p2: tuple[float, float],
    espesor: float = 0.15,
    altura: float = 2.60,
    color: tuple[int, int, int] = (230, 224, 212),
    textura_id: str | None = "pared",
) -> Malla:
    """Extruye un segmento 2D en un muro 3D B-Rep cerrado en sus extremos."""
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    d = b - a
    largo = float(np.linalg.norm(d))
    if largo < 1e-6:
        return Malla(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), color=color)

    u = d / largo
    n = np.array([-u[1], u[0]], dtype=np.float64)
    h = espesor / 2.0
    m = _caja_local_muro(-h, largo + h, 0.0, altura, -h, h, a, u, n, color, textura_id)
    m.nombre = "muro"
    return m


def csg_diferencia(
    p1: tuple[float, float],
    p2: tuple[float, float],
    espesor: float,
    altura: float,
    aberturas: list[dict],
    color: tuple[int, int, int] = (230, 224, 212),
    textura_id: str | None = "pared",
) -> Malla:
    """
    Aplica diferencia CSG (Muro - Vanos) subdividiendo el muro en bloques ortogonales locales:
    tramos laterales, antepecho (bajo ventanas) y dintel (sobre puertas y ventanas).

    Cada elemento de `aberturas` acepta:
      - 'pos': posición normalizada [0, 1] del centro sobre el muro (o 'inicio' en metros)
      - 'ancho': ancho del vano en metros
      - 'tipo': 'puerta' o 'ventana' (opcional si se indican 'y_min' y 'y_max')
    """
    if not aberturas:
        return extruir_muro(p1, p2, espesor, altura, color, textura_id)

    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    d = b - a
    largo = float(np.linalg.norm(d))
    if largo < 1e-6:
        return Malla(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), color=color)

    u = d / largo
    n = np.array([-u[1], u[0]], dtype=np.float64)
    h = espesor / 2.0

    # Normalizar especificaciones de vanos a intervalos [s_ini, s_fin, y_min, y_max]
    vanos_norm = []
    for ab in aberturas:
        ancho = float(ab.get("ancho", 0.9))
        if "inicio" in ab:
            s_ini = float(ab["inicio"])
        else:
            centro = float(ab.get("pos", 0.5)) * largo
            s_ini = centro - ancho / 2.0
        s_fin = s_ini + ancho

        tipo = ab.get("tipo", "puerta")
        if "y_min" in ab and "y_max" in ab:
            y_min, y_max = float(ab["y_min"]), float(ab["y_max"])
        elif tipo == "ventana":
            y_min, y_max = 0.90, min(2.10, altura - 0.1)
        else:  # puerta
            y_min, y_max = 0.0, min(2.10, altura - 0.1)

        vanos_norm.append((s_ini, s_fin, y_min, y_max))

    vanos_norm.sort(key=lambda item: item[0])

    bloques: list[Malla] = []
    cursor = -h
    fin_muro = largo + h

    for s_ini, s_fin, y_min, y_max in vanos_norm:
        s0 = max(cursor, min(s_ini, largo))
        s1 = max(s0, min(s_fin, largo))

        if s0 - cursor > 1e-4:
            bloques.append(_caja_local_muro(cursor, s0, 0.0, altura, -h, h, a, u, n, color, textura_id))
        if s1 - s0 > 1e-4 and y_min > 1e-4:
            bloques.append(_caja_local_muro(s0, s1, 0.0, y_min, -h, h, a, u, n, color, textura_id))
        if s1 - s0 > 1e-4 and altura - y_max > 1e-4:
            bloques.append(_caja_local_muro(s0, s1, y_max, altura, -h, h, a, u, n, color, textura_id))
        cursor = s1

    if fin_muro - cursor > 1e-4:
        bloques.append(_caja_local_muro(cursor, fin_muro, 0.0, altura, -h, h, a, u, n, color, textura_id))

    return Malla.combinar(bloques, nombre="muro_csg")


def marco_y_hoja(
    p1: tuple[float, float],
    p2: tuple[float, float],
    espesor: float,
    altura: float,
    abertura: dict,
) -> list[Malla]:
    """
    Piezas que "visten" el vano que `csg_diferencia` dejó abierto, en las mismas coordenadas
    locales del muro (s a lo largo, y vertical, w en el espesor):
      - puerta: marco (dos jambas y dintel) y hoja abierta a 90° con bisagra en el lado
        izquierdo, igual que el arco que dibuja el plano 2D;
      - ventana: marco con alféizar, una lámina de vidrio (opaca, celeste: el rasterizador no
        tiene transparencia) y una cruz de travesaños.
    Devuelve una malla por pieza; `abertura` usa las mismas claves que en `csg_diferencia`.
    """
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    d = b - a
    largo = float(np.linalg.norm(d))
    if largo < 1e-6:
        return []
    u = d / largo
    n = np.array([-u[1], u[0]], dtype=np.float64)
    h = espesor / 2.0

    ancho = float(abertura.get("ancho", 0.9))
    centro = float(abertura.get("pos", 0.5)) * largo
    s0, s1 = centro - ancho / 2.0, centro + ancho / 2.0
    tipo = abertura.get("tipo", "puerta")
    if tipo == "ventana":
        y0, y1 = 0.90, min(2.10, altura - 0.1)
    else:
        y0, y1 = 0.0, min(2.10, altura - 0.1)

    madera = (150, 108, 68)
    g = 0.05                                   # ancho de las piezas del marco
    w = h + 0.01                               # el marco sobresale 1 cm por cada cara del muro

    def caja(sa, sb, ya, yb, wa, wb, color, textura):
        return _caja_local_muro(sa, sb, ya, yb, wa, wb, a, u, n, color, textura)

    piezas = [caja(s0, s0 + g, y0, y1, -w, w, madera, "madera"),          # jamba izquierda
              caja(s1 - g, s1, y0, y1, -w, w, madera, "madera"),          # jamba derecha
              caja(s0 + g, s1 - g, y1 - g, y1, -w, w, madera, "madera")]  # dintel
    nombres = ["marco_jamba_izq", "marco_jamba_der", "marco_dintel"]

    if tipo == "ventana":
        piezas.append(caja(s0 + g, s1 - g, y0, y0 + g, -w, w, madera, "madera"))        # base del marco
        nombres.append("marco_base")
        piezas.append(caja(s0 - 0.03, s1 + 0.03, y0 - 0.02, y0 + 0.03, -h - 0.04, h + 0.03,
                           (225, 220, 210), None))                                      # alféizar
        nombres.append("alfeizar")
        piezas.append(caja(s0 + g, s1 - g, y0 + g, y1 - g, -0.006, 0.006, (178, 212, 240), None))
        nombres.append("vidrio")
        ym = (y0 + y1) / 2.0
        piezas.append(caja(centro - 0.015, centro + 0.015, y0 + g, y1 - g, -0.02, 0.02, madera, "madera"))
        piezas.append(caja(s0 + g, s1 - g, ym - 0.015, ym + 0.015, -0.02, 0.02, madera, "madera"))
        nombres += ["travesano_vertical", "travesano_horizontal"]
    else:
        # Hoja abierta 90°: pegada a la jamba de la bisagra y extendida hacia +n (como el arco 2D)
        piezas.append(caja(s0 + g, s0 + g + 0.04, 0.0, y1 - g, 0.0, ancho - 2 * g, (140, 96, 60), "madera"))
        nombres.append("hoja_puerta")

    for m, nom in zip(piezas, nombres):
        m.nombre = nom
    return piezas


def crear_piso(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int] = (185, 155, 120),
    textura_id: str | None = "piso",
) -> Malla:
    """Crea una malla plana subdividida en Y=0 para el piso de la vivienda."""
    nx, nz = 6, 6
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, nz + 1)
    verts, uvs = [], []
    for j in range(nz + 1):
        for i in range(nx + 1):
            verts.append([xs[i], 0.0, -ys[j]])
            uvs.append([xs[i] * 0.5, ys[j] * 0.5])

    caras = []
    cols = nx + 1
    for j in range(nz):
        for i in range(nx):
            p00 = j * cols + i
            p10 = p00 + 1
            p01 = (j + 1) * cols + i
            p11 = p01 + 1
            # Normal hacia +Y: (p10 - p00) x (p01 - p00) = (+X) x (-Z) = +Y
            caras.append([p00, p10, p01])
            caras.append([p10, p11, p01])

    return Malla(
        vertices=np.array(verts, dtype=np.float64),
        caras=np.array(caras, dtype=np.int32),
        uvs=np.array(uvs, dtype=np.float64),
        color=color,
        textura_id=textura_id,
        nombre="piso",
    )
