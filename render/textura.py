"""Carga de mapas de bits con Pillow y generación de texturas procedurales por defecto."""
import pathlib
import numpy as np
from PIL import Image

EXTENSIONES_IMAGEN = (".png", ".jpg", ".jpeg", ".bmp")
LADO_MAXIMO = 512          # una foto grande se reduce al cargarla (memoria y tiempo de carga)
TEXTURAS_BASICAS = ("piso", "pared", "madera", "ceramica")


class Textura:
    """Mapa de bits RGB (alto, ancho, 3) normalizado en [0, 1] con muestreo repetido (wrap)."""

    def __init__(self, datos: np.ndarray, nombre: str = "textura"):
        arr = np.asarray(datos, dtype=np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0
        self.datos = np.clip(arr, 0.0, 1.0)
        self.alto, self.ancho = self.datos.shape[:2]
        self.nombre = nombre

    @classmethod
    def desde_archivo(cls, ruta: str | pathlib.Path, lado_maximo: int = LADO_MAXIMO) -> "Textura":
        """Carga un mapa de bits (PNG, JPG, BMP…) con Pillow. Si es más grande que
        `lado_maximo` píxeles se reduce conservando la proporción."""
        img = Image.open(ruta).convert("RGB")
        if max(img.size) > lado_maximo:
            img.thumbnail((lado_maximo, lado_maximo), Image.LANCZOS)
        return cls(np.asarray(img, dtype=np.float64) / 255.0, nombre=pathlib.Path(ruta).stem.lower())

    def guardar(self, ruta: str | pathlib.Path) -> None:
        p = pathlib.Path(ruta)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((self.datos * 255.0).astype(np.uint8)).save(p)

    def muestrear_vectorizado(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Muestrea colores RGB (N, 3) para arreglos de coordenadas (u, v) con repetición."""
        u_wrap = np.mod(u, 1.0)
        v_wrap = np.mod(1.0 - v, 1.0)  # origen arriba-izquierda en imagen
        tx = np.minimum((u_wrap * (self.ancho - 1)).astype(np.int32), self.ancho - 1)
        ty = np.minimum((v_wrap * (self.alto - 1)).astype(np.int32), self.alto - 1)
        return self.datos[ty, tx]


def generar_texturas_defecto(carpeta: str | pathlib.Path | None = None) -> dict[str, Textura]:
    """
    Genera texturas procedurales realistas (piso de madera/parquet, pared estucada,
    madera de muebles y cerámica) y opcionalmente las guarda en `assets/texturas/`.
    """
    tam = 128
    yy, xx = np.mgrid[0:tam, 0:tam].astype(np.float64)

    # 1. Piso de parquet / tablones de madera
    tabla_y = (yy // 16).astype(int)
    desfase = (tabla_y % 2) * 32
    junta_y = (yy % 16) < 1.2
    junta_x = ((xx + desfase) % 64) < 1.2
    veta = 0.06 * np.sin(yy * 0.8 + 2.5 * np.sin(xx * 0.08 + tabla_y))
    tono_tabla = ((tabla_y * 37) % 11) / 110.0
    piso = np.zeros((tam, tam, 3), dtype=np.float64)
    piso[..., 0] = 0.74 + veta + tono_tabla
    piso[..., 1] = 0.56 + veta * 0.8 + tono_tabla * 0.8
    piso[..., 2] = 0.36 + veta * 0.6 + tono_tabla * 0.5
    piso[junta_y | junta_x] *= 0.55

    # 2. Pared estucada / ladrillo suave
    ruido = 0.04 * np.sin(xx * 0.35) * np.cos(yy * 0.35) + 0.02 * np.sin((xx + yy) * 0.7)
    pared = np.zeros((tam, tam, 3), dtype=np.float64)
    pared[..., 0] = 0.91 + ruido
    pared[..., 1] = 0.88 + ruido
    pared[..., 2] = 0.82 + ruido

    # 3. Madera vetada para muebles
    veta_m = 0.08 * np.sin(xx * 0.25 + 3.0 * np.sin(yy * 0.06))
    madera = np.zeros((tam, tam, 3), dtype=np.float64)
    madera[..., 0] = 0.62 + veta_m
    madera[..., 1] = 0.42 + veta_m * 0.75
    madera[..., 2] = 0.24 + veta_m * 0.5

    # 4. Cerámica decorativa con bandas geométricas
    banda = 0.08 * np.cos(yy * 2.0 * np.pi / 32.0)
    ceramica = np.zeros((tam, tam, 3), dtype=np.float64)
    ceramica[..., 0] = 0.80 + banda
    ceramica[..., 1] = 0.48 + banda * 0.6
    ceramica[..., 2] = 0.34 + banda * 0.4

    texturas = {
        "piso": Textura(piso, "piso"),
        "pared": Textura(pared, "pared"),
        "madera": Textura(madera, "madera"),
        "ceramica": Textura(ceramica, "ceramica"),
    }

    if carpeta is not None:
        base = pathlib.Path(carpeta)
        for nombre, tex in texturas.items():
            if not any((base / f"{nombre}{ext}").exists() for ext in EXTENSIONES_IMAGEN):
                tex.guardar(base / f"{nombre}.png")

    return texturas


def cargar_texturas(carpeta: str | pathlib.Path | None) -> tuple[dict[str, Textura], list[str]]:
    """
    Texturas de la escena a partir de mapas de bits. Cada imagen de `carpeta` (PNG, JPG o
    BMP) se carga como textura con el nombre del archivo (`ladrillo.jpg` → "ladrillo").
    Si falta alguna textura básica (piso, pared, madera, cerámica) se genera por código y se
    guarda en la carpeta, así siempre existe y se puede reemplazar por una foto.
    Devuelve (texturas, avisos): los avisos describen imágenes que no se pudieron leer.
    """
    texturas = generar_texturas_defecto(carpeta)
    avisos: list[str] = []
    if carpeta is None:
        return texturas, avisos
    for ruta in sorted(pathlib.Path(carpeta).iterdir()):
        if ruta.suffix.lower() in EXTENSIONES_IMAGEN:
            try:
                tex = Textura.desde_archivo(ruta)
                texturas[tex.nombre] = tex
            except Exception as exc:       # archivo dañado o formato no soportado
                avisos.append(f"No se pudo cargar {ruta.name}: {exc}")
    return texturas, avisos
