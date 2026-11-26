"""Pruebas unitarias del paquete render (ZBuffer, iluminación de Phong y rasterizador 3D)."""
import numpy as np
from raster2d import Framebuffer
from render import ZBuffer, Luz, sombreado_plano, sombreado_gouraud, rasterizar_triangulo


def test_zbuffer_oculta_superficie_trasera():
    fb = Framebuffer(40, 40, fondo=(0, 0, 0))
    zb = ZBuffer(40, 40)
    pts_lejos = np.array([[5.0, 5.0, 0.8], [35.0, 5.0, 0.8], [20.0, 35.0, 0.8]])
    pts_cerca = np.array([[5.0, 5.0, 0.2], [35.0, 5.0, 0.2], [20.0, 35.0, 0.2]])
    uvs = np.zeros((3, 2))
    inv_w = np.ones(3)

    rasterizar_triangulo(fb, zb, pts_cerca, inv_w, uvs, np.tile([1.0, 0.0, 0.0], (3, 1)), usar_textura=False)
    rasterizar_triangulo(fb, zb, pts_lejos, inv_w, uvs, np.tile([0.0, 0.0, 1.0], (3, 1)), usar_textura=False)

    assert tuple(fb.datos[15, 20]) == (255, 0, 0)


def test_imagenes_de_la_carpeta_se_usan_como_texturas():
    import pathlib, tempfile
    from PIL import Image
    from plano import Plano
    from escena import Escena
    with tempfile.TemporaryDirectory() as carpeta:
        carpeta = pathlib.Path(carpeta)
        Image.new("RGB", (64, 64), (220, 20, 20)).save(carpeta / "piso.png")          # piso propio: rojo
        Image.new("RGB", (2000, 1000), (20, 20, 220)).save(carpeta / "Ladrillo.jpg")  # foto grande
        (carpeta / "danada.png").write_bytes(b"esto no es una imagen")

        pl = Plano()
        pl.agregar_habitacion(0, 0, 4, 3)
        e = Escena(pl, carpeta_texturas=carpeta)

        assert np.allclose(e.texturas["piso"].datos.mean(axis=(0, 1)), np.array([220, 20, 20]) / 255, atol=0.01)
        assert "ladrillo" in e.texturas                                  # nombre del archivo
        assert max(e.texturas["ladrillo"].alto, e.texturas["ladrillo"].ancho) == 512
        assert (carpeta / "pared.png").exists()                          # básica que faltaba: generada
        assert any("danada.png" in a for a in e.avisos_texturas)         # no rompe la carga

        # Vista cenital del piso: con la imagen roja, el piso se ve rojo
        e.camara.mirar_a((2.0, 2.4, -1.5), (2.0, 0.0, -1.51))
        r, g, b = e.renderizar_3d(80, 60).leer_pixel(40, 30)
        assert r > 2 * g and r > 2 * b

        # Al reemplazar la imagen y recargar, cambia la textura
        Image.new("RGB", (64, 64), (20, 200, 20)).save(carpeta / "piso.png")
        e.recargar_texturas()
        r, g, b = e.renderizar_3d(80, 60).leer_pixel(40, 30)
        assert g > 2 * r and g > 2 * b


def test_iluminacion_phong_intensidad():
    luz = Luz(posicion=(0.0, 5.0, 0.0), intensidad_ambiental=0.2, intensidad_luz=0.8)
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    caras = np.array([[0, 1, 2]])
    normales = np.array([[0.0, 1.0, 0.0]])
    c_plano = sombreado_plano(verts, caras, normales, (200, 200, 200), np.array([0.0, 5.0, 0.0]), luz)
    c_gouraud = sombreado_gouraud(verts, np.tile([0.0, 1.0, 0.0], (3, 1)), (200, 200, 200), np.array([0.0, 5.0, 0.0]), luz)
    assert np.all(c_plano > 0.7)
    # En el vértice (0, 0, 0) directamente bajo la luz, Gouraud captura el pico especular completo
    assert np.all(c_gouraud[0] > 0.95)
