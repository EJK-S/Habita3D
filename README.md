# Habita3D – Diseñador interactivo de interiores 2D y 3D
Proyecto final de Computación Visual (UNMSM). Renderizador 2D y 3D implementado por software en Python y NumPy; Pillow se usa para texturas y capturas, y pygame para la ventana e interacción.

## Estado del proyecto
| Fase | Contenido | Estado |
|---|---|---|
| 1 | Base 2D: Framebuffer, DDA, Bresenham, Wu, círculo, relleno, grosor y color | Completada |
| 2 | Plano: matrices 3×3, muros, puertas, ventanas y muebles en planta | Completada |
| 3 | Vista 3D: matrices 4×4, cámara sintética, volumen de visión y proyecciones | Completada |
| 4 | Modelado: B-Rep, CSG para vanos, curvas de Bézier, revolución y fractales | Completada |
| 5 | Render: z-buffer, texturas con corrección de perspectiva, Phong y Gouraud | Completada |
| 6 | Interfaz interactiva (`pygame`), pruebas unitarias y mediciones de rendimiento | Completada |

## Uso
```bash
pip install -r requirements.txt
pytest -v                      # o: python tests/ejecutar_tests.py
python demos/fase1_2.py        # regenera fig_01 a fig_05 en docs/figuras/
python demos/fase3_5.py        # regenera fig_06 a fig_11 y mide rendimiento
python main.py                 # abre la aplicación interactiva en ventana
```
