# Habita3D – Diseñador interactivo de interiores 2D y 3D
Proyecto final de Computación Visual (UNMSM). Render por software con NumPy; Pillow solo guarda imágenes.

## Estado
| Fase | Contenido | Estado |
|---|---|---|
| 1 | Base 2D: Framebuffer, DDA, Bresenham, Wu, círculo, relleno, grosor y color | Hecha y probada |
| 2 | Plano: matrices 3×3, muros, puertas, ventanas y muebles en planta | Hecha y probada |
| 3 | Vista 3D (matrices 4×4 ya listas; falta cámara y proyecciones) | Pendiente |
| 4 | Modelado: B-Rep, CSG, Bézier, revolución, fractales | Pendiente |
| 5 | Render: z-buffer, texturas, Phong, Gouraud | Pendiente |
| 6 | Interfaz, integración, rendimiento | Pendiente |

## Uso
    pip install numpy pillow pytest
    pytest -v                      # o: python tests/ejecutar_tests.py
    python demos/fase1_2.py        # regenera las figuras en docs/figuras/
