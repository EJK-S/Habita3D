"""Ejecutor mínimo por si no está instalado pytest:  python tests/ejecutar_tests.py
(Con pytest instalado basta con:  pytest -v)"""
import importlib, sys, traceback, pathlib

raiz = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(raiz))
ok = fallos = 0
for archivo in sorted((raiz / "tests").glob("test_*.py")):
    mod = importlib.import_module(f"tests.{archivo.stem}") if False else importlib.import_module(archivo.stem) \
        if str(archivo.parent) in sys.path else None
    if mod is None:
        sys.path.insert(0, str(archivo.parent))
        mod = importlib.import_module(archivo.stem)
    for nombre in dir(mod):
        if nombre.startswith("test_"):
            try:
                getattr(mod, nombre)()
                ok += 1
                print(f"  OK   {archivo.stem}::{nombre}")
            except Exception:
                fallos += 1
                print(f"  FALLA {archivo.stem}::{nombre}")
                traceback.print_exc()
print(f"\n{ok} pruebas correctas, {fallos} fallidas")
sys.exit(1 if fallos else 0)
