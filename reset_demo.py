"""Deja la demo en su estado inicial y verifica que funcione (prueba de humo).

1. Quita de data/ los documentos del remate (los que están en remate/).
2. Reindexa TODO data/ desde cero: la base queda idéntica a los documentos actuales
   (si cambiaste la agenda, aquí se actualiza sola).
3. Precarga los modelos en la GPU.
4. Prueba de humo: la pregunta ancla debe responder Gonzalo Alfaro y
   la pregunta del remate debe responder "No lo sé".

Uso (con la app CERRADA):  python reset_demo.py
O con un clic:             reset_demo.bat   (reset + abre la app)
"""
import sys
import time
from pathlib import Path

from ingest import CARPETA_DATOS, EXTENSIONES, indexar
from rag import embeddings, responder

CARPETA_REMATE = Path("remate")

ANCLA = "¿Quién habla después del almuerzo en el Main Stage?"
REMATE = "¿Dónde puedo descargar el código de la charla de Heberht?"


def quitar_apuntes_de_data():
    for archivo in sorted(CARPETA_REMATE.glob("*")):
        copia = CARPETA_DATOS / archivo.name
        if copia.exists():
            copia.unlink()
            print(f"  borrado data/{archivo.name}")


def reindexar_todo():
    rutas = sorted(r for r in CARPETA_DATOS.iterdir() if r.suffix.lower() in EXTENSIONES)
    resumen, segundos, total = indexar(rutas, desde_cero=True)
    for archivo, n in resumen:
        print(f"  {archivo}: {n} fichas")
    print(f"  fichas en la base: {total} ({segundos:.1f} s)")


def prueba(nombre, pregunta, condicion):
    inicio = time.perf_counter()
    respuesta, _ = responder(pregunta)
    ok = condicion(respuesta)
    print(f"  {'✅' if ok else '❌'} {nombre} ({time.perf_counter() - inicio:.1f} s): {respuesta}")
    return ok


def main():
    print("1/4 · Quitando los apuntes del remate…")
    quitar_apuntes_de_data()

    print("2/4 · Reindexando data/ desde cero…")
    reindexar_todo()

    print("3/4 · Precargando modelos en la GPU…")
    embeddings().embed_query("hola")

    print("4/4 · Prueba de humo…")
    resultados = [
        prueba("ancla", ANCLA, lambda r: "Gonzalo Alfaro" in r),
        prueba("remate (antes de subir los apuntes)", REMATE, lambda r: "No lo sé" in r),
    ]

    if all(resultados):
        print("\n✅ Demo lista. Verifica la GPU con: ollama ps")
    else:
        print("\n❌ Algo no está como se espera. Revisa antes de subir al escenario.")
        sys.exit(1)


if __name__ == "__main__":
    main()
