"""Deja la demo en su estado inicial y verifica que funcione (prueba de humo).

1. Quita de data/ y de la base los documentos del remate (los que están en remate/).
2. Precarga los modelos en la GPU.
3. Prueba de humo: la pregunta ancla debe responder Gonzalo Alfaro y
   la pregunta del remate debe responder "No lo sé".

Uso (con la app CERRADA):  python reset_demo.py
O con un clic:             reset_demo.bat   (reset + abre la app)
"""
import sys
import time
from pathlib import Path

from rag import base_vectorial, embeddings, indice_palabras, responder

CARPETA_DATOS = Path("data")
CARPETA_REMATE = Path("remate")

ANCLA = "¿Quién habla después de la charla Más allá del Prompt en el Main Stage?"
REMATE = "¿Dónde puedo descargar el código de la charla de Heberht?"


def quitar_documentos_del_remate():
    db = base_vectorial()
    for archivo in sorted(CARPETA_REMATE.glob("*")):
        copia = CARPETA_DATOS / archivo.name
        if copia.exists():
            copia.unlink()
            print(f"  borrado data/{archivo.name}")
        db.delete(where={"archivo": archivo.name})
    indice_palabras.cache_clear()
    print(f"  fichas en la base: {db._collection.count()}")


def prueba(nombre, pregunta, condicion):
    inicio = time.perf_counter()
    respuesta, _ = responder(pregunta)
    ok = condicion(respuesta)
    print(f"  {'✅' if ok else '❌'} {nombre} ({time.perf_counter() - inicio:.1f} s): {respuesta}")
    return ok


def main():
    print("1/3 · Quitando los apuntes del remate…")
    quitar_documentos_del_remate()

    print("2/3 · Precargando modelos en la GPU…")
    embeddings().embed_query("hola")

    print("3/3 · Prueba de humo…")
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
