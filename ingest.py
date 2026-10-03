"""Ingesta: lee data/, parte en fichas (chunks), genera embeddings y guarda en ChromaDB con su fuente.

Uso:
    python ingest.py                      # reindexa TODA la carpeta data/ desde cero
    python ingest.py data/apuntes.pdf     # agrega o actualiza solo ese archivo (remate en vivo)

app.py usa la función indexar() para el botón "Agregar documento".
"""
import argparse
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import base_vectorial, indice_palabras

CARPETA_DATOS = Path("data")
EXTENSIONES = {".md", ".txt", ".pdf"}

# Corta primero por secciones de Markdown (## / ###), luego por párrafos y frases.
divisor = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=80,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
)


def cargar(ruta: Path):
    if ruta.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(ruta)).load()
    else:
        docs = TextLoader(str(ruta), encoding="utf-8").load()
    for d in docs:
        d.metadata["archivo"] = ruta.name
        pagina = d.metadata.get("page")
        d.metadata["fuente"] = f"{ruta.name}, pág. {pagina + 1}" if pagina is not None else ruta.name
    return docs


def indexar(rutas, desde_cero=False):
    """Indexa los archivos dados. Devuelve ([(archivo, n_fichas)], segundos, total_en_base)."""
    inicio = time.perf_counter()
    db = base_vectorial()
    if desde_cero:
        db.reset_collection()
    resumen = []
    for ruta in map(Path, rutas):
        fichas = divisor.split_documents(cargar(ruta))
        db.delete(where={"archivo": ruta.name})  # si ya estaba, se reemplaza (no se duplica)
        db.add_documents(fichas, ids=[f"{ruta.name}-{i}" for i in range(len(fichas))])
        resumen.append((ruta.name, len(fichas)))
    indice_palabras.cache_clear()  # el índice BM25 se reconstruye con las fichas nuevas
    return resumen, time.perf_counter() - inicio, db._collection.count()


def main():
    parser = argparse.ArgumentParser(description="Indexa los documentos del evento en ChromaDB.")
    parser.add_argument("archivos", nargs="*", help="Archivos a agregar. Sin argumentos: reindexa todo data/.")
    args = parser.parse_args()

    if args.archivos:
        rutas, desde_cero = args.archivos, False
    else:
        rutas = sorted(r for r in CARPETA_DATOS.iterdir() if r.suffix.lower() in EXTENSIONES)
        desde_cero = True

    if not rutas:
        print("No hay documentos en data/ (.md, .txt o .pdf).")
        return

    resumen, segundos, total = indexar(rutas, desde_cero)
    for archivo, n in resumen:
        print(f"  {archivo}: {n} fichas")
    print(f"\nListo: {sum(n for _, n in resumen)} fichas nuevas en {segundos:.1f} s. Total en la base: {total}")


if __name__ == "__main__":
    main()
