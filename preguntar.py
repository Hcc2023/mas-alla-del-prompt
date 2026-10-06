"""Prueba en terminal (y respaldo de la demo): la misma pregunta SIN y CON RAG.

Uso:
    python preguntar.py "¿Quién habla después del almuerzo en el Main Stage?"
    python preguntar.py --solo-rag "..."
"""
import argparse
import time

from rag import describir, responder, responder_sin_rag

parser = argparse.ArgumentParser()
parser.add_argument("pregunta", nargs="+")
parser.add_argument("--solo-rag", action="store_true", help="omitir la respuesta sin RAG")
args = parser.parse_args()
pregunta = " ".join(args.pregunta)

if not args.solo_rag:
    t = time.perf_counter()
    print("\n=== SIN RAG (el modelo solo) ===")
    print(responder_sin_rag(pregunta))
    print(f"({time.perf_counter() - t:.1f} s)")

t = time.perf_counter()
print("\n=== CON RAG ===")
respuesta, resultados = responder(pregunta)
print(respuesta)
print(f"({time.perf_counter() - t:.1f} s)")

print("\nFichas que trajo el bibliotecario (menor distancia = más parecida; 'por palabras' = coincidencia exacta):")
for i, (doc, distancia) in enumerate(resultados, 1):
    vista = doc.page_content[:90].replace("\n", " ")
    print(f"  [{i}] {doc.metadata.get('fuente')}  {describir(distancia)}  «{vista}…»")
