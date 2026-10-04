"""Núcleo del RAG. Lo usan preguntar.py (terminal), app.py (Streamlit) y evaluate.py (juez)."""
import re
import unicodedata
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rank_bm25 import BM25Okapi

# --- Configuración (decidida tras las pruebas de GPU del 3-oct) ---
MODELO_LLM = "qwen3:4b-instruct"     # generador: ~75% GPU, 20-25 tok/s
MODELO_EMBEDDINGS = "embeddinggemma"  # multilingüe, 622 MB (respaldo: bge-m3)
CARPETA_DB = "chroma_db"
COLECCION = "evento"
K = 5             # cuántas fichas trae el bibliotecario
NUM_CTX = 2048    # contexto corto = más capas en la GPU

PROMPT_RAG = """Eres el asistente del AWS Student Community Day Cochabamba 2026.
Responde SOLO con la información del CONTEXTO.
Si la respuesta no está en el CONTEXTO, responde exactamente: "No lo sé: eso no está en los documentos del evento."
Responde en español, en 1 a 3 frases, e indica entre corchetes el número de la ficha que usaste, por ejemplo [1].
Si la pregunta es sobre una sesión o un speaker, incluye SIEMPRE el nombre del speaker, el título completo de la sesión entre comillas y su horario.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}
RESPUESTA:"""


class EmbeddingsGemma(OllamaEmbeddings):
    """EmbeddingGemma rinde mejor con las instrucciones de tarea que recomienda Google."""

    def embed_query(self, text):
        return super().embed_query(f"task: search result | query: {text}")

    def embed_documents(self, texts):
        return super().embed_documents([f"title: none | text: {t}" for t in texts])


@lru_cache
def embeddings():
    if MODELO_EMBEDDINGS.startswith("embeddinggemma"):
        return EmbeddingsGemma(model=MODELO_EMBEDDINGS)
    return OllamaEmbeddings(model=MODELO_EMBEDDINGS)


@lru_cache
def base_vectorial():
    return Chroma(
        collection_name=COLECCION,
        embedding_function=embeddings(),
        persist_directory=CARPETA_DB,
    )


@lru_cache
def llm():
    # temperatura 0 + semilla fija = la misma respuesta en el ensayo y en la charla
    return ChatOllama(model=MODELO_LLM, temperature=0, seed=42,
                      num_ctx=NUM_CTX, keep_alive="60m")


# --- Búsqueda por palabras (BM25): encuentra nombres y títulos exactos ---
VACIAS = set("de la el los las en y a que un una es del al por con se su para como o lo le "
             "me mi te tu yo estoy esta este".split())


def palabras(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))  # sin tildes
    return [p for p in re.findall(r"\w+", texto) if p not in VACIAS]


@lru_cache(maxsize=1)
def _indice_palabras(total_fichas):
    datos = base_vectorial().get(include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m) for t, m in zip(datos["documents"], datos["metadatas"])]
    return docs, BM25Okapi([palabras(d.page_content) for d in docs]) if docs else None


def indice_palabras():
    """Índice BM25 sobre las mismas fichas de ChromaDB.
    Se reconstruye solo cuando cambia la cantidad de fichas (p. ej. al agregar o quitar los apuntes)."""
    return _indice_palabras(base_vectorial()._collection.count())


indice_palabras.cache_clear = _indice_palabras.cache_clear  # ingest.py lo limpia al indexar


def buscar(pregunta, k=K):
    """El bibliotecario con dos estrategias combinadas (búsqueda híbrida):
    - por significado (embeddings en ChromaDB)
    - por palabras exactas (BM25)
    Se fusionan por posición (Reciprocal Rank Fusion).
    Devuelve [(documento, distancia)]; distancia=None si la ficha llegó solo por palabras."""
    fusion = {}

    for pos, (doc, dist) in enumerate(base_vectorial().similarity_search_with_score(pregunta, k=k * 2)):
        fusion[doc.page_content] = {"doc": doc, "dist": dist, "puntaje": 1 / (60 + pos)}

    docs, bm25 = indice_palabras()
    if bm25:
        puntajes = bm25.get_scores(palabras(pregunta))
        orden = sorted(range(len(docs)), key=lambda i: -puntajes[i])[: k * 2]
        for pos, i in enumerate(orden):
            if puntajes[i] <= 0:
                break
            entrada = fusion.setdefault(docs[i].page_content, {"doc": docs[i], "dist": None, "puntaje": 0})
            entrada["puntaje"] += 1 / (60 + pos)

    mejores = sorted(fusion.values(), key=lambda e: -e["puntaje"])[:k]
    return [(e["doc"], e["dist"]) for e in mejores]


def armar_contexto(resultados):
    """La chuleta permitida: las fichas numeradas, con su fuente."""
    return "\n\n".join(
        f"[{i}] (fuente: {doc.metadata.get('fuente', '?')})\n{doc.page_content}"
        for i, (doc, _) in enumerate(resultados, 1)
    )


def armar_prompt(pregunta, resultados):
    return PROMPT_RAG.format(contexto=armar_contexto(resultados), pregunta=pregunta)


def responder(pregunta):
    """Con RAG. Devuelve (texto, resultados)."""
    resultados = buscar(pregunta)
    return llm().invoke(armar_prompt(pregunta, resultados)).content, resultados


def responder_stream(pregunta):
    """Con RAG, palabra por palabra (para Streamlit). Devuelve (generador, resultados)."""
    resultados = buscar(pregunta)
    generador = (trozo.content for trozo in llm().stream(armar_prompt(pregunta, resultados)))
    return generador, resultados


def responder_sin_rag(pregunta):
    """El modelo solo, sin apuntes: aquí es donde inventa.
    Misma regla de largo que con RAG, para que la comparación sea justa (y breve en el escenario)."""
    return llm().invoke(f"Responde en español, en 1 a 3 frases.\n\n{pregunta}").content


def describir(distancia):
    """Texto para mostrar cómo llegó cada ficha."""
    return "por palabras" if distancia is None else f"distancia {distancia:.2f}"
