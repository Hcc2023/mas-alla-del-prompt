"""Demo en el escenario: Asistente del AWS Student Community Day 2026.

Uso:  streamlit run app.py
Respuesta a la izquierda, fichas (fuentes) a la derecha, preguntas precargadas en botones.
"""
import time
from pathlib import Path

import streamlit as st

from ingest import CARPETA_DATOS, EXTENSIONES, indexar
from rag import (MODELO_EMBEDDINGS, MODELO_LLM, base_vectorial, describir, embeddings, llm,
                 responder_sin_rag, responder_stream)

# Preguntas precargadas: en el escenario no se escribe en vivo.
PREGUNTAS = {
    # Con formato exigido: sin RAG inventa un nombre y un título; con RAG responde Gonzalo + título real.
    "🎯 ¿Quién sigue?": ("¿Quién habla después del almuerzo en el Main Stage? "
                        "Solo respóndeme con el nombre del speaker y el título de su charla"),
    "🍽️ ¿Hay comida?": "¿Dan comida en el evento?",
    "📜 Certificados": "¿A quiénes dan certificados y en qué idioma son las sesiones?",
    "🪤 Wifi": "¿Cuál es la contraseña del wifi del evento?",
}

st.set_page_config(page_title="Asistente del evento", page_icon="🤖", layout="wide")
st.markdown("<style>html { font-size: 20px; }</style>", unsafe_allow_html=True)  # legible en proyector


@st.cache_resource(show_spinner="Precargando modelos en la GPU…")
def precargar():
    embeddings().embed_query("hola")
    llm().invoke("Responde solo: listo")
    return True


precargar()

if "pregunta" not in st.session_state:
    st.session_state.pregunta = ""
    st.session_state.ejecutar = False


def elegir(texto):
    st.session_state.pregunta = texto
    st.session_state.ejecutar = True


# --- Barra lateral: agregar documentos en vivo (el remate) ---
with st.sidebar:
    st.header("📚 Documentos")
    st.metric("Fichas en la base", base_vectorial()._collection.count())
    subido = st.file_uploader("Agregar documento", type=[e.lstrip(".") for e in EXTENSIONES])
    if subido and st.button("Indexar ahora", type="primary"):
        ruta = CARPETA_DATOS / subido.name
        ruta.write_bytes(subido.getvalue())
        with st.spinner(f"Indexando {subido.name}…"):
            resumen, segundos, total = indexar([ruta])
        st.success(f"{resumen[0][1]} fichas nuevas en {segundos:.1f} s. Total: {total}")

# --- Encabezado ---
st.title("🤖 Asistente del AWS Student Community Day 2026")
st.caption(f"100% local, sin internet · {MODELO_LLM} · {MODELO_EMBEDDINGS} · ChromaDB")

modo = st.radio("Modo", ["Con RAG", "Sin RAG (el modelo solo)"], horizontal=True)

columnas = st.columns(len(PREGUNTAS))
for col, (etiqueta, texto) in zip(columnas, PREGUNTAS.items()):
    col.button(etiqueta, on_click=elegir, args=(texto,), width="stretch")

st.text_input("Pregunta", key="pregunta")
if st.button("Preguntar", type="primary"):
    st.session_state.ejecutar = True

# --- Respuesta ---
pregunta = st.session_state.pregunta.strip()
if st.session_state.ejecutar and pregunta:
    st.session_state.ejecutar = False
    izquierda, derecha = st.columns([3, 2], gap="large")
    inicio = time.perf_counter()

    if modo.startswith("Sin RAG"):
        with izquierda:
            st.subheader("Respuesta")
            with st.spinner("Pensando…"):
                st.write(responder_sin_rag(pregunta))
            st.caption(f"{time.perf_counter() - inicio:.1f} s")
        with derecha:
            st.subheader("Fuentes")
            st.warning("Ninguna: el modelo responde de memoria.")
    else:
        generador, resultados = responder_stream(pregunta)
        with derecha:
            st.subheader("Fichas que trajo el bibliotecario")
            for i, (doc, distancia) in enumerate(resultados, 1):
                with st.expander(f"[{i}] {doc.metadata.get('fuente')} · {describir(distancia)}"):
                    st.text(doc.page_content)
        with izquierda:
            st.subheader("Respuesta")
            st.write_stream(generador)
            st.caption(f"{time.perf_counter() - inicio:.1f} s")
