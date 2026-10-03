"""LLM-as-a-Judge: el dataset dorado como pruebas de regresión para la IA.

Fase 1: el generador responde las preguntas de golden_set.json (con RAG).
Fase 2: un modelo juez, de otra familia, califica cada respuesta del 1 al 5:
  - relevancia: ¿las fichas recuperadas traen lo necesario?
  - fidelidad:  ¿todo lo que dice la respuesta está respaldado por las fichas?
  - correccion: ¿coincide con la respuesta esperada (escrita por un humano)?
Una pregunta aprueba si sus notas son >= 4.

Hay dos versiones del juez (para mostrar en la charla que el juez también se prueba y se mejora):
  v1: la primera versión. Da falsos negativos en las preguntas trampa (castiga un "No lo sé" correcto).
  v2: la versión pulida (por defecto). En preguntas trampa la relevancia no aplica y
      "no está en los documentos" cuenta como respuesta correcta aunque cambie la redacción.

Uso:
    python evaluate.py                    # las 10 preguntas, juez v2
    python evaluate.py --version-juez 1   # reproduce los falsos negativos
    python evaluate.py --ids 7 8          # solo algunas
    python evaluate.py --juez gemma3:1b   # otro modelo juez
Guarda los resultados en resultados/ (JSON y Markdown).
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama
from rich.console import Console
from rich.table import Table

from rag import MODELO_LLM, armar_contexto, armar_prompt, buscar, llm

MODELO_JUEZ = "gemma3:4b"
METRICAS = ("relevancia", "fidelidad", "correccion")

PROMPT_JUEZ_V1 = """Eres un evaluador estricto de un sistema RAG (preguntas sobre un evento). Califica del 1 al 5.

PREGUNTA:
{pregunta}

FICHAS RECUPERADAS (lo único que el sistema podía usar):
{contexto}

RESPUESTA DEL SISTEMA:
{respuesta}

RESPUESTA ESPERADA (referencia escrita por un humano):
{esperado}

Criterios:
- relevancia: ¿las fichas recuperadas contienen la información necesaria para responder? 5 = sí, completa; 1 = nada útil. Si la respuesta esperada dice que el dato no está en los documentos, pon 5 cuando las fichas efectivamente no lo contengan.
- fidelidad: ¿cada afirmación de la respuesta está respaldada por las fichas? 5 = todo respaldado; 3 = agrega suposiciones; 1 = inventa datos. Decir "No lo sé" cuando el dato no está es fiel.
- correccion: ¿la respuesta coincide con la respuesta esperada? 5 = coincide; 3 = incompleta; 1 = la contradice.

Responde SOLO con un JSON con esta forma:
{{"relevancia": n, "fidelidad": n, "correccion": n, "motivo": "una frase en español"}}"""

# v2: un cambio de una línea en "correccion" + la relevancia no aplica en trampas (ver juzgar()).
PROMPT_JUEZ_V2 = PROMPT_JUEZ_V1.replace(
    "- correccion: ¿la respuesta coincide con la respuesta esperada? 5 = coincide; 3 = incompleta; 1 = la contradice.",
    "- correccion: ¿la respuesta coincide con la respuesta esperada? 5 = coincide; 3 = incompleta; 1 = la contradice. "
    "Si la respuesta del sistema y la esperada coinciden en que el dato no está en los documentos, pon 5 "
    "aunque la redacción sea distinta.",
)
PROMPTS_JUEZ = {1: PROMPT_JUEZ_V1, 2: PROMPT_JUEZ_V2}

consola = Console()


def generar(casos):
    consola.print(f"\n[bold]Fase 1[/] · el generador ({MODELO_LLM}) responde con RAG")
    for c in casos:
        inicio = time.perf_counter()
        resultados = buscar(c["pregunta"])
        c["respuesta"] = llm().invoke(armar_prompt(c["pregunta"], resultados)).content.strip()
        c["contexto"] = armar_contexto(resultados)
        c["fuentes"] = [d.metadata.get("fuente") for d, _ in resultados]
        c["seg_respuesta"] = round(time.perf_counter() - inicio, 1)
        consola.print(f"  [{c['id']:>2}] {c['seg_respuesta']:>5} s · {c['respuesta'][:70]}")


def juzgar(casos, modelo, version):
    consola.print(f"\n[bold]Fase 2[/] · el juez v{version} ({modelo}) califica")
    juez = ChatOllama(model=modelo, temperature=0, seed=42, format="json", num_ctx=4096, keep_alive="10m")
    for c in casos:
        inicio = time.perf_counter()
        salida = juez.invoke(PROMPTS_JUEZ[version].format(**c)).content
        try:
            nota = json.loads(salida)
        except json.JSONDecodeError:
            nota = {"motivo": f"El juez no devolvió JSON válido: {salida[:80]}"}
        for m in METRICAS:
            try:
                c[m] = max(0, min(5, int(nota.get(m, 0))))
            except (TypeError, ValueError):
                c[m] = 0
        c["motivo"] = str(nota.get("motivo", ""))
        if version >= 2 and c["tipo"] == "trampa":
            c["relevancia"] = None  # no hay ficha "correcta" que recuperar: la métrica no aplica
        c["aprobada"] = min(c[m] for m in METRICAS if c[m] is not None) >= 4
        c["seg_juez"] = round(time.perf_counter() - inicio, 1)
        consola.print(f"  [{c['id']:>2}] {c['seg_juez']:>5} s · {'✅' if c['aprobada'] else '❌'}")


def color(n):
    if n is None:
        return "[dim]—[/]"
    return f"[green]{n}[/]" if n >= 4 else f"[yellow]{n}[/]" if n == 3 else f"[red]{n}[/]"


def promedio(casos, m):
    """Promedio de una métrica; '—' si no aplica a ninguna pregunta (p. ej. solo trampas)."""
    notas = [c[m] for c in casos if c[m] is not None]
    return f"{sum(notas) / len(notas):.1f}" if notas else "—"


def mostrar(casos, modelo, version):
    tabla = Table(title=f"Evaluación LLM-as-a-Judge · generador {MODELO_LLM} · juez v{version} {modelo}",
                  title_style="bold", header_style="bold cyan", show_lines=False)
    tabla.add_column("#", justify="right")
    tabla.add_column("Tipo")
    tabla.add_column("Pregunta", max_width=48, no_wrap=True)
    for m in ("Relev.", "Fidel.", "Correc."):
        tabla.add_column(m, justify="center")
    tabla.add_column("", justify="center")
    for c in casos:
        tabla.add_row(str(c["id"]), c["tipo"], c["pregunta"],
                      *(color(c[m]) for m in METRICAS), "✅" if c["aprobada"] else "❌")
    consola.print()
    consola.print(tabla)

    aprobadas = sum(c["aprobada"] for c in casos)
    promedios = " · ".join(f"{m} {promedio(casos, m)}" for m in METRICAS)
    consola.print(f"\n[bold]Aprobadas: {aprobadas}/{len(casos)}[/]  ·  Promedios: {promedios}\n")

    for c in (c for c in casos if not c["aprobada"]):
        consola.print(f"[red]❌ [{c['id']}][/] {c['pregunta']}")
        consola.print(f"   Respuesta: {c['respuesta']}")
        consola.print(f"   Esperado:  {c['esperado']}")
        consola.print(f"   Juez:      {c['motivo']}\n")


def guardar(casos, modelo, version):
    carpeta = Path("resultados")
    carpeta.mkdir(exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M") + f"_juez_v{version}"
    (carpeta / f"evaluacion_{marca}.json").write_text(
        json.dumps({"generador": MODELO_LLM, "juez": modelo, "version_juez": version, "casos": casos}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    filas = ["| # | Tipo | Pregunta | Relev. | Fidel. | Correc. | Aprobada | Motivo del juez |",
             "|---|---|---|---|---|---|---|---|"]
    filas += [f"| {c['id']} | {c['tipo']} | {c['pregunta']} | {'—' if c['relevancia'] is None else c['relevancia']} | {c['fidelidad']} | "
              f"{c['correccion']} | {'✅' if c['aprobada'] else '❌'} | {c['motivo']} |" for c in casos]
    (carpeta / f"evaluacion_{marca}.md").write_text("\n".join(filas), encoding="utf-8")
    consola.print(f"Resultados guardados en resultados/evaluacion_{marca}.json y .md")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el RAG con el dataset dorado (LLM-as-a-Judge).")
    parser.add_argument("--ids", nargs="*", type=int, help="ids de golden_set.json a evaluar")
    parser.add_argument("--juez", default=MODELO_JUEZ, help=f"modelo juez (por defecto {MODELO_JUEZ})")
    parser.add_argument("--version-juez", type=int, choices=[1, 2], default=2, help="versión del juez (1 o 2)")
    args = parser.parse_args()

    casos = json.loads(Path("golden_set.json").read_text(encoding="utf-8"))
    if args.ids:
        casos = [c for c in casos if c["id"] in args.ids]

    inicio = time.perf_counter()
    generar(casos)  # primero todas las respuestas: un solo modelo en la GPU a la vez
    juzgar(casos, args.juez, args.version_juez)
    mostrar(casos, args.juez, args.version_juez)
    guardar(casos, args.juez, args.version_juez)
    consola.print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
