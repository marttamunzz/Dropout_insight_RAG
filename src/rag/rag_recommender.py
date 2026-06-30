import json
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rag.llm_providers import LLM_MODELS, generate_with_llm


BASE_DIR = Path(__file__).resolve().parents[2]

DB_DIR = str(BASE_DIR / "rag_db")
RUNS_DIR = BASE_DIR / "rag_runs"

COLLECTION_NAME = "dropout_literature"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_MODEL_KEY = "gemini"

load_dotenv(BASE_DIR / ".env")


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"No se ha encontrado la coleccion '{COLLECTION_NAME}' en {DB_DIR}. "
            "Ejecuta primero src/rag/build_rag_index.py para construir el indice RAG."
        ) from exc


def retrieve_literature(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    embedding_model = get_embedding_model()
    collection = get_chroma_collection()

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_chunks = []

    documents_result = results.get("documents") or [[]]
    metadatas_result = results.get("metadatas") or [[]]
    distances_result = results.get("distances") or [[]]

    documents = documents_result[0]
    metadatas = metadatas_result[0]
    distances = distances_result[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}

        retrieved_chunks.append({
            "text": document,
            "source": metadata.get("source", "unknown"),
            "chunk_index": metadata.get("chunk_index", "unknown"),
            "page": metadata.get("page", "unknown"),
            "distance": float(distance)
        })

    return retrieved_chunks


def format_retrieved_evidence(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No se ha recuperado evidencia documental relevante."

    evidence_blocks = []

    for i, chunk in enumerate(chunks, start=1):
        evidence_blocks.append(
            f"""
EVIDENCIA {i}
Fuente: {chunk.get("source", "unknown")}
Pagina: {chunk.get("page", "unknown")}
Chunk: {chunk.get("chunk_index", "unknown")}
Distancia: {chunk.get("distance", "unknown")}

Fragmento:
{chunk.get("text", "")}
"""
        )

    return "\n".join(evidence_blocks)


def build_search_query(group_description: str, counterfactuals_text: str) -> str:
    return f"""
abandono academico
riesgo de abandono
retencion estudiantil
intervencion educativa
sistemas de alerta temprana
tutorizacion personalizada
rendimiento academico
participacion del estudiante
explicaciones contrafactuales en analitica del aprendizaje
recomendaciones prescriptivas educativas

Descripcion del grupo:
{group_description}

Contrafactuales grupales:
{counterfactuals_text}
""".strip()


def build_final_prompt(
    group_description: str,
    counterfactuals_text: str,
    evidence_text: str
) -> str:
    return f"""
Eres un asistente experto en analitica educativa, abandono academico,
explicabilidad de modelos de inteligencia artificial y recomendaciones prescriptivas.

Tu tarea es generar recomendaciones educativas prescriptivas para un grupo de estudiantes
en riesgo de abandono academico.

Debes utilizar exclusivamente:
1. La descripcion del grupo.
2. Los contrafactuales grupales proporcionados.
3. La evidencia cientifica recuperada mediante RAG.

Reglas obligatorias:
- No generes nuevos contrafactuales.
- No inventes nuevos valores numericos.
- No inventes variables que no aparezcan en la descripcion o en los contrafactuales.
- No contradigas los cambios sugeridos por los contrafactuales.
- No hagas una tabla.
- No te limites a describir el grupo.
- Transforma los cambios sugeridos por los contrafactuales en intervenciones educativas concretas.
- Si la evidencia recuperada no es suficiente para justificar alguna recomendacion, indicalo de forma prudente.
- Redacta la respuesta para docentes, tutores o responsables academicos no tecnicos.

DESCRIPCION DEL GRUPO:
{group_description}

CONTRAFACTUALES GRUPALES:
{counterfactuals_text}

EVIDENCIA CIENTIFICA RECUPERADA MEDIANTE RAG:
{evidence_text}

Genera la respuesta en espanol con esta estructura obligatoria:

1. Interpretacion del grupo.
2. Factores principales de riesgo.
3. Recomendaciones prescriptivas.
4. Justificacion basada en los contrafactuales y en la evidencia recuperada.
5. Limitaciones de la recomendacion.
6. Fuentes documentales utilizadas, indicando el nombre de los archivos PDF.

Instrucciones de formato:
- Debes completar obligatoriamente las seis secciones.
- No dejes secciones vacias.
- Cada seccion debe tener al menos dos frases, excepto la seccion de fuentes.
- Las recomendaciones deben ser concretas, accionables y orientadas a docentes o responsables academicos.
- No escribas frases introductorias como "Aqui tienes" o "A continuacion".
""".strip()


def call_llm_with_retries(
    prompt: str,
    model_key: str = DEFAULT_MODEL_KEY,
    retries: int = 3
) -> str:
    last_error = None

    for attempt in range(retries):
        try:
            response_text = generate_with_llm(
                prompt=prompt,
                model_key=model_key
            )

            if response_text:
                return response_text.strip()

            return f"No se ha recibido texto generado por el modelo '{model_key}'."

        except Exception as exc:
            last_error = exc

            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                break

    raise RuntimeError(
        f"Error generando recomendacion con el modelo '{model_key}': {last_error}"
    )


def get_model_metadata(model_key: str) -> dict[str, Any]:
    if model_key not in LLM_MODELS:
        return {
            "model_key": model_key,
            "provider": "unknown",
            "label": "unknown",
            "model_id": "unknown"
        }

    config = LLM_MODELS[model_key]

    return {
        "model_key": model_key,
        "provider": config.get("provider", "unknown"),
        "label": config.get("label", "unknown"),
        "model_id": config.get("model", "unknown")
    }


def save_rag_run(result: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_key = result.get("model_key", "unknown_model")
    output_path = RUNS_DIR / f"rag_run_{model_key}_{timestamp}.json"

    data_to_save = {
        "timestamp": timestamp,
        "model_key": result.get("model_key"),
        "provider": result.get("provider"),
        "model_label": result.get("model_label"),
        "model_id": result.get("model_id"),
        "response_time_seconds": result.get("response_time_seconds"),
        "search_query": result.get("search_query"),
        "group_description": result.get("group_description"),
        "counterfactuals_text": result.get("counterfactuals_text"),
        "retrieved_chunks": result.get("retrieved_chunks"),
        "recommendation": result.get("recommendation"),
        "manual_evaluation": {
            "clarity": None,
            "coherence_with_counterfactuals": None,
            "relation_with_literature": None,
            "practical_utility": None,
            "prescriptive_value": None,
            "hallucination_control": None,
            "interface_usability": None,
            "comments": ""
        }
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data_to_save, file, ensure_ascii=False, indent=2)

    return output_path


def generate_rag_recommendations(
    group_description: str,
    counterfactuals_text: str,
    top_k: int = 5,
    model_key: str = DEFAULT_MODEL_KEY,
    return_metadata: bool = False,
    save_run: bool = False
) -> str | dict[str, Any]:
    if model_key not in LLM_MODELS:
        available_models = ", ".join(LLM_MODELS.keys())
        raise ValueError(
            f"Modelo no reconocido: '{model_key}'. "
            f"Modelos disponibles: {available_models}"
        )

    model_metadata = get_model_metadata(model_key)

    search_query = build_search_query(
        group_description=group_description,
        counterfactuals_text=counterfactuals_text
    )

    retrieved_chunks = retrieve_literature(
        query=search_query,
        top_k=top_k
    )

    evidence_text = format_retrieved_evidence(retrieved_chunks)

    final_prompt = build_final_prompt(
        group_description=group_description,
        counterfactuals_text=counterfactuals_text,
        evidence_text=evidence_text
    )

    start_time = time.time()

    recommendation = call_llm_with_retries(
        prompt=final_prompt,
        model_key=model_key
    )

    response_time_seconds = round(time.time() - start_time, 2)

    result = {
        "model_key": model_metadata["model_key"],
        "provider": model_metadata["provider"],
        "model_label": model_metadata["label"],
        "model_id": model_metadata["model_id"],
        "response_time_seconds": response_time_seconds,
        "group_description": group_description,
        "counterfactuals_text": counterfactuals_text,
        "search_query": search_query,
        "retrieved_chunks": retrieved_chunks,
        "evidence_text": evidence_text,
        "prompt": final_prompt,
        "recommendation": recommendation,
        "saved_run_path": None
    }

    if save_run:
        result["saved_run_path"] = str(save_rag_run(result))

    if return_metadata:
        return result

    return recommendation