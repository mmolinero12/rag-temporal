"""
Módulo: services/ingestion.py
Descripción: Interfaz principal del módulo de ingesta (RAG).
Expone las funciones requeridas por el contrato para comunicarse con
'orquestacion-agentes' y el backend, orquestando los servicios internos.
"""

import logging
from typing import List

# Importamos las estructuras de datos definidas en el contrato unificado
# (Asumiendo que el archivo de contratos se llama contratos.py)
from schemas.ingesta_schemas import ChunkResultado

# Importamos los servicios internos que ya construimos
from services.indexing_service import indexar_documento as _indexar_impl
from services.embedding_service import GeminiEmbeddingsResiliente
from models.vector_db import buscar_en_chroma  # Asumimos esta función en tu capa de BD
from utils.key_manager import KeyManager
from utils.config import GEMINI_API_KEYS

logger = logging.getLogger(__name__)

# Instancia global del servicio de embeddings para las búsquedas (singleton para este módulo)
_key_manager = KeyManager(keys=GEMINI_API_KEYS)
_embedding_service = GeminiEmbeddingsResiliente(key_manager=_key_manager)


def indexar_documento(documento_titulo: str, documento_contenido: str) -> str:
    """
    Cumple con el Contrato 1: Chunkea, genera embeddings, e indexa en el vector store.
    
    Delega la lógica pesada a indexing_service.py para mantener el monolito modular.
    """
    logger.info(f"[INGESTION] Recibiendo petición de backend para indexar: '{documento_titulo}'")
    
    # Delegamos al servicio de indexación previamente definido
    doc_id = _indexar_impl(
        documento_titulo=documento_titulo, 
        documento_contenido=documento_contenido
    )
    
    return doc_id


def buscar_contexto(doc_id: str, query: str, top_k: int = 5) -> List[ChunkResultado]:
    """
    Cumple con el Contrato 1: Busca los chunks más relevantes dentro de un documento.
    
    Parámetros:
    - doc_id: ID devuelto previamente por indexar_documento.
    - query: La pregunta o consulta del usuario.
    - top_k: Cantidad de fragmentos a recuperar.
    
    Retorna:
    - Lista de objetos ChunkResultado validados por Pydantic.
    """
    logger.info(f"[INGESTION] Buscando contexto | Doc: {doc_id} | Query: '{query}' | Top-K: {top_k}")
    
    # 1. Convertir la consulta a vector usando el servicio resiliente
    try:
        query_embedding = _embedding_service.embed_query(query)
    except Exception as e:
        logger.error(f"[INGESTION] Fallo al generar embedding para la query: {e}")
        raise RuntimeError("No se pudo generar el embedding para la búsqueda.") from e

    # 2. Búsqueda en ChromaDB (se asume que la DB filtra por doc_id)
    # Resultados crudos esperados: lista de dicts con 'texto', 'score', y 'metadatos'
    resultados_crudos = buscar_en_chroma(
        doc_id=doc_id,
        query_embedding=query_embedding,
        n_results=top_k
    )

    if not resultados_crudos:
        logger.warning(f"[INGESTION] No se encontró contexto para el doc_id: {doc_id}")
        return []

    # 3. Mapeo estricto contra el contrato de Pydantic (ChunkResultado)
    resultados_validados = []
    for res in resultados_crudos:
        # Extracción segura de metadatos (ej. número de página, sección original)
        metadatos = res.get("metadatos", {})
        fuente_origen = metadatos.get("fuente", f"doc_id:{doc_id}")
        
        chunk = ChunkResultado(
            texto=res.get("texto", ""),
            score=float(res.get("score", 0.0)),
            fuente=fuente_origen
        )
        resultados_validados.append(chunk)

    logger.info(f"[INGESTION] Retornando {len(resultados_validados)} chunks al orquestador.")
    return resultados_validados