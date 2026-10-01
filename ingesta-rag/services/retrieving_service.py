"""
Servicio de recuperación de contexto RAG.

Expone una interfaz simple para backend/orquestación.
"""

from typing import List

from models.vector_db import buscar_en_chroma
from schemas.ingesta_schemas import ChunkResultado
from services.embedding_service import GeminiEmbeddingsResiliente
from utils.key_manager import KeyManager
from utils.config import GEMINI_API_KEYS


# Servicio compartido de embeddings
_key_manager = KeyManager(keys=GEMINI_API_KEYS)

_embedding_service = GeminiEmbeddingsResiliente(
    key_manager=_key_manager
)


def buscar_contexto(
    doc_id: str,
    query: str,
    top_k: int = 5
) -> List[ChunkResultado]:
    """
    Busca fragmentos relevantes dentro de un documento indexado.

    Parámetros:
    - doc_id: identificador del documento indexado.
    - query: pregunta o texto de búsqueda.
    - top_k: cantidad máxima de fragmentos.

    Retorna:
    - Lista de ChunkResultado.
    """

    if not doc_id:
        raise ValueError(
            "doc_id no puede estar vacío."
        )

    if not query:
        raise ValueError(
            "query no puede estar vacío."
        )


    resultados = buscar_en_chroma(
        doc_id=doc_id,
        query=query,
        embedding_function=_embedding_service,
        top_k=top_k
    )

    return resultados
