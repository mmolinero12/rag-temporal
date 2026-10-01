"""
Simulación de backend consumiendo ingesta-rag.

Este archivo representa un servicio externo.
No conoce:
- ChromaDB
- Gemini
- embeddings
- KeyManager

Solo consume la interfaz pública.
"""

import sys
from pathlib import Path


# Simula que backend importa el módulo instalado
RUTA_INGESTA = (
    Path(__file__)
    .resolve()
    .parent.parent
    / "ingesta-rag"
)

sys.path.append(str(RUTA_INGESTA))


from services.retrieving_service import buscar_contexto


def main():

    # Documento obtenido previamente por indexación
    doc_id = "4f08ccb6-84ce-4056-a134-1ffdf3ce52c6"


    pregunta_usuario = (
        "¿Qué competencias digitales debe desarrollar un docente?"
    )


    print("=== BACKEND SIMULADO ===")
    print("Consulta enviada al módulo RAG:")
    print(pregunta_usuario)


    contexto = buscar_contexto(
        doc_id=doc_id,
        query=pregunta_usuario,
        top_k=3
    )


    print("\nContexto recibido:")
    

    for i, chunk in enumerate(contexto, 1):

        print("\n----------------")
        print(f"Resultado {i}")
        print("Score:", chunk.score)
        print("Fuente:", chunk.fuente)
        print(chunk.texto[:300])


if __name__ == "__main__":
    main()
