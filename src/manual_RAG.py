from typing import Any, Dict
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection


def semantic_search(query: str,
                    model: SentenceTransformer,
                    collection: Collection,
                    n_results: int = 5) -> Dict[str, Any]:
    """
    Search for semantically similar text chunks using vector embeddings.

    This function encodes the input query using a sentence transformer model,
    performs a similarity search in a ChromaDB collection, and returns the most
    relevant chunks along with their metadata and similarity scores.

    Args:
        query (str): The search query text to find similar chunks for.
        model (SentenceTransformer): A sentence-transformers model instance used
            to encode the query into vector embeddings.
        collection (Collection): A ChromaDB collection object containing the
            indexed document chunks with their embeddings.
        n_results (int, optional): Maximum number of similar results to return.
            Defaults to 5.

    Returns:
        Dict[str, Any]: A dictionary containing search results with the following keys:
            - documents (List[List[str]]): Retrieved document texts (nested list).
            - embeddings (List[List[float]]): Vector embeddings of retrieved documents.
            - metadatas (List[List[Dict]]): Metadata associated with each document.
            - distances (List[List[float]]): Similarity distances (lower = more similar).

    Note:
        - The query is automatically converted to a list format required by ChromaDB.
        - Distance values depend on the embedding model's similarity metric.
        - Results are ordered by similarity (most similar first).
    """
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'embeddings', 'metadatas', 'distances']
    )

    return results


def print_search_results(results: Dict[str, Any], query: str) -> None:
    """
    Display search results in a formatted, human-readable format.

    This function prints semantic search results with metadata including similarity
    scores, page numbers, chunk IDs, and text previews. Results are displayed in
    ranked order with clear visual separation.

    Args:
        results (Dict[str, Any]): Search results dictionary from ChromaDB query
            containing the following keys:
            - ids (List[List[str]]): Document IDs for each result.
            - documents (List[List[str]]): Full text content of matched documents.
            - metadatas (List[List[Dict]]): Metadata dictionaries with 'page_num' key.
            - distances (List[List[float]]): Distance values (lower = more similar).
        query (str): The original search query text to display.

    Returns:
        None: This function prints directly to stdout and returns nothing.

    Note:
        - Similarity is calculated as (1 - distance) for intuitive interpretation.
        - Text previews are truncated to the first 300 characters.
        - The function expects ChromaDB result format with nested lists.
        - A horizontal line of 100 dashes separates the query from results.
    """
    print('Query:', query)
    print('-' * 100)

    for idx, (doc_id, doc, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 - distance

        print(f'Rank {idx} | Similarity: {similarity:.3f} | Page: {metadata["page_num"]}', end='')
        print(f' | Chunk ID: {doc_id} | Text preview below (first 300 chars):')
        print(f'{doc[:300]}...', end='\n\n')


def create_prompt(query: str, search_results: Dict[str, Any]) -> str:
    """
    Create a structured prompt for Retrieval-Augmented Generation (RAG) using retrieved context.

    This function constructs a comprehensive prompt that includes context chunks from
    semantic search results along with specific instructions for the language model.
    The prompt is designed for answering questions about Amazon Web Services documentation
    while maintaining citation accuracy and avoiding hallucinations.

    Args:
        query (str): The user's question or query to be answered.
        search_results (Dict[str, Any]): Search results dictionary from ChromaDB query
            containing the following keys:
            - documents (List[List[str]]): Retrieved document texts.
            - metadatas (List[List[Dict]]): Metadata dictionaries containing 'page_num'.

    Returns:
        str: A formatted prompt string ready for language model consumption, containing:
            - System instructions for answering behavior
            - All context chunks with page numbers
            - The user's question
            - Request for step-by-step reasoning

    Note:
        - Context chunks are numbered sequentially starting from 1.
        - Each chunk includes its source page number for citation.
        - The prompt explicitly instructs the model to cite sources and avoid speculation.
        - Instructions are tailored for AWS documentation but can be adapted.
    """
    context = []
    for idx, (doc, metadata) in enumerate(zip(
        search_results['documents'][0],
        search_results['metadatas'][0],
    ), 1):
        context_piece = f"""
[Context chunk {idx} - Page {metadata['page_num']}]
{doc}
"""
        context.append(context_piece)

    context_block = "\n".join(context)

    prompt = f"""You are an expert at answering questions about Amazon Web Services documentation.

INSTRUCTIONS:
1. Read all context chunks from documentation carefully
2. Identify which chunks contain relevant information
3. Synthesize a clear answer using ONLY the provided context
4. Do NOT use your general knowledge and do not make assumptions
5. Cite page numbers for each piece of information
6. Explicitly state if the answer is not in the provided context
7. Write in PLAIN TEXT without any formatting (no bold, no italics, no markdown syntax like ** or __)
8. You may use line breaks and simple numbering/bullet points for clarity

CONTEXT CHUNKS FROM DOCUMENTATION:
{context_block}

USER QUESTION:
{query}

Think step-by-step, then provide your final ANSWER only without steps.

ANSWER:"""

    return prompt