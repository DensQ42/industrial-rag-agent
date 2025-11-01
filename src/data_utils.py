from pathlib import Path
import requests
from tqdm.auto import tqdm
import pandas as pd
import pymupdf
from typing import Union
from IPython.display import display
import re
from typing import Any, Dict
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection


def download_file(url: str, name: str, mode: str = 'wb', overwrite: bool = False) -> None:
    """
    Downloads a file from a URL and saves it to the raw data directory.

    This utility function handles file downloads with automatic directory creation,
    duplicate checking, and error handling. It is designed for downloading datasets
    and other resources needed for data analysis pipelines, ensuring that files
    are properly saved in the designated raw data folder structure.

    The function includes overwrite protection to prevent accidental re-downloads
    of existing files, and provides clear status messages about download progress
    and file locations.

    Args:
        url (str): The complete URL of the file to download. Should be a direct
            download link that responds to HTTP GET requests.
        name (str): The filename to use when saving the downloaded file. Should
            include the appropriate file extension (e.g., 'data.csv', 'model.pkl').
        mode (str, optional): File writing mode for saving the downloaded content.
            Use 'wb' for binary files (default) or 'w' for text files. Defaults to 'wb'.
        overwrite (bool, optional): Whether to overwrite existing files with the
            same name. If False, skips download when file already exists. If True,
            always downloads and replaces existing file. Defaults to False.

    Returns:
        None: This function performs file operations but does not return any value.

    Raises:
        requests.HTTPError: If the HTTP request fails (non-200 status code).
        IOError: If there are issues writing the file to disk.
        requests.RequestException: For network-related errors during download.

    Note:
        - Creates '../data/raw/' directory structure if it doesn't exist
        - Uses pathlib.Path for cross-platform file path handling
        - Prints status messages for user feedback on download progress
        - Binary mode ('wb') is recommended for most file types to preserve integrity
        - File existence check prevents unnecessary re-downloads by default
        - Response content is written in full after complete download
    """
    folder = Path('../data/raw/')
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / name

    if file_path.exists() and not overwrite:
        print(f'File {file_path.name} already exists')
    else:
        response = requests.get(url=url)
        response.raise_for_status()
        with open(file=file_path, mode=mode) as f:
            f.write(response.content)
        print(f'File has been downloaded to {file_path}')


def _extract_text_from_pdf(doc:pymupdf.Document) -> pd.DataFrame:
    """
    Extract text content from a PDF document and return structured data.

    This function iterates through all pages of a PDF document, extracts the text
    content from each page, and compiles the results into a pandas DataFrame with
    metadata including page numbers and character counts.

    Args:
        doc: A PyMuPDF (fitz) document object representing an opened PDF file.
            This should be a document instance created using pymupdf.open().

    Returns:
        pd.DataFrame: A DataFrame containing the extracted text data with the
            following columns:
            - page_num (int): Page number (1-indexed) in the document.
            - text (str): Extracted text content from the page.
            - char_count (int): Number of characters in the extracted text.

    Example:
        >>> import pymupdf
        >>> import pandas as pd
        >>> doc = pymupdf.open('document.pdf')
        >>> df = extract_text_from_pdf(doc)
        >>> print(df.head())
           page_num                    text  char_count
        0         1  First page content...         150
        1         2  Second page content...        200

    Note:
        - The function displays a progress bar using tqdm during extraction.
        - Empty pages will have empty strings with char_count of 0.
        - Page numbers in the output are 1-indexed for user readability.
    """
    pages_data = []

    for page_num in tqdm(range(len(doc)), desc='Extracting pages'):
        page = doc[page_num]
        text = page.get_text()

        pages_data.append({
            'page_num': page_num + 1,
            'text': text,
            'char_count': len(text)
        })

    df = pd.DataFrame(pages_data)

    return df


def load_and_analyze_pdf(path: Union[str, Path]) -> pd.DataFrame:
    """
    Open a PDF document, extract its text, and display comprehensive metadata.

    This function opens a PDF file, extracts text from all pages into a DataFrame,
    and prints detailed information about the document including metadata, file size,
    page count, and character distribution statistics. The PDF document is
    automatically closed after processing.

    Args:
        path (Union[str, Path]): Path to the PDF file. Can be either a string path
            or a pathlib.Path object.

    Returns:
        pd.DataFrame: DataFrame with extracted text data containing columns:
            - page_num (int): Page number (1-indexed).
            - text (str): Extracted text content from the page.
            - char_count (int): Number of characters in the extracted text.

    Note:
        - The function prints metadata and statistics directly to stdout.
        - Character distribution statistics are displayed using IPython's display().
        - The PDF document is automatically closed after extraction.
    """
    doc = pymupdf.open(path)

    df = _extract_text_from_pdf(doc)

    print(f'Filename: {Path(path).name}')
    print(f"Title: {doc.metadata.get('title', 'N/A')}")
    print(f"Author: {doc.metadata.get('author', 'N/A')}")
    print(f"Subject: {doc.metadata.get('subject', 'N/A')}")

    file_size = Path(path).stat().st_size / (1024 * 1024)
    print(f'File size: {file_size:.2f} MB')

    print(f'Number of pages: {len(doc)}')
    print(f'Total number of chars: {df["char_count"].sum()}')

    print('\nDescribe char distribution')
    display(df['char_count'].describe())

    return df


def create_chunks(pages_data: pd.DataFrame,
                  chunk_size: int,
                  overlap: int) -> pd.DataFrame:
    """
    Split page texts into overlapping chunks with intelligent sentence boundary detection.

    This function processes text from multiple pages, splitting long texts into smaller
    chunks while attempting to preserve sentence boundaries. Each chunk overlaps with
    the next to maintain context continuity. Text is normalized by removing extra
    whitespace before chunking.

    Args:
        pages_data (pd.DataFrame): DataFrame containing page data with columns:
            - text (str): Text content of the page.
            - page_num (int): Page number identifier.
        chunk_size (int): Maximum character length for each chunk. Chunks may be
            slightly smaller if sentence boundary splitting is applied.
        overlap (int): Number of characters to overlap between consecutive chunks.
            Must be less than chunk_size.

    Returns:
        pd.DataFrame: DataFrame containing text chunks with the following columns:
            - chunk_id (int): Unique identifier for each chunk (sequential).
            - text (str): The chunk text content (stripped of leading/trailing spaces).
            - page_num (int): Original page number from which the chunk was extracted.
            - char_count (int): Number of characters in the chunk text.
            - start_char (int): Starting character position in the original page text.
            - end_char (int): Ending character position in the original page text.

    Note:
        - Text is normalized by stripping whitespace and collapsing multiple spaces.
        - For chunks near the end, the function attempts to split at the last period
          within the last 100 characters to preserve sentence integrity.
        - Pages with text shorter than chunk_size are kept as single chunks.
        - Progress is displayed using tqdm during processing.
    """
    chunks = []
    chunk_id = 0

    for page in tqdm(range(pages_data.shape[0]), desc='Chunking', total=pages_data.shape[0]):
        text = pages_data.iloc[page]['text']
        page_num = pages_data.iloc[page]['page_num']

        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

        if len(text) <= chunk_size:
            chunks.append({
                'chunk_id': chunk_id,
                'text': text,
                'page_num': page_num,
                'char_count': len(text),
                'start_char': 0,
                'end_char': len(text),
            })

            chunk_id += 1

        else:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]

                if end < len(text):
                    last_period = chunk_text.rfind('. ', -100)

                    if last_period != -1:
                        end = start + last_period + 2
                        chunk_text = text[start:end]

                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text.strip(),
                    'page_num': page_num,
                    'char_count': len(chunk_text),
                    'start_char': start,
                    'end_char': end,
                })

                start = end - overlap

                chunk_id += 1

    return pd.DataFrame(chunks)


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