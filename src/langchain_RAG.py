import pandas as pd
import re, os
from tqdm.auto import tqdm
import chromadb
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from typing import List


def format_retrieved_docs(docs: List[Document]) -> str:
    """
    Format and concatenate a list of retrieved documents into a single text block.

    Each document is numbered and prefixed with its corresponding context chunk
    and page number, providing a clear structure for further processing in RAG
    (Retrieval-Augmented Generation) pipelines or text analysis.

    Args:
        docs (List[Document]):
            A list of `Document` objects, where each document contains text content
            and metadata with at least a `'page_num'` key.

    Returns:
        str:
            A formatted string containing all document chunks separated by two
            newlines. Each chunk includes its index, page number, and text content.
    """
    context = []
    for idx, doc in enumerate(docs, 1):
        context.append(f"[Context chunk {idx} - Page {doc.metadata['page_num']}]\n{doc.page_content}")
    return "\n\n".join(context)


def create_chunks(pages_data: pd.DataFrame,
                  chunk_size: int,
                  overlap: int,
                  EOS:int = 100) -> pd.DataFrame:
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
        EOS (int, optional): Number of characters from the end of each chunk
            to search backward for a sentence boundary (a period followed by a space).
            This helps ensure that chunks end at natural sentence boundaries, improving
            text coherence in RAG contexts. Default is 100.

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
                    last_period = chunk_text.rfind('. ', -EOS)

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
