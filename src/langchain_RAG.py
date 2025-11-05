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
