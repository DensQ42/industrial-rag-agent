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


def setup_data_collection(
    chunks_filename: str = 'chunks',
    collection_name: str = 'aws_docs_langchain',
    overwrite: bool = False,
    device: str = 'cpu',
) -> Chroma:
    """
    Prepare and load a Chroma vector database with document embeddings using LangChain and Hugging Face.

    This function either creates a new Chroma collection from preprocessed text chunks
    or loads an existing collection if `overwrite` is set to False. When creating a new
    collection, it reads text chunks from a JSON file, converts them into LangChain
    `Document` objects, and computes embeddings using the Hugging Face
    `all-MiniLM-L6-v2` model.

    Args:
        chunks_filename (str, optional):
            Base filename (without extension) for the JSON file containing text chunks.
            The file is expected at `../data/processed/{chunks_filename}.json`.
            Defaults to `'chunks'`.
        collection_name (str, optional):
            Name of the Chroma collection to create or load. Defaults to `'aws_docs_langchain'`.
        overwrite (bool, optional):
            If True, deletes any existing collection with the same name and rebuilds it
            from the JSON file. If False, loads an existing persistent collection.
            Defaults to False.
        device (str, optional):
            Device for embedding computation (`'cpu'` or `'cuda'`). Defaults to `'cpu'`.

    Returns:
        Chroma:
            A Chroma vector store object ready for similarity search or RAG pipelines.

    Notes:
        - Embeddings are computed using the `'sentence-transformers/all-MiniLM-L6-v2'` model.
        - The Chroma database is stored persistently at `'../data/chromadb'`.
        - The vector space uses cosine similarity for nearest-neighbor search.
    """
    embedding_function = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32},
    )

    if overwrite:
        chunks = pd.read_json(f'../data/processed/{chunks_filename}.json', orient='records')

        documents: List[Document] = [
            Document(
                page_content=row['text'],
                metadata={
                    'chunk_id': int(row['chunk_id']),
                    'page_num': int(row['page_num']),
                    'char_count': int(row['char_count']),
                    'start_char': int(row['start_char']),
                    'end_char': int(row['end_char']),
                },
            )
            for _, row in chunks.iterrows()
        ]

        try:
            client = chromadb.PersistentClient(path='../data/chromadb')
            client.delete_collection(collection_name)
        except Exception:
            pass

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory='../data/chromadb',
            collection_metadata={'hnsw:space': 'cosine'},
        )

    else:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory='../data/chromadb',
        )

    return vectorstore


def langchain_rag_pipeline(
    query: str,
    vectorstore: Chroma,
    k: int = 5,
    temperature: float = 0.3,
    llm_name: str = 'claude-haiku-4-5-20251001',
) -> dict:
    """
    Execute a LangChain Retrieval-Augmented Generation (RAG) pipeline using a Chroma vectorstore
    and an Anthropic large language model (LLM).

    The pipeline performs the following steps:
    1. Retrieves the top-k most relevant document chunks from the provided vectorstore.
    2. Extracts and stores the raw text content (`page_content`) from retrieved documents.
    3. Formats the retrieved documents into a single context string using the
       `format_retrieved_docs` callable (expected to be available in the current scope).
    4. Constructs a structured prompt enforcing context-based answering only, with page citations.
    5. Invokes the Anthropic LLM and returns the generated plain-text answer along with
       the retrieved contexts.

    Args:
        query (str):
            The user's input question to be answered based on retrieved context.
        vectorstore (Chroma):
            A Chroma vectorstore instance providing similarity-based document retrieval
            via the `.as_retriever(...)` interface.
        k (int, optional):
            Number of most relevant chunks to retrieve from the vectorstore. Defaults to 5.
        temperature (float, optional):
            Sampling temperature for the LLM. Lower values yield more deterministic responses.
            Defaults to 0.3.
        llm_name (str, optional):
            Identifier of the Anthropic model to use. Defaults to `'claude-haiku-4-5-20251001'`.

    Returns:
        dict:
            A dictionary with two keys:
            - 'answer' (str): The plain-text answer generated by the LLM.
            - 'contexts' (List[str]): The list of text contents from retrieved document chunks.

    Raises:
        EnvironmentError:
            If the `ANTHROPIC_API_KEY` environment variable is missing when initializing the LLM.
        AttributeError:
            If the provided `vectorstore` object does not support `.as_retriever(...)` or
            produces an incompatible output for `format_retrieved_docs`.
        ValueError:
            If `k` is not a positive integer.

    Notes:
        - The function assumes that a callable `format_retrieved_docs` is defined and
          can format a list of `Document` objects into a single context string.
        - The prompt enforces that the LLM:
            * Uses only the retrieved context,
            * Cites page numbers,
            * Outputs plain text (no markdown or formatting).
        - The Anthropic API key is read from the `ANTHROPIC_API_KEY` environment variable.
        - Retrieved document texts are also returned for reference and transparency.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError('k must be a positive integer.')

    prompt_template = """You are an expert at answering questions about Amazon Web Services documentation.

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
{context}

USER QUESTION:
{query}

Think step-by-step, then provide your final ANSWER only without steps.

ANSWER:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'query'])

    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise EnvironmentError('ANTHROPIC_API_KEY not found in environment.')

    llm = ChatAnthropic(
        model=llm_name,
        temperature=temperature,
        max_tokens=500,
        anthropic_api_key=api_key,
    )

    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': k})

    retrieved_docs = retriever.invoke(query)
    contexts = [doc.page_content for doc in retrieved_docs]

    retriever_step = RunnableParallel(
        {
            'context': retriever | format_retrieved_docs,
            'query': RunnablePassthrough(),
        }
    )

    chain = (retriever_step | prompt | llm | StrOutputParser())

    result = chain.invoke(query)

    return {
        'answer': result,
        'contexts': contexts,
    }
