from pathlib import Path
from datetime import datetime
import uvicorn, nest_asyncio, sys, torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import warnings, logging
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

sys.path.append('..')
from src.data_utils import download_file, load_and_analyze_pdf, create_chunks
from src.langchain_RAG import setup_data_collection, langchain_rag_pipeline


logging.basicConfig(level=logging.INFO)
api_logger = logging.getLogger('rag_api')


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Manage the lifespan of the FastAPI application with startup and shutdown events.

    This async context manager handles the initialization and cleanup of application
    resources, specifically the vector database connection. It runs before the
    application starts serving requests and ensures proper cleanup on shutdown.

    The function performs the following operations:
    - Startup: Initializes the vector database (ChromaDB collection) with pre-computed
      embeddings from the chunks file
    - Shutdown: Cleans up resources by setting vectorstore to None

    Args:
        _app (FastAPI): The FastAPI application instance (prefixed with underscore as
            it's not used in the function body).

    Yields:
        None: Control is yielded back to FastAPI after startup, and the function
            resumes execution during shutdown.

    Raises:
        Exception: Re-raises any exception that occurs during vector database
            initialization to prevent the application from starting in a broken state.

    Note:
        - Uses global `vectorstore` variable to make it accessible across the application.
        - The vector database is recreated on each startup (overwrite=True).
        - All initialization steps are logged using api_logger for monitoring.
        - If initialization fails, the application will not start.
        - This follows FastAPI's lifespan event pattern introduced in version 0.93.0+.

    See Also:
        setup_data_collection: Function that initializes the ChromaDB collection.
    """
    global vectorstore
    try:
        api_logger.info('Establishing vector database...')
        vectorstore = setup_data_collection(
            chunks_filename='final_chunks',
            collection_name='aws_docs_final',
            overwrite=True,
            device=device,
        )
        api_logger.info('Vector database initialized successfully')
    except Exception as e:
        api_logger.error(f'Failed to initialize vector database: {str(e)}')
        raise e

    yield
    api_logger.info('Shutting down...')
    vectorstore = None


vectorstore = None


app = FastAPI(
    title='Industrial RAG Agent API',
    description='RAG',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc',
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # in production this must be specified
    allow_credentials=True,
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)

