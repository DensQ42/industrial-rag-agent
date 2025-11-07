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
from models import QueryRequest, QueryResponse, HealthResponse


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


@app.get('/health', response_model=HealthResponse)
async def health_check():
    """
    Check the health status of the RAG API service.

    This endpoint provides real-time information about the service health,
    including the availability of the vector database. It's designed for
    monitoring, load balancing, and automated health checks by orchestration
    systems like Kubernetes or Docker Swarm.

    The health status is determined as follows:
    - 'healthy': Vector database is loaded and available
    - 'degraded': Service is running but vector database is not available
    - 'unhealthy': An unexpected error occurred during the health check

    Returns:
        HealthResponse: A Pydantic model containing:
            - status (str): Overall service health ('healthy', 'degraded', or 'unhealthy')
            - timestamp (str): ISO 8601 formatted timestamp of the health check
            - vectorstore_loaded (bool): Whether the vector database is initialized

    Raises:
        None: All exceptions are caught and returned as 'unhealthy' status rather
            than propagating errors to the caller.

    Note:
        - This endpoint always returns HTTP 200, even when unhealthy.
        - The endpoint is lightweight and suitable for frequent polling.
        - Errors during health checks are logged but don't crash the endpoint.
        - This follows the health check pattern recommended for microservices.
        - No authentication is required for this endpoint.

    See Also:
        HealthResponse: The response model definition.
    """
    try:
        vectorstore_loaded = vectorstore is not None

        if vectorstore_loaded:
            overall_status = 'healthy'
        else:
            overall_status = 'degraded'

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            vectorstore_loaded=vectorstore_loaded,
        )

    except Exception as e:
        api_logger.error(f'Health check failed: {str(e)}')
        return HealthResponse(
            status='unhealthy',
            timestamp=datetime.now().isoformat(),
            vectorstore_loaded=False,
        )


@app.post('/query', response_model=QueryResponse)
async def query_rag(q: QueryRequest):
    """
    Process a user question through the RAG pipeline and return a generated answer.

    This endpoint receives a user's question about AWS documentation, retrieves
    relevant context from the vector database using semantic search, and generates
    a comprehensive answer using a language model. The entire RAG (Retrieval-Augmented
    Generation) pipeline is executed, including context retrieval, prompt construction,
    and answer generation.

    Args:
        q (QueryRequest): A Pydantic model containing the user's question with
            validation for length and format.

    Returns:
        QueryResponse: A Pydantic model containing:
            - timestamp (str): ISO 8601 formatted timestamp when the response was generated
            - question (str): The original user question echoed back
            - answer (str): The generated answer from the RAG pipeline

    Raises:
        HTTPException (400): If the query request validation fails (handled by FastAPI/Pydantic).
        HTTPException (500): If an unexpected error occurs during RAG pipeline execution.
        HTTPException: Any HTTP exceptions from the RAG pipeline are re-raised as-is.

    Note:
        - Processing time is logged for monitoring and performance analysis.
        - The vectorstore must be initialized (checked in lifespan) for this endpoint to work.
        - All errors are logged before raising HTTP exceptions.
        - The endpoint is async but calls synchronous RAG pipeline internally.
        - Response time varies based on query complexity and retrieved context size.

    See Also:
        QueryRequest: The request model definition.
        QueryResponse: The response model definition.
        langchain_rag_pipeline: The core RAG processing function.
    """
    start_time = datetime.now()

    try:
        result = langchain_rag_pipeline(q.question, vectorstore)

        response = QueryResponse(
            timestamp=datetime.now().isoformat(),
            question=q.question,
            answer=result['answer'],
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        api_logger.info(f'Completed successfully in {processing_time:.2f}ms')
        return response

    except HTTPException:
        raise

    except Exception as e:
        api_logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed: {str(e)}')