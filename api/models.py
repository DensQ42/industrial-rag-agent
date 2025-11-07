from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.

    This Pydantic model defines the structure of the health check response,
    providing information about the service status, timing, and vector database
    availability. Used for monitoring and debugging purposes.

    Attributes:
        status (str): Service status indicating overall health ('healthy' or 'unhealthy').
        timestamp (str): ISO 8601 formatted timestamp of when the health check was performed.
        vectorstore_loaded (bool): Indicates whether the vector database is successfully
            loaded and available for queries.

    Note:
        - All fields are required (indicated by ellipsis `...` in Field).
        - Timestamp should follow ISO 8601 format (YYYY-MM-DDTHH:MM:SS.mmmmmm).
        - This model is typically returned by GET /health endpoint.
    """
    status: str = Field(..., description='Service status (healthy/unhealthy)')
    timestamp: str = Field(..., description='Health check timestamp (ISO format)')
    vectorstore_loaded: bool = Field(..., description='Whether vector database is loaded')


class QueryRequest(BaseModel):
    """
    Request model for RAG query endpoint.

    This Pydantic model defines the structure and validation rules for incoming
    query requests to the RAG (Retrieval-Augmented Generation) system. It ensures
    that user questions meet minimum quality standards before processing.

    Attributes:
        question (str): The user's question about AWS documentation. Must be between
            5 and 500 characters in length to ensure meaningful queries while preventing
            abuse or overly complex questions.

    Note:
        - Questions shorter than 5 characters will raise a validation error.
        - Questions longer than 500 characters will be rejected.
        - This model is typically used with POST /query endpoint.
        - Field is required (indicated by ellipsis `...`).
    """
    question: str = Field(
        ...,
        description='User question about AWS documentation',
        min_length=5,
        max_length=500,
        examples=['How do I create an AWS account?'],
    )


class QueryResponse(BaseModel):
    """
    Response model for RAG query endpoint.

    This Pydantic model defines the structure of the response returned by the RAG
    (Retrieval-Augmented Generation) system after processing a user's question.
    It includes the generated answer along with metadata for tracking and auditing.

    Attributes:
        timestamp (str): ISO 8601 formatted timestamp indicating when the response
            was generated.
        question (str): The original user question that was submitted, echoed back
            for reference and logging purposes.
        answer (str): The generated answer produced by the RAG pipeline, synthesized
            from retrieved documentation chunks and language model processing.

    Note:
        - All fields are required (indicated by ellipsis `...` in Field).
        - Timestamp follows ISO 8601 format (YYYY-MM-DDTHH:MM:SS.mmmmmm).
        - This model is typically returned by POST /query endpoint.
        - The answer field length is not restricted as responses can vary significantly.
    """
    timestamp: str = Field(..., description='Response timestamp (ISO format)')
    question: str = Field(..., description='Original user question')
    answer: str = Field(..., description='Generated answer from RAG pipeline')
