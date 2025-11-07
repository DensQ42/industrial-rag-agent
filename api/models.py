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