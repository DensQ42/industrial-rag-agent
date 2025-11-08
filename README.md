# Industrial RAG Agent
This project implements a production-ready Retrieval-Augmented Generation (RAG) system designed for querying technical documentation. In the beginning the core mechanics were built from scratch, then refactored using LangChain.

This project demonstrates end-to-end ML engineering: from data processing and chunking strategies to quantitative evaluation with RAGAS and REST API deployment with Docker.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.0-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.2-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Key Features
- Key concepts were built manually from scratch to understand retrieval, prompt engineering, and generation pipeline
- Production-ready refactoring using LangChain framework (LCEL chains)
- RAGAS metrics (faithfulness: 0.99, answer relevancy: 0.69) with A/B testing of chunking strategies
- Data-driven selection of chunk size (1000 chars), overlap (200 chars), and retrieval parameters ($k=5$)
- All generated answers include page number citations for verification
- Strict prompt engineering and temperature tuning ensure that responses are grounded only in retrieved context
- FastAPI endpoints with Pydantic validation, automatic documentation, and proper error handling
- Docker-ready containerized deployment

## Dataset Description
This projects uses [AWS Toolkit for Microsoft Azure DevOps - User Guide](https://docs.aws.amazon.com/pdfs/vsts/latest/userguide/vsts-ug.pdf) (100+ pages) as an example of real technical documentation with complex structure.

## Project Structure
The project uses custom utility module to keep clean notebook structure.
```
INDUSTRIAL_RAG_AGENT/
├── api/
│   ├── main.py                      # API endpoints and application setup
│   └── models.py                    # Pydantic models for request/response
├── data/
│   ├── chromadb/                    # Vector Database Chroma
│   ├── processed/                   # Processed files (not all of them in repository)
│   │   ├── chunks.json
│   │   └── final_chunks.py
│   └── raw/                         # Original PDF documents (not in repository)
├── notebooks/                       # Jupyter notebooks for analysis and development
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_manual_RAG.ipynb
│   ├── 03_LangChain_RAG.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_deployment.ipynb
├── src/                             # Helper utility functions
│   ├── data_utils.py
│   ├── langchain_RAG.py
│   └── manual_RAG.py
├── .dockerignore
├── .env.example                     # Environment variables template (Antropic Console API key)
├── docker-compose.yml
├── Dockerfile
├── requirements-docker.txt          # Production dependencies
└── requirements-full.txt            # Dependencies for the entire project
```

## Methodology

### 1. Data Processing Pipeline (notebook 1)
- Used PyMuPDF for text extraction from PDF document with metadata preservation
- Split the extracted text into overlapping sentence-boundary-aware chunks
- Established semantic vector representation using sentence transformer `all-MiniLM-L6-v2` model
- Built vector database store with cosine similarity indexing using ChromaDB

### 2. RAG Architecture: Manual Implementation (Notebook 2)
- Create custom semantic search function using cosine similarity
- Engineered prompt template with clear structured instruction of 8 steps
- Integrated directly Anthropic Claude Haiku 4.5 model with temperature 0.3 via API for answer generating

### 3. RAG Architecture: LangChain Implementation (Notebook 3)
Implemented production-ready RAG pipeline using LangChain Expression Language (LCEL) chains with 3 main elements: modular retriever, prompt template and LLM (Haiku 4.5).

### 4. Testing and Evaluation (Notebook 4)
- Created custom dataset of 12 relevant and 3 irrelevant query-answer pairs regarding selected PDF document
- Evaluated 3 chunking configurations (200, 500, 1000 chars) across 4 RAGAS metrics (answer relevancy, faithfulness, context precision and context recall) using Claude Sonnet 4.5 as evaluator with 0 value temperature
- Analyzed results and applied the best chunking strategy with 1000 char size and 200 overlap

### 5. Deployment (Notebook 5)
- Deployed FastAPI-based REST API with comprehensive validation
- Implemented Docker containerization for consistent deployment

## Results

### Configuration Comparison

| Chunk Size | Overlap | Answer Relevancy | Faithfulness | Context Precision | Context Recall |
|------------|---------|------------------|--------------|-------------------|----------------|
| **1000**   | **200** | **0.690**        | **0.992**    | **0.674**         | **1.000**      |
| 500        | 100     | 0.639            | 0.968        | 0.652             | 0.933          |
| 200        | 50      | 0.442            | 0.943        | 0.650             | 0.900          |

### Key Findings
- Larger chunks consistently outperform smaller chunks across all metrics
- Faithfulness remains high (>0.94) across all configurations, that confirms well designed hallucinating-prevent prompt instruction
- Answer relevancy shows most variance with the largest difference of 0.248
- Perfect context recall (1.0) with 1000-char chunks showing that no information lost with this strategy

## Installation and Setup
### Prerequisites
- Python 3.12+
- Docker (optional)
- Anthropic API key

### Option 1: Local Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/DensQ42/industrial-rag-agent.git
   cd industrial-rag-agent
   ```
2. [Optional] **Create and activate virtual environment** (for example using conda)
   ```bash
   conda create -n rag_agent python=3.12
   conda activate rag_agent
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements-full.txt
   ```
4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   ```
5. **Run notebooks**
   - Run notebooks (01-05)
   - Optionally, change the URL address of the PDF document
6. **Run the API**
   ```bash
   uvicorn api/main.py
   ```
   Or just run the last cell in the fifth notebook

### Option 2: Docker Deployment
1. **Clone the repository**
   ```bash
   git clone https://github.com/DensQ42/industrial-rag-agent.git
   cd industrial-rag-agent
   ```
2. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   ```
3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```
The API will be available at:
- **API Server**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **API Documentation**: http://localhost:8000/redoc

## API Usage
### Endpoints
- `GET /` - API information and available endpoints
- `GET /health` - Service health check
- `POST /query` - Main RAG pipeline endpoint

### Example Usage
**Query:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "How do I create an AWS account?"
}'
```

**Response Format:**
```json
{
  "timestamp": "2025-11-08T21:50:51.568649",
  "question": "How do I create an AWS account?",
  "answer": "To create an AWS account, follow these steps (Page 10):\n\n1. Open https://portal.aws.amazon.com/billing/signup\n\n2. Follow the online instructions\n\n3. Complete the verification process, which involves receiving a phone call or text message and entering a verification code on the phone keypad\n\nWhen you sign up for an AWS account, an AWS account root user is created. This root user has access to all AWS services and resources in the account. As a security best practice, you should assign administrative access to an IAM user and use the root user only for tasks that require root user access (Page 10)."
}
```

## Tech Stack
**Data:** Pandas, ChromaDB, PyMuPDF <br>
**LLM:** Hugging Face Transformers, Anthropic Claude, LangChain, RAGAS <br>
**Deployment:** FastAPI, Pydantic, Uvicorn, Docker

## Limitations & Future Work
Current version works with text-based PDF documents. For scanned documents, OCR preprocessing (pytesseract/easyOCR) can be added as a pipeline step.

## License
This project is licensed under the MIT License.