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