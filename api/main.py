from pathlib import Path
from datetime import datetime
import uvicorn, nest_asyncio, sys, torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
device = 'cuda' if torch.cuda.is_available() else 'cpu'