# config.py
import os
from dotenv import load_dotenv
from schemas import *

load_dotenv()

# --- Project Base Directory ---
# This ensures all paths are relative to the project's root folder, making it portable.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- API Keys & Model Configuration ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_KEY_1 = os.getenv("MISTRAL_API_KEY_1")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
CSE_ID = os.getenv("CSE_ID")
MISTRAL_MODEL = "mistral-large-latest"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
SPACY_MODEL = 'en_core_web_sm'

# --- File & Execution Configuration (CLOUD-READY PATHS) ---
RACE_INPUT_FILE = "races.json" # This remains a temporary name
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CRAWL_CACHE_DIR = os.path.join(BASE_DIR, "crawl_cache")
KNOWLEDGE_CACHE_DIR = os.path.join(BASE_DIR, "knowledge_cache")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

# --- Performance & Tuning Configuration ---
TOP_N_URLS_TO_PROCESS = 3
MAX_SEARCH_RESULTS = 10
MAX_RETRIES = 3
DEBUG = True # Set to False in production for cleaner logs
MAX_CONCURRENT_CRAWLERS = 5
MIN_CONFIDENCE_THRESHOLD = 0.65

# --- RAG & Re-ranking Configuration ---
RAG_CANDIDATE_POOL_SIZE = 50
RAG_FINAL_EVIDENCE_COUNT = 5