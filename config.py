import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VECTOR_DB_PATH = './db'
COLLECTION_NAME = 'csn_python'
TOP_K = 4

# UTSA LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GENERATOR_MODEL = "gpt-4o-mini"

