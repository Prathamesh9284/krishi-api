import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Model settings
LLM_MODEL = 'deepseek-r1-distill-llama-70b'
TEMPERATURE = 0.6
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Vector store settings
VECTOR_STORE_PATH = 'vectors'

# API settings
HOST = "0.0.0.0"
PORT = 8000