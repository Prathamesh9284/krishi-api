from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL

class EmbeddingService:
    """Service for generating embeddings."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    def get_embeddings(self):
        """Return the embeddings model."""
        return self.embeddings