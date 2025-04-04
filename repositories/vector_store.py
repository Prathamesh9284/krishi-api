from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import VECTOR_STORE_PATH, EMBEDDING_MODEL

class VectorStoreRepository:
    """Repository for interacting with vector stores."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.load_vector_store()
    
    def load_vector_store(self):
        """Load the vector store from disk."""
        self.vector_store = FAISS.load_local(
            folder_path=VECTOR_STORE_PATH,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def get_retriever(self):
        """Get the retriever from the vector store."""
        return self.vector_store.as_retriever()