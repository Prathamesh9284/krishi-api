from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import services
from services.llm_service import LLMService
from services.embedding_service import EmbeddingService
from services.search_service import SearchService
from services.agent_service import AgentService

# Import repositories
from repositories.vector_store import VectorStoreRepository

# Import routes
from routes.agent_routes import AgentRoutes

# Import config
from config.settings import HOST, PORT

def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize services and repositories
    embedding_service = EmbeddingService()
    llm_service = LLMService()
    vector_store_repo = VectorStoreRepository()
    search_service = SearchService()
    
    # Initialize agent service
    agent_service = AgentService(
        llm_service=llm_service,
        vector_store_repo=vector_store_repo,
        search_service=search_service
    )
    
    # Initialize routes
    agent_routes = AgentRoutes(agent_service)
    
    # Include routers   
    app.include_router(agent_routes.router)
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)