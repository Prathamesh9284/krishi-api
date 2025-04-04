from fastapi import APIRouter, Form
from services.agent_service import AgentService

class AgentRoutes:
    """Routes for agent functionality."""
    
    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up the routes."""
        self.router.add_api_route(
            path="/agent",
            endpoint=self.process_agent_query,
            methods=["POST"]
        )
    
    async def process_agent_query(self, query: str = Form(...)):
        """Process an agent query."""
        return self.agent_service.process_query(query)