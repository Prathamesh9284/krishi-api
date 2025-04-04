from langchain_community.tools.tavily_search import TavilySearchResults
from config.settings import TAVILY_API_KEY
import os

class SearchService:
    """Service for performing internet searches."""
    
    def __init__(self):
        # Set API key
        os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
        
        # Initialize search tool
        self.search_tool = TavilySearchResults()
    
    def search(self, query):
        """Perform a search with the given query."""
        return self.search_tool.run(query)