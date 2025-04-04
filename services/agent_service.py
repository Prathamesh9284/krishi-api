from langgraph.graph import StateGraph, START, END
from models.state import State, AgentResponse, ActivityTracker
from utils.prompts import (
    get_intent_analysis_prompt, 
    format_agri_bot_messages, 
    get_search_system_prompt, 
    format_search_human_prompt
)

class AgentService:
    """Service for agent functionality and graph definition."""
    
    def __init__(self, llm_service, vector_store_repo, search_service):
        self.llm_service = llm_service
        self.vector_store_repo = vector_store_repo
        self.search_service = search_service
        self.qa_chain = llm_service.create_qa_chain(vector_store_repo.get_retriever())
        self.activity_tracker = ActivityTracker()
        self.agent_response = AgentResponse()
        self.graph = self._build_graph()
    
    def _analyze(self, state: State):
        """Analyze user intent to determine which agent to use."""
        prompt_template = get_intent_analysis_prompt()
        llm_chain = self.llm_service.create_llm_chain(prompt_template)
        message = state['messages'][-1]
        result = llm_chain.run(message)
        self.activity_tracker.activity = result.split('</think>')[-1].split('\n')[-1]
    
    def _agri_bot_agent(self, state: State):
        """Handle agricultural-related queries."""
        query = state['messages'][-1].content
        context = self.qa_chain.invoke({"query": query}).get("source_documents", [])[0].page_content
        
        formatted_messages = format_agri_bot_messages(query, context)
        resp = self.llm_service.invoke_llm(formatted_messages)
        self.agent_response.tool = 'rag'
        
        return {'messages': [resp.content]}
    
    def _search_agent(self, state: State):
        """Handle search-related queries."""
        query = state['messages'][-1].content
        search_results = self.search_service.search(query)
        
        system_prompt = get_search_system_prompt()
        human_prompt = format_search_human_prompt(query, search_results)
        
        resp = self.llm_service.invoke_with_system_human(system_prompt, human_prompt)
        self.agent_response.tool = 'search'
        
        return {'messages': [resp.content]}
    
    def _conditional_edge(self, state):
        """Determine which agent to route to based on activity."""
        return 'agri_bot_agent' if self.activity_tracker.activity == 'agri_bot_agent' else 'search_agent'
    
    def _build_graph(self):
        """Build and compile the agent graph."""
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node('analyze', self._analyze)
        graph_builder.add_node('agri_bot_agent', self._agri_bot_agent)
        graph_builder.add_node('search_agent', self._search_agent)
        
        # Add edges
        graph_builder.add_edge(START, 'analyze')
        graph_builder.add_edge('agri_bot_agent', END)
        graph_builder.add_edge('search_agent', END)
        
        # Add conditional edge
        graph_builder.add_conditional_edges(
            'analyze',
            self._conditional_edge,
            {
                'agri_bot_agent': 'agri_bot_agent',
                'search_agent': 'search_agent'
            }
        )
        
        return graph_builder.compile()
    
    def process_query(self, query):
        """Process a user query through the agent graph."""
        events = self.graph.stream(
            {'messages': [{'role': 'user', 'content': query}]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="values",
        )
        
        for event in events:
            self.agent_response.response = event['messages'][-1].content.split('</think>')[-1]
        
        return {
            'response': self.agent_response.response,
            'tool': self.agent_response.tool
        }