from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema import SystemMessage, HumanMessage
from config.settings import GROQ_API_KEY, LLM_MODEL, TEMPERATURE
import os

class LLMService:
    """Service for interacting with LLMs."""
    
    def __init__(self):
        # Set API key
        os.environ['GROQ_API_KEY'] = GROQ_API_KEY
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
        )
    
    def get_llm(self):
        """Return the LLM instance."""
        return self.llm
    
    def create_qa_chain(self, retriever):
        """Create a QA chain using the retriever."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
    
    def create_llm_chain(self, prompt_template):
        """Create an LLM chain with the given prompt template."""
        return LLMChain(llm=self.llm, prompt=prompt_template)
    
    def invoke_llm(self, messages):
        """Invoke the LLM with the given messages."""
        return self.llm.invoke(messages)
    
    def invoke_with_system_human(self, system_content, human_content):
        """Invoke the LLM with system and human messages."""
        return self.llm([
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ])