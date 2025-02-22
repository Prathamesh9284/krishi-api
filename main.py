from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
import os 
from langchain_huggingface import HuggingFaceEmbeddings  # Updated Import
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain.chains import LLMChain
from langgraph.checkpoint.memory import MemorySaver
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

# print(groq)
# print(tavily)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = FAISS.load_local(
    folder_path=r'D:\KrishiShakti\rag\notebooks\vectors',  # Path to the folder where the index is saved
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)


llm = ChatGroq(
    model='deepseek-r1-distill-llama-70b',
    temperature=0.6,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)


class State(TypedDict):
    messages : Annotated[list,add_messages]
    
class Activity:
    activity = ''

class Response:
    response = ''
    tool =''
    
activity = Activity()

response = Response()

def analyze(state:State):
    prompt_template = PromptTemplate(
        input_variables = ['query'],
        template = "User's query:{query}\n\nUnderstand the user's intent: determine if they want to ask a agri-related question or want to search fertilizer and pesticide on ineternet. Reply with only 'agri_bot_agent' or 'search_agent' based on their intent."
    )
    llm_chain = LLMChain(llm=llm,prompt=prompt_template)
    message = state['messages'][-1]
    result = llm_chain.run(message)
    activity.activity = result.split('</think>')[-1].split('\n')[-1]

def agri_bot_agent(state:State):
    query = state['messages'][-1].content
    context = qa_chain.invoke({"query": query}).get("source_documents", [])[0].page_content
    
    formatted_messages = f"""
        developer : You are an AI language model designed to answer queries strictly related to the agriculture.
        You must not respond to any questions that fall outside the scope of agriculture.
        Understand the user's intent if the user generally says that he/she wants to know about the agri.
        If a user greets you (e.g., "Hi," "Hello"), you may respond politely and inform them that you are here to answer questions only related to agriculture.
        Your responses should be based on the information available in the policy document provided to you.
        If the requested information is not present in the document, you can anser them based on your knowledge.
        
        Ensure that all responses are well-structured and easy to understand.
        
        assistant : {context}
        
        user : {query}
    """
    resp = llm.invoke(formatted_messages)
    response.tool = 'rag'
    # print(context)
    return {'messages':[resp.content]}

    
def search_agent(state:State):
    # Initialize Tavily search tool
    search = TavilySearchResults()
    search_results = search.run(state['messages'][-1].content)
    
    resp = llm([
        SystemMessage(content="You are an AI assistant helping with product searches."),
        HumanMessage(content=f"Based on the following search results, summarize the best options for {state['messages'][-1].content} in india: {search_results} give the link as well")
    ])
    
    response.tool = 'search'
    return {'messages':[resp.content]}


graph_builder = StateGraph(State)
graph_builder.add_node('analyze',analyze)
# graph_builder.add_node(START,'analyze')
graph_builder.add_node('agri_bot_agent',agri_bot_agent)
graph_builder.add_node('search_agent',search_agent)
graph_builder.add_edge(START,'analyze')
graph_builder.add_edge('agri_bot_agent',END)
graph_builder.add_edge('search_agent',END)
graph_builder.add_conditional_edges(
    'analyze',  
    lambda state: 'agri_bot_agent' if activity.activity=='agri_bot_agent' else 'search_agent',
    {
        'agri_bot_agent':'agri_bot_agent',
        'search_agent':'search_agent'
    }
)

graph = graph_builder.compile()

def graph_stream_update(input:str):
    events = graph.stream(
        {'messages':[{'role':'user','content':input}]},
         {"configurable": {"thread_id": "1"}},
        stream_mode="values",
    )
    for event in events:
        response.response = event['messages'][-1].content.split('</think>')[-1]
    return {
        'response':(response.response),
        'tool': response.tool
    }



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.post("/agent")
async def agent(
    query: str = Form(...)
):
    return graph_stream_update(query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
