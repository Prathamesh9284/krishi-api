from langchain.prompts import PromptTemplate

def get_intent_analysis_prompt():
    """Return the prompt template for intent analysis."""
    return PromptTemplate(
        input_variables=['query'],
        template="User's query:{query}\n\nUnderstand the user's intent: determine if they want to ask a agri-related question or want to search fertilizer and pesticide on ineternet. Reply with only 'agri_bot_agent' or 'search_agent' based on their intent."
    )

def format_agri_bot_messages(query, context):
    """Format messages for the agricultural bot agent."""
    return f"""
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

def get_search_system_prompt():
    """Return the system prompt for the search agent."""
    return "You are an AI assistant helping with product searches."

def format_search_human_prompt(query, search_results):
    """Format the human prompt for the search agent."""
    return f"Based on the following search results, summarize the best options for {query} in india: {search_results} give the link as well"