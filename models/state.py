from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    """State definition for the agent graph."""
    messages: Annotated[list, add_messages]

class AgentResponse:
    """Class to hold agent responses."""
    def __init__(self):
        self.response = ''
        self.tool = ''

class ActivityTracker:
    """Class to track the current activity selection."""
    def __init__(self):
        self.activity = ''