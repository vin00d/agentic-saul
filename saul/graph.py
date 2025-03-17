from typing import Annotated, Dict, TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from tools import tools
from loguru import logger


##### Research Subgraph #####
# State for the research agent
class ResearchState(TypedDict):
    """State for the research agent."""
    messages: Annotated[list, add_messages]

# Initialize the LLM
research_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# bind tools to the llm
research_model = research_model.bind_tools(tools)


# Agent node implementation
async def pick_research_tool(state: ResearchState) -> Dict:
    """Agent node that decides which tool to use."""
    logger.success(f"Calling research_model from research_agent with state: {state}")
    # print(f"State:: {state}\n\n")
    response = research_model.invoke(state["messages"])
    return {"messages": [response]}


execute_tool = ToolNode(tools)

# Create the Research Sub Graph
research_sg_builder = StateGraph(ResearchState)

# Add nodes
research_sg_builder.add_node("research_agent", pick_research_tool)
research_sg_builder.add_node("research_action", execute_tool)


# conditional edge function
def should_continue(state: ResearchState) -> str:
    last_message = state["messages"][-1]

    if last_message.get("tool_calls"):
        return "research_action"

    return END


# Add edges
research_sg_builder.add_edge(START, "research_agent")
research_sg_builder.add_conditional_edges("research_agent", should_continue)
research_sg_builder.add_edge("research_action", "research_agent")


# system_prompt = """You are a helpful legal research assistant. 
# Only answer questions that are related to legal research, else politely decline to answer.
# Only answer the last question.
# """

## Report Graph
# State for the report agent
class ReportState(TypedDict):
    """State for the report agent."""
    messages: Annotated[list, add_messages]

# Initialize the LLM
report_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
# bind tools to the llm
report_model = report_model.bind_tools(tools)
# Agent node implementation
async def pick_report_tool(state: ReportState) -> Dict:
    """Agent node that decides which tool to use."""
    logger.success(f"Calling report_model from report_agent with state: {state}")
    # print(f"State:: {state}\n\n")
    response = report_model.invoke(state["messages"])
    return {"messages": [response]}

execute_tool = ToolNode(tools)


## Create full graph
builder = StateGraph(ReportState)
# Add nodes
builder.add_node("conduct_research", research_sg_builder.compile())
builder.add_node("report_agent", pick_report_tool)