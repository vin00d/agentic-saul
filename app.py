from typing import List, Dict, TypedDict, Union, Annotated
import chainlit as cl
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from operator import itemgetter
from pydantic import BaseModel, Field, ConfigDict
from saul.tools import tools

from loguru import logger
import json
from dotenv import load_dotenv

load_dotenv()

# Types for our nodes
class AgentState(TypedDict):
    """State for the research agent."""
    messages: Annotated[list, add_messages]


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# bind tools to the llm
llm = llm.bind_tools(tools)


# Agent node implementation
async def call_model(state: AgentState) -> Dict:
    """Agent node that decides which tool to use."""
    logger.success(f"Calling agent model with state: {state}")
    # print("...........................................Calling agent model...........................................")
    # print(f"State:: {state}\n\n")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


execute_tool = ToolNode(tools)

# Create the graph
uncompiled_graph = StateGraph(AgentState)

# Add nodes
uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", execute_tool)


# conditional edge function
def should_continue(state):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "action"

    return END


# Add edges
uncompiled_graph.add_conditional_edges("agent", should_continue)
uncompiled_graph.add_edge("action", "agent")

# Set entry point
uncompiled_graph.set_entry_point("agent")

# Compile the graph
compiled_graph = uncompiled_graph.compile()

system_prompt = """You are a helpful legal research assistant. 
Only answer questions that are related to legal research, else politely decline to answer.
Only answer the last question.
"""

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    # Initialize session state
    initial_state = AgentState(
        messages=[SystemMessage(content=system_prompt)]
    )
    image = cl.Image(path="./bcs-calling-card.jpg")

    # Set initial state in session
    cl.user_session.set("state", initial_state)

    await cl.Message(
        content="""### Better Call Agentic-Saul
        🫵 Yeah! I'm Saul, your legal research assistant. 
        I'm here to help you with your legal research needs.
        
        I will find information from:
        - 📄 Legal Glossary - legal terms and definitions        
        - 📚 Wikipedia - basic information
        - 💬 Reddit discussions - current chatter in social media
        - 📖 Google Scholar Case Law - judicial opinions from numerous federal and state courts

        What would you like to research?""",
        elements=[image],

    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    # Get current session state
    state_dict = cl.user_session.get("state")
    state = AgentState(**state_dict)

    # Update messages in state
    state["messages"].append(HumanMessage(content=message.content))
    inputs = {"messages": state["messages"]}
    # try:
    msg = cl.Message(content="")
    # Run the graph with current state
    async for chunk in compiled_graph.astream(inputs, stream_mode="updates"):
        for node, values in chunk.items():

            logger.success(f"Receiving update from node: {node}")
            # print(f"-------------- Receiving update from node: '{node}' --------------")
            await msg.stream_token(f"Receiving update from node: **{node}**\n")
            if node == "action":
                for tool_msg in values["messages"]:
                    output = f"Tool used: {tool_msg.name}"
                    # output += f"\nTool output: {tool_msg.content}"
                    logger.success(output)
                    # print(output)
                    await msg.stream_token(f"{output}\n\n")
            else: # node == "agent"
                if values["messages"][0].tool_calls:
                    tool_names = [tool["name"] for tool in values["messages"][0].tool_calls]
                    output = f"Tool(s) Selected: {', '.join(tool_names)}"
                    logger.success(output)
                    # print(output)
                    await msg.stream_token(f"{output}\n\n")
                else:
                    # output = f"\n\n\n**Final Model output**: {values['messages'][-1].content}"
                    output = "\n**Final output**\n"
                    logger.success(output)
                    # print(output)
                    print(values["messages"][-1].content)
                    await msg.stream_token(f"{output}")
                    # await msg.stream_token(values["messages"][-1].content)
                    print("\n\n")
                    
                    # stream messages to the UI
                    if token := values["messages"][-1].content:
                        await msg.stream_token(token)

            # Update messages in state
            # state["messages"].extend(values["messages"])
            # msg = cl.Message(content=values["messages"][-1].content)
            # await message.send()
    

    # Update session state
    cl.user_session.set("state", state)

