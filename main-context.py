from __future__ import annotations

import json
from typing import List, Literal, Dict, Any, Optional, Annotated

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from uipath_langchain.chat import UiPathChat
from uipath_langchain.retrievers import ContextGroundingRetriever
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode, tools_condition

# ----------------------------
# 1) Tools
# ----------------------------

retriever = ContextGroundingRetriever(index_name = "Company Invoice Policy",folder_path="Shared")
#print(retriever.invoke("What is the company policy for bank information update?"))
retriever_tool = create_retriever_tool(
    retriever,
    "ContextforInvoicepolicy",
   """
   Use this tool to search the company internal documents for information about invoice processing in case of exceptions.
   Use a meaningful query to load relevant information from the documents. Save the citation for later use.
   """
)

# Simulated ERP lookup (simple, hardcoded)
_FAKE_ERP_DB: Dict[str, Dict[str, Any]] = {
    "PO-1000123": {      
        "po_change_history": [
            "2025-11-02: Vendor bank details updated",
            "2025-11-05: Amount increased from 12,500 to 18,900",
        ],
    },
    "PO-1000456": {      
        "po_change_history": [
            "2025-10-18: Delivery date adjusted +7 days",
        ],
    },
}

@tool
def erp_lookup_tool(po_number: str) -> Dict[str, Any]:
    """
    Look up Purchase Order (PO) details and change history from the ERP system.
    Returns a dictionary containing po_change_history.
    """
    return _FAKE_ERP_DB.get(
        po_number,
        {
          "po_change_history": ["No recent changes found"],
        },
    )

tools = [erp_lookup_tool, retriever_tool]

# ----------------------------
# 2) LLM with Tools
# ----------------------------
llm = UiPathChat(model="gpt-4o-mini-2024-07-18", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ----------------------------
# 3) Inputs & State
# ----------------------------
class GraphState(BaseModel):
    # Input fields
    po_number: str = Field(..., description="PO number, e.g., PO-1000123")
    vendor_name: str = Field(..., description="Vendor name, e.g., Acme Corp")
    exception_text: str = Field(..., description="Exception message from AP/procurement")
    amount: float = Field(..., description="Invoice amount")
    currency: str = Field(..., description="Currency, e.g., USD")
    
    # Message history for the agent loop
    messages: Annotated[List[BaseMessage], add_messages] = []


# ----------------------------
# 4) Output (what agent returns)
# ----------------------------
class GraphOutput(BaseModel):
    po_number: str
    vendor_name: str

    payment_hold: bool
    po_change_history: List[str]

    severity: Literal["Low", "Medium", "High", "Critical"]
    recommended_action: str
    rationale: str
    next_steps: List[str]
    

# ----------------------------
# 5) Nodes
# ----------------------------

async def agent(state: GraphState):
    """
    The main agent node. It decides whether to call tools or provide a final answer.
    """
    # If this is the first turn, provide a system prompt and initial user context
    if not state.messages:
        system_prompt = (
            "You are a Procurement PO Exception Triage Agent.\n"
            "Your goal is to triage a PO exception by checking ERP data and Company Policy.\n"
            "1. ALWAYS check the ERP history for the PO first.\n"
            "2. Then check Company Policy regarding the exception and any ERP findings.\n"
            "3. Finally, output the triage decision in specific JSON format."
        )
        
        user_context = (
            f"Triage this exception:\n"
            f"PO: {state.po_number}\n"
            f"Vendor: {state.vendor_name}\n"
            f"Exception: {state.exception_text}\n"
            f"Amount: {state.amount} {state.currency}"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_context)
        ]
    else:
        messages = state.messages

    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def final_answer(state: GraphState) -> GraphOutput:
    """
    Parses the final LLM response (or history) into the structured GraphOutput.
    This node is called when the agent decides it has enough info.
    """
    # We ask the LLM one last time to format the answer as JSON, based on the conversation history.
    
    formatting_prompt = (
        "Based on the investigation above, produce ONLY valid JSON with this schema:\n"
        "{\n"
        '  "severity": "Low|Medium|High|Critical",\n'
        '  "payment_hold": bool,\n'
        '  "po_change_history": ["string", ...],  // Extract from ERP tool output\n'
        '  "recommended_action": "string",\n'
        '  "rationale": "string",\n'
        '  "next_steps": ["string", ...]\n'
        "}\n"
        "Rules:\n"
        "- If company policy indicates payment hold for certain exceptions, set payment_hold to true else false.\n"
        "- determine recommended action and next steps based on company policy and severity."
    )
    
    # We append this request to the history
    messages = state.messages + [HumanMessage(content=formatting_prompt)]
    
    # We use the raw LLM (no tools) for strict formatting
    resp = await llm.ainvoke(messages)
    
    try:
        # Cleanup json block if present
        content = resp.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        result = json.loads(content.strip())
        
        # Ensure po_change_history is present
        history = result.get("po_change_history", [])
        if not history:
             # Fallback: try to find it in tool outputs if LLM missed it
             for msg in reversed(state.messages):
                 if hasattr(msg, 'tool_calls') and len(msg.tool_calls) > 0 and msg.tool_calls[0]['name'] == 'erp_lookup_tool':
                     pass # Tool calls don't have output. ToolMessages do.
                 if msg.type == 'tool' and msg.name == 'erp_lookup_tool':
                     try:
                        tool_out = json.loads(msg.content)
                        history = tool_out.get("po_change_history", [])
                        break
                     except:
                         pass

        return GraphOutput(
            po_number=state.po_number,
            vendor_name=state.vendor_name,
            payment_hold=result["payment_hold"],
            po_change_history=history,
            severity=result["severity"],
            recommended_action=result["recommended_action"],
            rationale=result["rationale"],
            next_steps=result["next_steps"],
        )
            
    except Exception as e:
        # Fallback error handling
        return GraphOutput(
            po_number=state.po_number,
            vendor_name=state.vendor_name,
            payment_hold=True,
            po_change_history=["Error parsing agent output"],
            severity="Critical",
            recommended_action="Manual Review Required",
            rationale=f"Agent failed to produce structured output: {str(e)}",
            next_steps=["Check agent logs"]
        )


# ----------------------------
# 6) Build graph (must be named `graph`)
# ----------------------------
builder = StateGraph(GraphState, output=GraphOutput)

# Add nodes
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_node("final_answer", final_answer)

# Add edges
builder.add_edge(START, "agent")

# Conditional edge: Agent -> Tools OR Final Answer
def should_continue(state: GraphState) -> Literal["tools", "final_answer"]:
    messages = state.messages
    last_message = messages[-1]
    
    # If the LLM wants to call tools, route to "tools"
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, we are done
    return "final_answer"

builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent") # Loop back to agent
builder.add_edge("final_answer", END)

graph = builder.compile()
