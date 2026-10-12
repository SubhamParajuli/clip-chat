from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph,START,END,MessagesState
from sentence_transformers import SentenceTransformer
from langgraph.prebuilt import ToolNode,tools_condition
from src.db_client import get_chroma_client,get_video_collection

text_embedder=SentenceTransformer("all-MiniLM-L6-v2")
clip_model=SentenceTransformer("clip-ViT-B-32")
collection=get_video_collection(get_chroma_client())

@tool
def search_video_database(query:str)->str:
    """Searches the multi-modal video database for visual description or dialogue."""
    query_vector=clip_model.encode(query).tolist()
    results=collection.query(query_embeddings=[query_vector],n_results=4)

    if not results["documents"][0]:
        return "No matching timestamps found in the video database."

    formatted_output=[]
    for doc,meta in zip(results["documents"][0],results["metadatas"][0]):
        formatted_output.append(
        f"- Video: {meta['video_id']} | Type: {meta['type'].upper()} | "
        f"Timestamp: {meta['start_time']}s - {meta['end_time']}s | Details: {doc}"
    )
    return "\n".join(formatted_output)


tools=[search_video_database]

llm=ChatOllama(model="llama3",temperature=0)
llm_with_tools=llm.bind_tools(tools)

##make it async

async def call_model(state:MessagesState):
    system_prompt=SystemMessage(
        content=("You are an advanced agentic video index assistant. Your job is to"
          " find exact timestamps when users ask about events, objects, or"
          " speech in videos. Use the search_video_database tool to discover"
          " matches and report precise time intervals.")
    )
    response=await llm_with_tools.ainvoke([system_prompt]+state["messages"])
    return{"messages":[response]}

workflow=StateGraph(MessagesState)
workflow.add_node("agent",call_model)
workflow.add_node("tools",ToolNode(tools))

workflow.add_edge(START,"agent")
workflow.add_conditional_edges("agent",tools_condition)
workflow.add_edge("tools","agent")

app=workflow.compile()

png_bytes=app.get_graph().draw_mermaid_png()
output_path="agent_grpah.png"
with open(output_path,"wb") as f:
    f.write(png_bytes)

print(f"Agent workflow graph successfully saved as {output_path}!")

    