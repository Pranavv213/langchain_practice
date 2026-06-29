from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

app = FastAPI(title="Thread Tracking Memory API")

# =====================================================================
# 1. Globals (Shared Storages & Active Session Mappings)
# =====================================================================
shared_store = InMemoryStore()      # Global database cross-sessions
checkpointer = MemorySaver()        # Chat transcript memory per thread

# This lookup table maps thread_id -> user_id globally so tools can
# find the user without requiring the LLM to understand injection schemas.
thread_to_user_map: Dict[str, str] = {}

# =====================================================================
# 2. Simplified Tools (100% safe for smaller models like Llama 3.2 3b)
# =====================================================================
@tool
def get_store(config: RunnableConfig) -> str:
    """Retrieve your permanent profile records and application preferences."""
    # Pull the thread_id from the graph's execution context
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = thread_to_user_map.get(thread_id)
    
    if not user_id:
        return "Error: No active user session context found for this thread."

    # Directly query the global shared store
    data = shared_store.get(("users",), user_id)
    if data and data.value:
        return f"Retrieved records for {user_id}: {data.value}"
    return f"No long-term context data recorded yet for user: {user_id}."

@tool
def put_store(preferences: str, config: RunnableConfig) -> str:
    """Save or update your permanent application preferences. 
    Use this when the user explicitly states a preference they want remembered long-term.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = thread_to_user_map.get(thread_id)
    
    if not user_id:
        return "Error: No active user session context found for this thread."

    # Fetch snapshot of existing row data to merge and protect other properties
    existing = shared_store.get(("users",), user_id)
    current_value = existing.value if existing else {}
    
    current_value["preferences"] = preferences
    
    shared_store.put(("users",), user_id, current_value)
    return f"Successfully updated your long-term preferences."

# =====================================================================
# 3. Agent Architecture Definition
# =====================================================================
agent_graph = create_agent(
    model="ollama:llama3.2:3b",  
    tools=[get_store, put_store],
    checkpointer=checkpointer,
    store=shared_store
)

# =====================================================================
# 4. Frontend Data Delivery Schemas
# =====================================================================
class ConfigureRequest(BaseModel):
    user_id: str
    thread_id: str
    user_name: Optional[str] = None
    preferences: Optional[str] = None

class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str

# =====================================================================
# 5. FastAPI Operational Core
# =====================================================================

@app.post("/configure")
async def configure_session(payload: ConfigureRequest):
    """
    Initial configuration endpoint. Connects a thread_id to a user_id
    and seeds the long-term memory store.
    """
    try:
        # Save the relationship in our server memory map
        thread_to_user_map[payload.thread_id] = payload.user_id
        
        if payload.user_name or payload.preferences:
            data_payload = {
                "name": payload.user_name,
                "preferences": payload.preferences
            }
            shared_store.put(("users",), payload.user_id, data_payload)
            return {"status": "configured", "message": f"Pre-seeded long-term data for {payload.user_id}."}
            
        return {"status": "configured", "message": f"Bound thread '{payload.thread_id}' to user '{payload.user_id}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def process_chat(payload: ChatRequest):
    """
    Accepts incoming messages. Keeps conversation histories isolated using thread_id.
    """
    # Ensure this thread has been mapped to a user via /configure first
    if payload.thread_id not in thread_to_user_map:
        thread_to_user_map[payload.thread_id] = payload.user_id

    config = {
        "configurable": {
            "thread_id": payload.thread_id
        }
    }
    
    try:
        response = agent_graph.invoke(
            {"messages": [{"role": "user", "content": payload.message}]},
            config=config
        )
        
        bot_reply = response["messages"][-1].content
        return {"response": bot_reply}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
