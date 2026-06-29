from dataclasses import dataclass
from typing_extensions import TypedDict
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import Runnable
from langgraph.store.memory import InMemoryStore

# 1. Define the Runtime Context
@dataclass
class Context:
    user_id: str

# 2. Define structured input for saving data
class UserInfo(TypedDict):
    name: str

# 3. Create a single, shared Memory Store
shared_store = InMemoryStore()

# 4. Define the Tools
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info to the store."""
    assert runtime.store is not None
    user_id = runtime.context.user_id
    
    # Store the extracted data into the namespace ("users",)
    runtime.store.put(("users",), user_id, dict(user_info))
    return f"Successfully saved user info for {user_id}."

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info from the store."""
    assert runtime.store is not None
    user_id = runtime.context.user_id
    
    # Retrieve data from the same namespace
    user_info = runtime.store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# 5. Initialize the Agent with both tools
agent: Runnable = create_agent(
    model="ollama:north-mini-code-1.0",
    tools=[save_user_info, get_user_info],
    store=shared_store,  # Inject the shared store
    context_schema=Context,
)

# ==========================================
# STEP 1: Extract and Save Name from Message
# ==========================================
print("--- STEP 1: Agent extracts 'Alice' and saves it ---")
save_result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Alice, please remember that."}]},
    context=Context(user_id="user_abc"),
)
print(f"Agent Output: {save_result}\n")

# ==========================================
# STEP 2: Retrieve Name from Store via Agent
# ==========================================
print("--- STEP 2: Agent retrieves the name from the store ---")
retrieve_result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    context=Context(user_id="user_abc"),
)
print(f"Agent Output: {retrieve_result}\n")

# ==========================================
# STEP 3: Verification (Direct Access)
# ==========================================
print("--- STEP 3: Double checking the store directly ---")
direct_check = shared_store.get(("users",), "user_abc")
print(f"Direct Store Value: {direct_check.value if direct_check else 'Empty'}")
