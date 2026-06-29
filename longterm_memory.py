import uuid
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import IndexConfig

# ==========================================
# 1. DEFINE RUNTIME DEPENDENCIES (NO HARDCODING)
# ==========================================

# Define the Context data structure. 
# This dictates what details MUST be attached whenever an agent run executes.
@dataclass
class UserContext:
    user_id: str
    application_context: str = "coding_assistant"


# ==========================================
# 2. CREATE TRULY DYNAMIC TOOLS
# ==========================================

@tool
def save_user_preference(preference: str, runtime: ToolRuntime[UserContext]) -> str:
    """
    Call this tool when the user explicitly states a long-term preference, rule, 
    or fact about themselves (e.g., programming language choices, brief explanations).
    """
    # 🌟 DYNAMIC READ: Absolutely no hardcoded user strings.
    # The runtime pulls whichever user is actively running this thread.
    user_id = runtime.context.user_id
    app_context = runtime.context.application_context
    namespace = (user_id, app_context)
    
    # Safely write to long-term memory
    existing_item = runtime.store.get(namespace, "user_preferences")
    current_rules = existing_item.value.get("rules", []) if existing_item else []
    
    if preference not in current_rules:
        current_rules.append(preference)
        
    runtime.store.put(namespace, "user_preferences", {"rules": current_rules})
    return f"Successfully learned and saved preference for user '{user_id}'"


# ==========================================
# 3. CONSTRUCT THE ENGINE
# ==========================================

model = ChatOpenAI(model="gpt-4o", temperature=0.2)
def mock_embed(text: str) -> list[float]: return [0.1, 0.2]

# Shared storage instances (would normally map to Postgres / Redis in production)
global_store = InMemoryStore(index=IndexConfig(embed=mock_embed, dims=2))
global_checkpointer = InMemorySaver()

# Set up the agent blueprint
agent = create_agent(
    model=model,
    tools=[save_user_preference], 
    store=global_store,      
    checkpointer=global_checkpointer,
    context_schema=UserContext  # ◄ Forces the agent runtime to accommodate this schema
)


# ==========================================
# 4. EXECUTION SIMULATION (PASSING DATA DYNAMICALLY)
# ==========================================

def run_chat_pipeline(incoming_user_id: str, message: str, incoming_thread_id: str = None):
    """
    Acts like your production API endpoint. It accepts parameters externally 
    and handles thread/context setup entirely dynamically.
    """
    # Dynamic Thread Generation: If none is passed, it assumes a "New Chat" window
    active_thread_id = incoming_thread_id or f"thread_{uuid.uuid4().hex[:8]}"
    
    # Dynamic Config Mapping
    config = {"configurable": {"thread_id": active_thread_id}}
    
    # Dynamic Context Injection
    runtime_context = UserContext(user_id=incoming_user_id)
    
    # Fire Agent Execution
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        context=runtime_context
    )
    
    return {
        "reply": response["messages"][-1].content,
        "thread_id": active_thread_id
    }


# --- PRODUCTION SIMULATION RUNS ---

print("=== SCENARIO 1: Alice opens a brand new chat window ===")
# She triggers a conversation with no thread_id passed.
res1 = run_chat_pipeline(
    incoming_user_id="alice_dev_99", 
    message="Hey, write code blocks in Python from now on."
)
print(f"Generated Thread ID: {res1['thread_id']}")
print(f"AI Response: {res1['reply']}\n")


print("=== SCENARIO 2: Alice sends a message in the SAME chat window ===")
# The frontend sends back the generated Thread ID to preserve short-term chat window state
res2 = run_chat_pipeline(
    incoming_user_id="alice_dev_99", 
    message="Can you give me a template to read an array?", 
    incoming_thread_id=res1['thread_id'] # Injected dynamically
)
print(f"AI Response (Short Term Context Maintained):\n{res2['reply']}\n")


print("=== SCENARIO 3: Bob logs in entirely separately ===")
# Bob has a completely different ID and a clean slate thread.
res3 = run_chat_pipeline(
    incoming_user_id="bob_manager_01", 
    message="Can you give me a template to read an array?"
)
print(f"Generated Thread ID: {res3['thread_id']}")
print(f"AI Response to Bob: {res3['reply']}\n")


print("=== SCENARIO 4: Alice returns next week in a NEW thread ===")
# Alice starts a brand new conversation channel (None thread_id). 
# Let's check if her long-term rules persist across system boundaries.
res4 = run_chat_pipeline(
    incoming_user_id="alice_dev_99", 
    message="Can you give me a basic array reading template?"
)
print(f"Generated Thread ID: {res4['thread_id']}")
print(f"AI Response (Long Term Store Retrieved):\n{res4['reply']}\n")
