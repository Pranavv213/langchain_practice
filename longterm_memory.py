from dataclasses import dataclass
from typing_extensions import TypedDict
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import Runnable
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage

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
    """Save user info to the store. Use this whenever the user shares personal details like their name."""
    assert runtime.store is not None
    user_id = runtime.context.user_id
    
    runtime.store.put(("users",), user_id, dict(user_info))
    return f"Successfully saved user info for {user_id}."

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info from the store to recall facts about the user."""
    assert runtime.store is not None
    user_id = runtime.context.user_id
    
    user_info = runtime.store.get(("users",), user_id)
    return str(user_info.value) if user_info else "No info found for this user."

# 5. Initialize the Agent
# Note: Ensure the system prompt encourages checking the store if it needs information.
agent: Runnable = create_agent(
    model="ollama:llama3.2:3b",  # Use the correct Ollama model string
    tools=[save_user_info, get_user_info],
    store=shared_store,
    context_schema=Context,
)

# ==========================================
# 6. Interactive Chatbot Loop
# ==========================================
def start_chatbot():
    print("==================================================")
    print("🤖 Chatbot initialized with Long-Term Memory!")
    print("Type 'exit' or 'quit' to end the chat.")
    print("==================================================\n")
    
    # Mocking a specific user session
    current_user = "user_123"
    context = Context(user_id=current_user)
    
    # This list maintains the short-term conversation history for the current session
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
            
        if not user_input.strip():
            continue

        # Append the new user message to the ongoing conversation history
        chat_history.append({"role": "user", "content": user_input})

        try:
            # Invoke the agent with the entire conversation history and user context
            response = agent.invoke(
                {"messages": chat_history},
                context=context
            )
            
            # The agent returns a message object or dictionary containing the response
            # Depending on your precise LangChain version, handle string extraction:
            output_text = response["messages"][-1].content if isinstance(response, dict) else response
            
            print(f"Bot: {output_text}\n")
            
            # Append the agent's response to the history so it remembers the immediate context
            chat_history.append({"role": "assistant", "content": output_text})
            
        except Exception as e:
            print(f"System Error: {e}\n")

if __name__ == "__main__":
    start_chatbot()
