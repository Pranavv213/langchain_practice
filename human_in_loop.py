import os
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.types import Command  # Crucial import for resuming middleware interrupts

# ==========================================
# 1. TOOL DEFINITIONS
# ==========================================

@tool
def write_file(filename: str, content: str) -> str:
    """
    Writes content to a specified local file. 
    Use this when you need to save code, logs, scripts, or textual data.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote content to '{filename}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def execute_sql(query: str) -> str:
    """
    Executes a read/write SQL query against the database and returns the result.
    Use this for modifying tables, deleting data, or inserting rows.
    """
    print(f"\n[SYSTEM LOG] Executing write/modify SQL: {query}")
    return f"SQL execution completed successfully for query: '{query}'."


@tool
def read_data(query: str) -> str:
    """
    Executes a safe read-only query or reads analytical information.
    Use this for SELECT statements or querying structural information.
    """
    print(f"\n[SYSTEM LOG] Executing safe read SQL: {query}")
    return f"Retrieved data records for query: '{query}'."


# ==========================================
# 2. AGENT & MIDDLEWARE CONFIGURATION
# ==========================================

agent = create_agent(
    model="ollama:llama3.2:3b",  
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # All decisions allowed
                "write_file": True,  
                
                # Critical operation: Only approve or reject allowed
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  
                
                # Safe read operation: No human confirmation needed
                "read_data": False, 
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)


# ==========================================
# 3. INTERACTIVE TERMINAL EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Agent compiled successfully. Starting interactive session...")
    
    # Configuration thread ID for state management
    config = {"configurable": {"thread_id": "interactive_session_789"}}
    
    # 1. Ask the user for input in the terminal
    user_prompt = input("\nEnter your command for the agent (e.g., 'Delete the user table using execute_sql'): ")
    initial_input = {"messages": [("user", user_prompt)]}

    print("\nProcessing request...")
    agent.invoke(initial_input, config)

    # 2. Check if the agent's graph execution was paused by the middleware
    state = agent.get_state(config)
    
    if state.next:
        print("\n" + "!" * 50)
        print(f"🛑 AGENT PAUSED: Human-in-the-Loop Intercept triggered on node: {state.next}")
        print(f"Details: {state.tasks[0].interrupts[0].value}")
        print("!" * 50 + "\n")
        
        # 3. Capture your actual human live decision in the console
        decision = input("What should the agent do? Type (approve / reject): ").strip().lower()
        
        if decision == "approve":
            print("\nResuming execution with approval...")
            
            # Formulate the explicit decision block requested by the middleware
            approval_command = Command(
                resume={
                    "decisions": [
                        {
                            "type": "approve"
                        }
                    ]
                }
            )
            final_result = agent.invoke(approval_command, config)
            print(f"\nFinal Agent Response:\n{final_result['messages'][-1].content}")
        
        elif decision == "reject":
            print("\nAction rejected. Aborting execution stream...")
            
            # Send the explicit rejection routing schema to the middleware
            rejection_command = Command(
                resume={
                    "decisions": [
                        {
                            "type": "reject",
                            "message": "User rejected this action due to safety risks."
                        }
                    ]
                }
            )
            final_result = agent.invoke(rejection_command, config)
            print(f"\nFinal Agent Response:\n{final_result['messages'][-1].content}")
        
        else:
            print(f"\nUnknown action '{decision}'. Session terminated safely without tool execution.")
            
    else:
        # If the tool used was 'read_data', it bypasses the interrupt step and finishes immediately
        final_state = agent.get_state(config)
        print(f"\nFinal Agent Response:\n{final_state.values['messages'][-1].content}")
