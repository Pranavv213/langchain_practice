from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool  # <-- Import the tool decorator

# --- 1. DEFINE YOUR TOOLS FIRST ---
@tool
def customer_service_tool(query: str) -> str:
    """Look up customer account information and help tickets."""
    return f"Processing customer service request: {query}"

@tool
def email_tool(recipient: str, body: str) -> str:
    """Send an email to a recipient with a message body."""
    return f"Email successfully sent to {recipient}."

# --- 2. CREATE THE AGENT WITH MIDDLEWARE ---
agent = create_agent(
    model="ollama:llama3.2:3b", 
     # Adjusted to use the correct ollama provider string format
    tools=[customer_service_tool, email_tool],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

# --- 3. INVOKE THE AGENT ---
result = agent.invoke({
    "messages": [{"role": "user", "content": "My email is john.doe@example.com and card is 5105-1051-0510-5100"}]
})

final_message = result["messages"][-1]
print(final_message.content)
