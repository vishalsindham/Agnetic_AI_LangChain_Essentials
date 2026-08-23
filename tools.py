"""
We exploring how agents interact with tools. Tools with detailed description let the agent know 
which operations it can perform and it will decide when to make a particular tool call.
If the description is limited in defining what it can do the agent will not be able to call the tool even if the 
tool can perform the operation.
"""
from dotenv import load_dotenv
from langchain.tools import tool
from typing import Literal
from langchain.agents import create_agent

load_dotenv()

# @tool
# def real_number_calculator(
#     a:float, b:float, operation: Literal["add", "subtract", "multiply", "divide"]
# ) -> float:
#     """Perform basic arithmetic operations on two real numbers."""
#     print("Invoking calculator tool")
#     # Perform the specified operation
#     if operation == "add":
#         return a + b
#     elif operation == "subtract":
#         return a - b
#     elif operation == "multiply":
#         return a * b
#     elif operation == "divide":
#         if b == 0:
#             raise ValueError("Division by zero if not allowed.")
#         return a/b
#     else:
#         raise ValueError(f"Invalid operation : {operation}.")

# agent = create_agent(
#     model="google_genai:gemini-3.1-flash-lite",
#     tools=[real_number_calculator],
#     system_prompt="You are a helpful assistant."
# )

# # Invoking the agent using real numbers, which makes the tool call.
# response = agent.invoke(
#     {"messages" : [{"role":"user", "content":"what is 3.125 * 4.1234..?"}]}
# )

# print(response["messages"][-1].content[0]['text'])

# Invoking the agent with word numbers to check if the agent makes a tool call or not. Tool description mentions
# perform basic arithmetic operations on rel numbers. 
# response = agent.invoke({"messages" : [{"role":"user", "content": "What is three multiplied by four."}]})

# print(response["messages"][-1].content[0]['text'])


## If there is a tool which is shared standard tool and needs a agent-specific instructions.
## overwrite the instructions using the decorator arguments.

@tool(
    "calculator",
    parse_docstring=True,
    description=(
        "Perform basic arithmetic operations on two real numbers."
        "Use this whenever you have operations on any numbers, even if they are integers."
    )
)
def detailed_real_number_calculator(
    a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"]
) -> float:
    """Perform basic arithmetic operations on two real numbers.

    Args:
        a (float): The first number.
        b (float): The second number.
        operation (Literal["add", "subtract", "mulitply", "divide"]):
            The arithmetic operation to perform.

            -`"add"` : Returns the sum of `a` and `b`.
            -`"subtract"` : Returns the result of `a - b`.
            - `"multiply"` : Returns the product of `a` and `b`.
            - `"divide"`  : Returns the result of `a/b`. Raises an error if `b` is zero.
    
    Returns:
        float: The numerical result of the specified  operation.
    
    Raises:
        ValueError: If an invalid operation is provided or division by zero is attempted.
    """
    print("Invoking detailed calculator")
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero if no allowed.")
        return a/b
    else:
        raise ValueError(f"Invalid operation: {operation}")

detailed_agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[detailed_real_number_calculator],
    system_prompt="You are a helpful assistant."
)


detailed_response = detailed_agent.invoke({"messages" : [{"role" : "user", "content" : "what is 5.0 * 5.0"}]})

print(detailed_response["messages"][-1].content[0]['text'])



