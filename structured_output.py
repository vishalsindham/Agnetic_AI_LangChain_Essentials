from dotenv import load_dotenv
from langchain.agents import create_agent
from pydantic import BaseModel
from typing_extensions import TypedDict

load_dotenv()
## Langchain supports multiple formats to be passed to the response_format
## here we are using the TypedDict.


class ContactInfo(TypedDict):
    name: str
    email: str
    phone: str


agent_tdso = create_agent(
    model="google_genai:gemini-3.1-flash-lite", response_format=ContactInfo
)

recorded_conversation = """We talked with John Doe. He works over at Example. His number is, let's see, 
five, five, five, one two three, four, five, six seven. Did you get that?
And, his email was john at example.com. He wanted to order 50 boxes of Captain Crunch."""

response = agent_tdso.invoke(
    {"messages": [{"role": "user", "content": recorded_conversation}]}
)

print(response["structured_response"])


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str
    product: str
    quantity: int


agent_bmso = create_agent(
    model="google_genai:gemini-3.1-flash-lite", response_format=ContactInfo
)

response = agent_bmso.invoke(
    {"messages": [{"role": "user", "content": recorded_conversation}]}
)

print(response["structured_response"])
