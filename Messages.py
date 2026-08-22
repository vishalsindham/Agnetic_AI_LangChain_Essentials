from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

load_dotenv()

comedian_agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    system_prompt="You are a Generative AI comedian.",
)

# There are multiple ways to pass the context to the LLM.
# Here we will create a HumanMessage object and use it to invoke the agent.

human_msg = HumanMessage("Hello, how are you..?")

message_response = comedian_agent.invoke({"messages": [human_msg]})

pprint(message_response["messages"][-1].content[0]["text"])

# The interaction happens with different type of messages objects.
print(type(message_response["messages"][-1]))

# The response we got will have list of messages which are of different types

for msg in message_response["messages"]:

    if msg.type == "ai":
        print(f"{msg.type}: {msg.content[0]['text']}\n")
    else:
        print(f"{msg.type} : {msg.content}")


# We can pass the context and invoke the agent with strings, where the LangChain will retrieve the context and create message for agent.

sports_poet = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    system_prompt="You are a terse sports poet.",
)

str_response = sports_poet.invoke({"messages": "Tell me about baseball."})

print(str_response, end="\n" * 5)

print(str_response["messages"][-1].content, end="\n" * 5)

for msg in str_response["messages"]:
    msg.pretty_print()
    if msg.type == "ai":
        print(f" \n {msg.content[0]['text']}")


# we can pass the context in the dictionary format as well
# Invoke the agent using dictionary

dict_response = sports_poet.invoke(
    {"messages": {"role": "user", "content": "Write a haiku about Kabbadi."}}
)

print(dict_response)

messages = [
    {
        "role": "system",
        "content": "You are a sports poetry expert who completes haikus that have been started.",
    },
    {"role": "user", "content": "Write a haiku about sprinters"},
    {"role": "assistant", "content": "Feet don't fail me..."},
]

dict_response_1 = sports_poet.invoke({"messages": messages})

for msg in dict_response_1["messages"]:
    msg.pretty_print()

print(msg.content[0]["text"])

# We have tools in the workflow, tools respond in Tool Message object which are a type of message object.


@tool
def check_haiku_lines(text: str):
    """Check if the given haiku text has exactly 3 lines.

    Returns None if it's correct, otherwise an error message.
    """

    # Split the text into lines, ignoring leading/trailing spaces.
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 3:
        return f"Checking haiku, it has {len(lines)} lines. A haiku must have exactly 3 lines."
    return f"Correct, this haiku has 3 lines."


haiku_agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[check_haiku_lines],
    system_prompt="You are a sports poet who only writes haiku. You always check your work.",
)

haiku_response = haiku_agent.invoke({"messages": "Please write me a poem."})

print(haiku_response["messages"][-1].content[0]["text"])

print(len(haiku_response["messages"]))

for i, msg in enumerate(haiku_response["messages"]):
    msg.pretty_print()

# system_prompt is the instruction which inform's model how to behave.

paragraph_agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    system_prompt="You write a short 300 words paragraph for a given topic or word",
)

paragraph_response = paragraph_agent.invoke({"messages": "Steve Jobs"})

for msg in paragraph_response["messages"]:
    msg.pretty_print()
