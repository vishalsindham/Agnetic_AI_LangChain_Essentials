from dotenv import load_dotenv
from langgraph.config import get_stream_writer
from langchain.agents import create_agent

load_dotenv()


agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    system_prompt="You are a comedian, mostly you share dad jokes."
)

# Here we will invoke the graph, which will directly respond with the final output. 
invoke_response = agent.invoke({"messages": "Hi, Share a dad joke without sharing."})
print(invoke_response['messages'][-1].content[0]['text'])


# Here we will stream the output in values mode which will share the output as each step is completed.

for value_response in agent.stream(
    {"messages": [{"role": "user", "content":"Share a dad joke."}]},
    stream_mode="values"
    ):
    value_response['messages'][-1].pretty_print()


# # Here we will stream in messages mode which will stream token by token, which means as soon something is produced it will 
# # be streamed.

for token in agent.stream(
    {"messages" : {"role" : "user", "content":"Share a dad joke with 30 seconds arc"}},
    stream_mode="messages"
):
    if token[0].content:
        print(token[0].content[0]['text'], end="", flush=True)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city : {city}")
    writer(f"Acquired data for city : {city}")
    return f"It's always sunny in {city}"

weather_agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[get_weather],
)

# When we stream with multiple modes, the response will be tuple the mode and response.

for chunk in weather_agent.stream(
    {"messages" : [{"role":"user", "content": "What is the weather in Surat..?"}]},
    stream_mode=["values","custom"]
):
    print(chunk)


    



