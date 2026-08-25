from dotenv import load_dotenv
from sql_db import SQLDatabase
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

db =  SQLDatabase.from_uri("sqlite:///Chinook.db")

@dataclass
class RuntimeContext:
    db: SQLDatabase

@tool
def execute_sql(query: str) -> str:
    """Execute the SQLite command and return results"""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error : {e}"

SYSTEM_PROMPT = """You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
- If the database is offline, ask user to try again later without further comment.
"""

agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
    context_schema=RuntimeContext,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"execute_sql" : {"allowed_decisions" : ["approve", "reject"]}},
        ),
    ],
)



question = "What are the names of all the employes..?"

config  = {"configurable" : {"thread_id" : "1"}}

response = agent.invoke(
    {"messages" : [{"role": "user", "content": question}]},
    config=config,
    context=RuntimeContext(db=db)
)

## When we are rejecting the request 

if "__interrupt__" in response:
    description = response["__interrupt__"][-1].value['action_requests'][-1]['description']
    print(f"Interrupt: {description}")

    response = agent.invoke(
        Command(
            resume={
                "decisions" : [{"type" : "reject", "message":"the database is offline."}]
            }
        ),
        config=config,
        context=RuntimeContext(db=db)
    )
print(response["messages"][-1].content)



config  = {"configurable" : {"thread_id" : "2"}}


response = agent.invoke(
    {"messages" : [{"role": "user", "content": question}]},
    config=config,
    context=RuntimeContext(db=db)
)


## Due to the same query will be executed multiple times, while loop is used to execute till we get response we require.


while "__interrupt__" in response:
    description = response["__interrupt__"][-1].value['action_requests'][-1]['description']
    print(f"Interrupt: {description}")

    response = agent.invoke(
        Command(
            resume={
                "decisions" : [{"type" : "approve"}]
            }
        ),
        config=config,
        context=RuntimeContext(db=db)
    )

for msg in response["messages"]:
    msg.pretty_print()
