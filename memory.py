from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import get_runtime

from sql_db import SQLDatabase

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///Chinook.db")


@dataclass
class RuntimeContext:
    db: SQLDatabase


@tool
def execute_sql(query: str) -> str:
    """Execute a SQLite command and return results."""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


SYSTEM_PROMPT = """You are a careful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.execute_sql.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT.
"""


agent_without_memory = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=RuntimeContext,
)

question = "This is Frank Harris, What was the the total on my last invoice.?"
steps = []

# First invocation with out memory

for step in agent_without_memory.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
    context=RuntimeContext(db=db),
):
    step["messages"][-1].pretty_print()
    steps.append(step)

## Next invocation with out memory.

print(f"Follow up question invocation", end="\n" * 5)

follow_up_question = "What are the title.?"

for step in agent_without_memory.stream(
    {"messages": [{"role": "user", "content": follow_up_question}]},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


## Here we will add the temporary memory which will keep the context.
## We are calling it temporary as we are not saving to permanent storage.

agent_with_memory = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=RuntimeContext,
    checkpointer=InMemorySaver(),
)

steps = []

for step in agent_with_memory.stream(
    {"messages": [{"role": "user", "content": question}]},
    {"configurable": {"thread_id": "1"}},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
    steps.append(step)


for step in agent_with_memory.stream(
    {"messages": [{"role": "user", "content": follow_up_question}]},
    {"configurable": {"thread_id": "1"}},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
    steps.append(step)
