from dotenv import load_dotenv
from sql_db import SQLDatabase
from dataclasses import dataclass
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain.agents.middleware.types import ModelRequest, dynamic_prompt
from langchain.agents import create_agent

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

@dataclass
class RuntimeContext:
    is_employee: True
    db: SQLDatabase

@tool
def execute_sql(query: str) -> str:
    """Execute a SQLite command and return results."""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error : {e}"

## As the agent and LLM's are non-deteministic applications, where the instructions are placed change the context
## which affects how the data is processed and affects the final output.

## We are exploring that instructions can be dynamically updated as per the information we receive from user.

SYSTEM_PROMPT_TEMPLATE = """ You are a careful SQLite analyst.

Rules:
{dynamic_instructions}
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows unless the user explicitly asks otherwise. 
- If the tool returns `Error : `, revise the SQL and try again.
- Prefer explicit column lists; avoid SELECT *.
"""

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    if not request.runtime.context.is_employee:
        table_limits = "Don't access any tables and inform that the user needs to login or reach out to a employee."
    else:
        table_limits=""
    return SYSTEM_PROMPT_TEMPLATE.format(dynamic_instructions=table_limits)


agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    context_schema=RuntimeContext,
    tools=[execute_sql],
    middleware=[dynamic_system_prompt]
)


question = "What is the most costly purchase by Frank Harris. ?"

for step in agent.stream(
    {"messages": [{"role":"user", "content": question}]},
    context=RuntimeContext(is_employee=True, db=db),
    stream_mode="values"
):
    step["messages"][-1].pretty_print()