import asyncio
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

uvx = shutil.which("uvx")
if uvx is None:
    raise RuntimeError("uvx was not found. Install uv from https://docs.astral.sh/uv/.")

mcp_client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": uvx,
            "args": [
                "--cache-dir",
                str(Path(tempfile.gettempdir()) / "mcp-uv-cache"),
                "--with",
                "mcp<2",
                "mcp-server-time",
            ],
        }
    },
)


async def main():

    mcp_tools = await mcp_client.get_tools()
    print(
        f"Number of tools avaliable {len(mcp_tools)}, which are {[ t.name for t in mcp_tools]} "
    )

    agent_with_mcp = create_agent(
        model="google_genai:gemini-3.1-flash-lite",
        tools=mcp_tools,
        system_prompt="You are a helpful assistant",
    )

    city = input("Which city do you want to know..? ")

    response = await agent_with_mcp.ainvoke(
        {
            "messages": [
                {"role": "user", "content": f"What's the time in {city} right now..?"}
            ]
        }
    )

    for msg in response["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
