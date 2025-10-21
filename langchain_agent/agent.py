import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


load_dotenv()
# Now you can access your key
openai_api_key = os.getenv("OPENAI_API_KEY")
print("Key loaded:", openai_api_key is not None)

# 1. Define a Tool
def mult(query: str) -> str:
    """
    Mult two numbers
    :param query:
    :return: string
    """

    try:
        parts = query.split(" and ")
        a, b = float(parts[0].strip()), float(parts[1].strip())
        return str(a * b)
    except Exception as e:
        return f"Error: {e}"


multiply_tool = Tool(
    name="Multiplier",
    func=mult,
    description="Tool to multiply numbers. Input format 'X and Y'"
)

# 2. Set up LLM & prompt
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = hub.pull("hwchase17/react")


# 3. Create agent
tools = [multiply_tool]
agent = create_react_agent(llm,
                           tools,
                           prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # shows reasoning step
    handle_parsing_errors=True
)

# Run agent

if __name__ == "__main__":
    question = "What is 12 by 7.5?"
    print(f"Question: {question}\n")
    response = agent_executor.invoke({"input": question})
    print(f"""Response: {response["output"]}""")



