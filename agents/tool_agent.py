import json
import re
import inspect
from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from utils.tool_pattern import Tool, validate_arguments
from utils.completions import build_prompt_structure, ChatHistory, completions_create, update_chat_history
from utils.extract import extract_tag_content
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class ToolAgent:
    """
    The ToolAgent class represents an agent that can interact with a language model and use tools
    to assist with user queries. It generates function calls based on user input, validates arguments,
    and runs the respective tools. If the tool is an asynchronous function use "await" before calling the function.

    Attributes:
        tools (Tool | list[Tool]): A list of tools available to the agent.
        model (str): The model to be used for generating tool calls and responses.
        client (Groq): The Groq client used to interact with the language model.
        tools_dict (dict): A dictionary mapping tool names to their corresponding Tool objects.
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self) -> str:
        """
        Collects the function signatures of all available tools.

        Returns:
            str: A concatenated string of all tool function signatures in JSON format.
        """
        return "".join([tool.fn_signature for tool in self.tools])

    async def process_tool_calls(self, tool_calls_content: list) -> dict:
      observations = {}
      for tool_call_str in tool_calls_content:
          tool_call = json.loads(tool_call_str)
          tool_name = tool_call["name"]
          tool = self.tools_dict[tool_name]

          print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

          validated_tool_call = validate_arguments(
              tool_call, json.loads(tool.fn_signature)
          )
          print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")

          # Check if function is coroutine
          if inspect.iscoroutinefunction(tool.fn):
              result = await tool.fn(**validated_tool_call["arguments"])
          else:
              result = tool.run(**validated_tool_call["arguments"])

          print(Fore.GREEN + f"\nTool result: \n{result}")
          observations[validated_tool_call["id"]] = result

      return observations


    async def run(self, SYSTEM_PROMPT, user_msg: str) -> str:
        user_prompt = build_prompt_structure(prompt=user_msg, role="user")

        tool_chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=SYSTEM_PROMPT % self.add_tool_signatures(),
                    role="system",
                ),
                user_prompt,
            ]
        )
        agent_chat_history = ChatHistory([user_prompt])

        tool_call_response = completions_create(
            self.client, messages=tool_chat_history, model=self.model
        )
        tool_calls = extract_tag_content(str(tool_call_response), "tool_call")

        if tool_calls.found:
            observations = await self.process_tool_calls(tool_calls.content)
            update_chat_history(
                agent_chat_history, f'f"Observation: {observations}"', "user"
            )

        return completions_create(self.client, agent_chat_history, self.model)
    

    