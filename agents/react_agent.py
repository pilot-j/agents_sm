import json
import os
import inspect
from colorama import Fore
from dotenv import load_dotenv
from groq import Groq
from fastapi import HTTPException
from pydantic import BaseModel
import httpx
import logging
import asyncio
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
####
##List of functions to control prompt structure and chat inputs.

def completions_create(client, messages: list, model: str) -> str:
    """
    Sends a request to the client's `completions.create` method to interact with the language model.

    Args:
        client (Groq): The Groq client object
        messages (list[dict]): A list of message objects containing chat history for the model.
        model (str): The model to use for generating tool calls and responses.

    Returns:
        str: The content of the model's response.
    """
    response = client.chat.completions.create(messages=messages, model=model)
    return str(response.choices[0].message.content)


def build_prompt_structure(prompt: str, role: str, tag: str = "") -> dict:
    """
    Builds a structured prompt that includes the role and content.

    Args:
        prompt (str): The actual content of the prompt.
        role (str): The role of the speaker (e.g., user, assistant).

    Returns:
        dict: A dictionary representing the structured prompt.
    """
    if tag:
        prompt = f"<{tag}>{prompt}</{tag}>"
    return {"role": role, "content": prompt}


def update_chat_history(history: list, msg: str, role: str):
    """
    Updates the chat history by appending the latest response.

    Args:
        history (list): The list representing the current chat history.
        msg (str): The message to append.
        role (str): The role type (e.g. 'user', 'assistant', 'system')
    """
    history.append(build_prompt_structure(prompt=msg, role=role))


class ChatHistory(list):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        if messages is None:
            messages = []

        super().__init__(messages)
        self.total_length = total_length

    def append(self, msg: str):
        """Add a message to the queue.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(0)
        super().append(msg)


class FixedFirstChatHistory(ChatHistory):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        super().__init__(messages, total_length)

    def append(self, msg: str):
        """Add a message to the queue. The first messaage will always stay fixed.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(1)
        super().append(msg)

import re
from dataclasses import dataclass


@dataclass
class TagContentResult:
    """
    A data class to represent the result of extracting tag content.

    Attributes:
        content (List[str]): A list of strings containing the content found between the specified tags.
        found (bool): A flag indicating whether any content was found for the given tag.
    """

    content: list[str]
    found: bool


def extract_tag_content(text: str, tag: str) -> TagContentResult:
    """
    Extracts all content enclosed by specified tags (e.g., <thought>, <response>, etc.).

    Parameters:
        text (str): The input string containing multiple potential tags.
        tag (str): The name of the tag to search for (e.g., 'thought', 'response').

    Returns:
        dict: A dictionary with the following keys:
            - 'content' (list): A list of strings containing the content found between the specified tags.
            - 'found' (bool): A flag indicating whether any content was found for the given tag.
    """
    # Build the regex pattern dynamically to find multiple occurrences of the tag
    tag_pattern = rf"<{tag}>(.*?)</{tag}>"

    # Use findall to capture all content between the specified tag
    matched_contents = re.findall(tag_pattern, text, re.DOTALL)

    # Return the dataclass instance with the result
    return TagContentResult(
        content=[content.strip() for content in matched_contents],
        found=bool(matched_contents),
    )

import time

from colorama import Fore
from colorama import Style


def fancy_print(message: str) -> None:
    """
    Displays a fancy print message.

    Args:
        message (str): The message to display.
    """
    print(Style.BRIGHT + Fore.CYAN + f"\n{'=' * 50}")
    print(Fore.MAGENTA + f"{message}")
    print(Style.BRIGHT + Fore.CYAN + f"{'=' * 50}\n")
    time.sleep(0.5)


def fancy_step_tracker(step: int, total_steps: int) -> None:
    """
    Displays a fancy step tracker for each iteration of the generation-reflection loop.

    Args:
        step (int): The current step in the loop.
        total_steps (int): The total number of steps in the loop.
    """
    fancy_print(f"STEP {step + 1}/{total_steps}")


import json
from typing import Callable
import inspect

def get_fn_signature(fn: Callable) -> dict:
    """
    Generates the signature for a given function.

    Args:
        fn (Callable): The function whose signature needs to be extracted.

    Returns:
        dict: A dictionary containing the function's name, description,
              and parameter types.
    """
    fn_signature: dict = {
        "name": fn.__name__,
        "description": fn.__doc__,
        "parameters": {"properties": {}},
    }
    schema = {
        k: {"type": v.__name__} for k, v in fn.__annotations__.items() if k != "return"
    }
    fn_signature["parameters"]["properties"] = schema
    return fn_signature


def validate_arguments(tool_call: dict, tool_signature: dict) -> dict:
    """
    Validates and converts arguments in the input dictionary to match the expected types.

    Args:
        tool_call (dict): A dictionary containing the arguments passed to the tool.
        tool_signature (dict): The expected function signature and parameter types.

    Returns:
        dict: The tool call dictionary with the arguments converted to the correct types if necessary.
    """
    properties = tool_signature["parameters"]["properties"]

    # TODO: This is overly simplified but enough for simple Tools.
    type_mapping = {
        "int": int,
        "str": str,
        "bool": bool,
        "float": float,
    }

    for arg_name, arg_value in tool_call["arguments"].items():
        expected_type = properties[arg_name].get("type")

        if not isinstance(arg_value, type_mapping[expected_type]):
            tool_call["arguments"][arg_name] = type_mapping[expected_type](arg_value)

    return tool_call


class Tool:
    """
    A class representing a tool that wraps a callable and its signature.

    Attributes:
        name (str): The name of the tool (function).
        fn (Callable): The function that the tool represents.
        fn_signature (str): JSON string representation of the function's signature.
    """

    def __init__(self, name: str, fn: Callable, fn_signature: str):
        self.name = name
        self.fn = fn
        self.fn_signature = fn_signature

    def __str__(self):
        return self.fn_signature

    async def run(self, **kwargs):
        """
        Executes the tool (function) with provided arguments.

        Args:
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The result of the function call.
        """
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        return self.fn(**kwargs)


def tool(fn: Callable):
    """
    A decorator that wraps a function into a Tool object.

    Args:
        fn (Callable): The function to be wrapped.

    Returns:
        Tool: A Tool object containing the function, its name, and its signature.
    """

    def wrapper():
        fn_signature = get_fn_signature(fn)
        return Tool(
            name=fn_signature.get("name", fn.__name__), fn=fn, fn_signature=json.dumps(fn_signature)
        )

    return wrapper()
####

BASE_SYSTEM_PROMPT = """Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
You are tasked with generating critique and recommendations to the user's generated content.
If there are no critiques, output only the token <OK>. If there is any critique output <NOT OK> and list them.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques. To improve the content if you need any information from the Internet call the search tool.
"""

TOOL_PROMPT = """
##Instruction to use "search" tool

You operate by running a loop with the following steps: Response, Action and Reflection.
You are provided with function signatures within <tools></tools> XML tags.
You may call the search tool to assist with the user query. Don' make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools / actions:

<tools>
%s
</tools>


Additional constraints:

- If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
"""


class ReactAgent:
    """
    A class that represents an agent using the ReAct logic that interacts with tools to process
    user inputs, make decisions, and execute tool calls. The agent can run interactive sessions,
    collect tool signatures, and process multiple tool calls in a given round of interaction.

    Attributes:
        client (Groq): The Groq client used to handle model-based completions.
        model (str): The name of the model used for generating responses. Default is "llama-3.3-70b-versatile".
        tools (list[Tool]): A list of Tool instances available for execution.
        tools_dict (dict): A dictionary mapping tool names to their corresponding Tool instances.
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "meta-llama/llama-4-maverick-17b-128e-instruct",
        system_prompt: str = BASE_SYSTEM_PROMPT,
    ) -> None:
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.system_prompt = system_prompt
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
        """
        Processes each tool call, validates arguments, executes the tools, and collects results.

        Args:
            tool_calls_content (list): List of strings, each representing a tool call in JSON format.

        Returns:
            dict: A dictionary where the keys are tool call IDs and values are the results from the tools.
        """
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

            # Validate and execute the tool call
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

            # Store the result using the tool call ID
            observations[validated_tool_call["id"]] = result

        return observations

    async def run(
        self,
        user_msg: str,
        max_rounds: int = 3,
    ) -> str:
        """
        Executes a user interaction session, where the agent processes user input, generates responses,
        handles tool calls, and updates chat history until a final response is ready or the maximum
        number of rounds is reached.

        Args:
            user_msg (str): The user's input message to start the interaction.
            max_rounds (int, optional): Maximum number of interaction rounds the agent should perform. Default is 10.

        Returns:
            str: The final response generated by the agent after processing user input and any tool calls.
        """
        user_prompt = build_prompt_structure(
            prompt=user_msg, role="user", tag="question"
        )
        if self.tools:
            self.system_prompt += (
                "\n" + TOOL_PROMPT  % self.add_tool_signatures()
            )

        chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=self.system_prompt,
                    role="system",
                ),
                user_prompt,
            ]
        )

        if self.tools:
            # Run the ReAct loop for max_rounds
            for _ in range(max_rounds):

                completion = completions_create(self.client, chat_history, self.model)

                response = extract_tag_content(str(completion), "response")
                reflection = extract_tag_content(str(completion), "reflection")
                if "<OK>" in reflection.content:
                # If no additional suggestions are made, stop the loop
                    print(
                        Fore.RED,
                        "\n\nStop Sequence found. Stopping the reflection loop ... \n\n",
                    )
                    return response.content[0]

                
                tool_calls = extract_tag_content(str(completion), "tool_call")

                update_chat_history(chat_history, completion, "assistant")

                print(Fore.MAGENTA + f"\nThought: {reflection.content}")

                if tool_calls.found:
                    observations = await self.process_tool_calls(tool_calls.content)
                    update_chat_history(
                        chat_history, f'f"Observation: {observations}"', "user"
                        )

        return completions_create(self.client, chat_history, self.model)
    
class SearchResponse(BaseModel):
    query: str
    response: str

SONAR_API_KEY = "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"

async def sonar_search(
        system_prompt: str, 
        search_query: str
    ):
    url = f"https://api.perplexity.ai/chat/completions"
    json_in = "Please output a JSON object containing the following fields: query:, response: "

    model = "sonar"
    context_size = "low"
    max_tokens= 500
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (search_query + json_in)}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "web_search_options": {"search_context_size": context_size},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": SearchResponse.model_json_schema()
            }
        }
    }

    payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {SONAR_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=3000.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"SONAR API error: {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Request failed: {str(e)}"
            )
        

async def main():
    search_tool = tool(sonar_search)
    agent = ReactAgent([search_tool])

    user_msg = "Hi I am Anjum from Kerala. What are best places to eat in Prayagraj?"
    response = await agent.run(user_msg)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
