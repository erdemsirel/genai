import inspect
import json
import os
import re
from typing import Callable


def type_mapping(dtype):
    """Map Python data types to a JSON equivalent."""
    if dtype == float:
        return "number"
    elif dtype == int:
        return "integer"
    elif dtype == str:
        return "string"
    else:
        return "object"


def extract_params(doc_str: str):
    """This function can extract paramter infromation from a docstring.

    When reST-style doc strings are passed to this function, it will return
    a dict with paramaters as keys and what they represent as values.
    """

    # split doc string by newline, skipping empty lines
    params_str = [line for line in doc_str.split("\n") if line.strip()]
    params = {}
    for line in params_str:
        # we only look at lines starting with ':param'
        if line.strip().startswith(":param"):
            param_match = re.findall(r"(?<=:param )\w+", line)
            if param_match:
                param_name = param_match[0]
                desc_match = line.replace(f":param {param_name}:", "").strip()
                # if there is a description, store it
                if desc_match:
                    params[param_name] = desc_match
    return params


def function_to_json(func):
    """Convert a Python function into the JSON format expected by the model."""

    # first we get function name
    function_name = func.__name__
    # then we get the function annotations
    argspec = inspect.getfullargspec(func)
    # get the function docstring
    function_doc = inspect.getdoc(func)
    # parse the docstring to get the description
    function_description = "".join(
        [line for line in function_doc.split("\n") if not line.strip().startswith(":")]
    )
    # parse the docstring to get the descriptions for each parameter in dict format
    param_details = extract_params(function_doc) if function_doc else {}
    # get params
    params = {}
    for param_name in argspec.args:
        # attach parameter types to params
        params[param_name] = {
            "description": param_details.get(param_name) or "",
            "type": type_mapping(argspec.annotations.get(param_name, type(None))),
        }
    # get optional params
    optional_params = inspect.getfullargspec(func).defaults
    # return everything in a dict
    return {
        "name": function_name,
        "description": function_description,
        "parameters": {"type": "object", "properties": params},
        "required": [arg for arg in argspec.args if arg not in (optional_params or [])],
    }


def function_calling_agent(
    client,
    system_prompt: str,
    user_prompt: str,
    *functions: Callable,
    react: bool = False,
    model_name: str = "GPT_4_MODEL_NAME",
    temperature: float = 0.0,
    iterations: int = 3,
) -> str:
    """Returns a response to the given prompts, employing the given functions to help when necessary"""

    tools = [
        {"type": "function", "function": function_to_json(func)} for func in functions
    ]

    available_functions = {func.__name__: func for func in functions}

    if react:
        react_prompt = get_react_prompt(functions)
        system_prompt += react_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=os.environ[model_name],  # need a model version 0613 or higher
        messages=messages,
        tools=tools,
        seed=0,
        temperature=temperature,
    )

    response_message = response.choices[0].message
    messages.append(response_message)

    output = ""
    content = response_message.content
    if content:
        output += f"\n### Model output: \n \n{content}\n"

    tool_calls = response_message.tool_calls

    i = 0  # counter to prevent too many model calls and infinite looping
    while tool_calls is not None:
        if i > iterations:
            raise Exception(
                "Raising exception to prevent too many model calls and infinite looping"
            )
        i += 1

        output += "### Tools called: \n"
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            if function_name in available_functions.keys():
                function_to_use = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = json.dumps(function_to_use(**function_args))

                output += f"\nThe {function_name} function, with the arguments: {function_args}.\n\n"

            else:
                function_response = ask_for_available_tool(
                    function_name, available_functions
                )

                output += f"\nReporting to model {function_name} is not an available function.\n\n"

            messages.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                    "tool_call_id": tool_call.id,
                }
            )  # extend conversation with function respons

        response = client.chat.completions.create(
            model=os.environ[model_name],  # need a model version 0613 or higher
            messages=messages,
            tools=tools,
            seed=0,
            temperature=temperature,
        )  # get a new response using the result of the function call(s)

        response_message = response.choices[0].message
        messages.append(response_message)

        content = response_message.content
        if content:
            output += f"### Model output: \n \n{content}"

        tool_calls = response_message.tool_calls

    return output


def ask_for_available_tool(function_name, available_functions):
    return f"""Was not able to use {function_name}. It is not one of the available tools: {available_functions}.
    Try an alternative approach to answering the user's question.""".strip()


def get_react_prompt(tools):
    tool_names = [tool.__name__ for tool in tools]

    return f"""
Answer the following questions as best you can.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, could be one of the given tools: [{tool_names}]
At this point you may call for an action to be taken, but you must return the Question and Thought as content in your response
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""
