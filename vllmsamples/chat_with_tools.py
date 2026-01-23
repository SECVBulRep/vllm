# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa
import json
import random
import string

from vllm import LLM
from vllm.sampling_params import SamplingParams


import re
import json

def parse_tool_calls(text: str):
    # 1) Try <tool_call> ... </tool_call>
    m = re.search(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", text)
    if m:
        return [json.loads(m.group(1))]  # make it a list to match your existing code

    # 2) Try pure JSON object
    text = text.strip()
    if text.startswith("{"):
        return [json.loads(text)]

    # 3) Try JSON array
    if text.startswith("["):
        return json.loads(text)

    raise ValueError(f"Can't parse tool calls from: {text!r}")


# This script is an offline demo for function calling
#
# If you want to run a server/client setup, please follow this code:
#
# - Server:
#
# ```bash
# vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tokenizer-mode mistral --load-format mistral --config-format mistral
# ```
#
# - Client:
#
# ```bash
# curl --location 'http://<your-node-url>:8000/v1/chat/completions' \
# --header 'Content-Type: application/json' \
# --header 'Authorization: Bearer token' \
# --data '{
#     "model": "mistralai/Mistral-7B-Instruct-v0.3"
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#             {"type" : "text", "text": "Describe this image in detail please."},
#             {"type": "image_url", "image_url": {"url": "https://s3.amazonaws.com/cms.ipressroom.com/338/files/201808/5b894ee1a138352221103195_A680%7Ejogging-edit/A680%7Ejogging-edit_hero.jpg"}},
#             {"type" : "text", "text": "and this one as well. Answer in French."},
#             {"type": "image_url", "image_url": {"url": "https://www.wolframcloud.com/obj/resourcesystem/images/a0e/a0ee3983-46c6-4c92-b85d-059044639928/6af8cfb971db031b.png"}}
#         ]
#       }
#     ]
#   }'
# ```
#
# Usage:
#     python demo.py simple
#     python demo.py advanced

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# or switch to "mistralai/Mistral-Nemo-Instruct-2407"
# or "mistralai/Mistral-Large-Instruct-2407"
# or any other mistral model with function calling ability

sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)
llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
)



def generate_random_id(length=9):
    characters = string.ascii_letters + string.digits
    random_id = "".join(random.choice(characters) for _ in range(length))
    return random_id


# simulate an API that can be called
def get_current_weather(city: str, state: str, unit: "str"):
    return (
        f"The weather in {city}, {state} is 85 degrees {unit}. It is "
        "partly cloudly, with highs in the 90's."
    )


tool_functions = {"get_current_weather": get_current_weather}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

messages = [
    {
        "role": "system",
        "content": (
            "You are a function-calling assistant. "
            "If a tool is needed, respond with ONLY a JSON array of tool calls. "
            "No extra text. JSON only."
        ),
    },
    {
        "role": "user",
        "content": "Can you tell me what the temperature will be in Dallas, TX in fahrenheit?",
    },
]


outputs = llm.chat(messages, sampling_params=sampling_params, tools=tools)
output = outputs[0].outputs[0].text.strip()

# append the assistant message
messages.append(
    {
        "role": "assistant",
        "content": output,
    }
)

# let's now actually parse and execute the model's output simulating an API call by using the
# above defined function

print("RAW OUTPUT >>>", repr(output))

tool_calls = parse_tool_calls(output)
tool_answers = [
    tool_functions[call["name"]](**call["arguments"]) for call in tool_calls
]

# append the answer as a tool message and let the LLM give you an answer
messages.append(
    {
        "role": "tool",
        "content": "\n\n".join(tool_answers),
        "tool_call_id": generate_random_id(),
    }
)

outputs = llm.chat(messages, sampling_params, tools=tools)

print(outputs[0].outputs[0].text.strip())
# yields
#   'The weather in Dallas, TX is 85 degrees Fahrenheit. '
#   'It is partly cloudly, with highs in the 90's.'