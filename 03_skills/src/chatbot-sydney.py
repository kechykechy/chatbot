import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

BOT_NAME  = "Sydney"

import chainlit as cl
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.planning.basic_planner import BasicPlanner

CHAT_MODEL = "gpt-3.5-turbo"
COMPLETION_MODEL = "gpt-3.5-turbo-instruct"
api_key = os.environ.get("OPENAI_API_KEY")

kernel = sk.Kernel()
kernel.add_chat_service(
    "Sydney Chatbot",
    OpenAIChatCompletion(CHAT_MODEL, api_key=api_key)
)
context = kernel.create_new_context()

shakespeare_prompt = """
{{$input}}

Rewrite the above as a sonnet in the style of William Shakespeare.
"""

shakespeare_function = kernel.create_semantic_function(
    prompt_template=shakespeare_prompt,
    function_name="shakespeare",
    skill_name="ShakespeareSonnetSkill",
    description="create a love sonnet in the style of William Shakespeare",
    max_tokens=1000,
    temperature=0.7
)

frost_prompt = """
{{$input}}

Rewrite the above as a poem in the style of Robert Frost.
"""

frost_function = kernel.create_semantic_function(
    prompt_template=frost_prompt,
    function_name="frost",
    skill_name="FrostPoemSkill",
    description="create a poem about making decision in the style of Robert Frost",
    max_tokens=1000,
    temperature=0.7
)

@cl.on_message
async def main(query: str):
    response = ""
    planner = BasicPlanner()
    try:
        plan = await planner.create_plan_async(query, kernel)
        print(f"plan: {plan.generated_plan}")
        response = await planner.execute_plan_async(plan, kernel)
    except Exception as e:
        response = f"Cannot proceed: {e}"
    await cl.Message(
        content = response
    ).send()

@cl.on_chat_start
async def start():
    await cl.Message(
        content=f"Hello there!  I am {BOT_NAME}.  How can I help you?"
    ).send()