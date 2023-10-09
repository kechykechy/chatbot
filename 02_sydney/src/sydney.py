import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

MODEL = "gpt-3.5-turbo"
api_key, org_id = sk.openai_settings_from_dot_env()

kernel = sk.Kernel()
kernel.add_chat_service(
    "Sydney Chatbot",
    OpenAIChatCompletion(MODEL, api_key=api_key, org_id=org_id)
)
context = kernel.create_new_context()

BOT_NAME  = "Sydney"
USER_NAME = "   You"

chat_prompt = """
{{$user_input}}
"""

chat_function = kernel.create_semantic_function(
    chat_prompt,
    BOT_NAME,
    max_tokens=2000,
    temperature=0.8
)

async def chat(user_input):
    context["user_input"] = user_input
    response = await chat_function.invoke_async(context=context)
    print(f"{BOT_NAME} > {response}")

async def chat_session():
    user_input = input(f"{BOT_NAME} > Hello there!  How are you?\n{USER_NAME} > ")
    while user_input.strip().casefold() != 'quit':
        await chat(user_input)
        user_input = input(f"{USER_NAME} > ")
    print("Come back soon ... talk to you later!")

if __name__ == '__main__':
    asyncio.run(
        chat_session()
    )