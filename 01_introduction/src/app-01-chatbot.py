import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage, 
    SystemMessage
)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

@cl.on_message
async def main(query: str):
    messages = [
        SystemMessage(content="You are a helpful assistant who speaks \
            with Shakespearean style.  You reply in sonnets."),
        HumanMessage(content=query)
    ]

    try:    
        chat_model = ChatOpenAI(temperature=0.5)
        response = chat_model.predict_messages(messages=messages)
        response = response.content
    except Exception as e:
        response = f"no response: {e}"

    await cl.Message(
        content=response
    ).send()

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello there!  How are you?"
    ).send()