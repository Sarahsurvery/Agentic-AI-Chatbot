import os

from dotenv import load_dotenv #(environment variable loading klye)
from typing import cast  # type choosing
import chainlit as cl
#custom classes
#agent AI assisstant frame (jsme sara kam hota hai)
#runner conversation handling
#asyncopenai (gemini or openai client)
#openaichatcompletionmodel (model batata hai)
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.run import RunConfig #configuration object hai jo ai model control krta hai


load_dotenv ()
gemini_api_key = os.getenv ("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@cl.on_chat_start
async def start():
    
    #Reference: https://ai.google.dev/gemini-api/docs/openai
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    # set_tracing_disabled(True)
    model = OpenAIChatCompletionsModel(
        model='gemini-2.5-flash',
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    cl.user_session.set("chat history", [])
    cl.user_session.set("config ", config)

    agent: Agent = Agent (name="Personal assistant", instructions="You are very helpful", model= model)

    cl.user_session.set("agent", agent)
    await cl.Message(content= "Welcome to Panaversity AI Assistant! How can i help?").send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="Thinking...")
    await msg.send()

    agent : Agent = cast(Agent, cl.user_session.get ("agent"))
    config : RunConfig = cast(RunConfig, cl.user_session.get ("config"))
    history = cl.user_session.get ("chat history") or []
    history.append({"role": "user", "content": message.content})

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")

        result = Runner.run_sync(   #ai ko run krta h user k msg k sath
            starting_agent=agent,
            input=history,
            run_config=config
        )

        response_content = result.final_output
        msg.content = response_content
        await msg.update()  #response ya answer update

        cl.user_session.set("chat history", result.to_input_list())

        print(f"User:{message.content}")
        print(f"Assistant:{response_content}")

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")