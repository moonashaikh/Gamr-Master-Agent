# def main():
#     print("Hello from game-agent!")


# if __name__ == "__main__":
#     main()


import os

from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel ,RunConfig
from game_tools import roll_dice , generate_event
from agents.run import RunConfig

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)
# model = OpenAIChatCompletionsModel(
#    model="gpt-4o",
#      openai_client=external_client
#      )

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)
config = RunConfig(
     model=model,
     model_provider=external_client,
     tracing_disabled=True
 )

narrator_agent = Agent(
     name="NarratorAgent",
     instructions="You narrate the adventure.Ask the player for choice",
     model = model
)

monster_agent = Agent(
     name="MonsterAgent",
     instructions="You handle the monster encounter using roll_dice and generate_event",
     model = model,
     tools=[roll_dice , generate_event]
     
)

item_agent = Agent(
     name="itemAgent",
     instructions="You provide reward or items to the player",
     model = model
)

def main():
    print("\U0001F3A3 Welcome to Fantasy Game!\n")
    choice= input("Do you enter the forecast or turn back?->")
    
    result1 = Runner.run_sync(narrator_agent , choice , run_config=config)
    
    print("\n story:", result1.final_output)

    result2 = Runner.run_sync(monster_agent , "start_counter" , run_config=config)
    
    print("\n Encounter", result2.final_output)
    
    result3 = Runner.run_sync(item_agent , "Give reward" , run_config=config)
    
    print("\n Reward", result3.final_output)

if __name__ == "__main__":
     main()



