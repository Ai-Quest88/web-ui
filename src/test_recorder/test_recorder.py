import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()  # Load environment variables from .env file

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.custom_agent import CustomAgent
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.controller.custom_controller import CustomController
from langchain_mistralai import ChatMistralAI
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig

browser = Browser(
	config=BrowserConfig(
		disable_security=True,
		headless=False,
		new_context_config=BrowserContextConfig(save_recording_path='./tmp/recordings'),
	)
)
llm = ChatMistralAI(model_name="mistral-large-latest", temperature=0.0, api_key=SecretStr("3LwHm6LjdjaooZwnEAHPjtpaJfjIHesg"))

async def main():
	# Create the recorded_tests directory if it doesn't exist
	os.makedirs('test_recorder/recorded_tests', exist_ok=True)
	
	controller = CustomController()
	agent = CustomAgent(
		task='Navigate to www.example.com and verify that the page title contains the text "Example Domain"',
		llm=llm,
		browser=browser,
		controller=controller,
		system_prompt_class=CustomSystemPrompt,
		agent_prompt_class=CustomAgentMessagePrompt,
		use_vision=False,
		tool_calling_method="json_schema"
	)
	
	# Run the task and save raw model_actions
	result = await agent.run()
	actions = list(result.model_actions())
	
	with open('src/test_recorder/recorded_tests/example_test.json', 'w') as f:
		json.dump(actions, f, indent=2, default=str)
	
	await browser.close()

if __name__ == '__main__':
	asyncio.run(main())
