from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool
from smolagents.models import TransformersModel
from transformers import AutoTokenizer

import config

login(config.password)

model_id = "Qwen/Qwen2-0.5B-Instruct"

model = TransformersModel(
    model_id=model_id,
    device_map="auto",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
)

agent = CodeAgent(
    model=model,
    tools=[DuckDuckGoSearchTool(), FinalAnswerTool()],
    additional_authorized_imports=["web_search", "requests"]
)

agent.run("Write a python code to get as much information as possible about the company OpenAI and their current work. The python code must be valid")