# OpenAI API 调用
# PowerShell 示例：
#   $env:OPENAI_API_KEY="your-api-key-here"
#   py -m venv .venv
#   .\.venv\Scripts\Activate.ps1
#   pip install openai

from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    input="Write a one-sentence bedtime story about a unicorn.",
)
# client.responses.create(...) 表示使用 Responses API 创建一次响应。
# 更早的接口路线还有 client.chat.completions.create(...) 和 client.completions.create(...)

print(response.output_text)

#注意：response 不是单纯的一段字符串，而是一个响应对象，里面包含生成内容、模型信息、token 用量等。

#图片的输入：

from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "What teams are playing in this image?",
                },
                {
                    "type": "input_image",
                    "image_url": "https://api.nga.gov/iiif/a2e6da57-3cd1-4235-b20e-95dcaefed6c8/full/!800,800/0/default.jpg",
                },
            ],
        }
    ],
)

print(response.output_text)


#tool use
#网络搜索
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    tools=[{"type": "web_search"}],
    input="What was a positive news story from today?",
)

print(response.output_text)

#文件搜索：
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    input="What is deep research by OpenAI?",
    tools=[{"type": "file_search", "vector_store_ids": ["<vector_store_id>"]}],
)
print(response)
#vector = 文本的语义数字表示
#vector store = 存放这些语义向量的知识库
#file_search = 让模型先去知识库检索，再基于检索结果回答

#函数调用：（Function calling）
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia",
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

response = client.responses.create(
    model="gpt-5.6",
    input=[
        {"role": "user", "content": "What is the weather like in Paris today?"},
    ],
    tools=tools,
)

print(response.output[0].to_json())

#这段代码定义了一个函数工具；用户提问天气后，大模型会返回一个工具调用请求和参数，但不会自己真正调用，真正调用的是外部程序。
#模型 = 调度员
#你的函数 = 真正干活的人
#你的程序 = 中间协调者

#流式反馈，实时
from openai import OpenAI

client = OpenAI()

stream = client.responses.create(
    model="gpt-5.6",
    input=[
        {
            "role": "user",
            "content": "Say 'double bubble bath' ten times fast.",
        },
    ],
    stream=True,
)

for event in stream:
    print(event)

#构建代理：（handoff），将工作分配给其他的agent
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
