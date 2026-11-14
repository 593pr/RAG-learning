
"""
8 月 16 日练习：OpenAI API 调用（离线练习版）

目标：
1. 输入英文文本
2. 模拟一次 OpenAI 返回
3. 输出中文翻译 + 要点总结

以后如果你有额度了，可以把 mock_openai_response() 换成真实 API 调用。
"""
from openai import OpenAI

client = OpenAI()

def build_prompt(text: str) -> str:
    """构造提示词。"""
    return f"""请完成下面两个任务：
1.把英文翻译成流利的中文
2.将中文进行提炼总结出三个要点

英文内容：{text}

请按下面格式输出：
翻译：
<中文翻译>

要点总结：
1. 要点1
2. 要点2
3. 要点3
"""


def mock_openai_response(text: str) -> dict:
    """模拟 OpenAI API 的返回结果。"""
    return {
        "translation":f"[模拟翻译]{text}",
        "summary":[
            "这是第一个要点",
            "这是第二个要点",
            "这是第三个要点",
        ],
    }

def openai_calling(text):
    
    try:
        response = client.responses.create (
        model = "GPT-5.6",
        input = build_prompt(text),
        )
        print(f"这次对话的token用量为：{response.usage}")
        return response.output_text
    
    except Exception as error:
        print("openai API调用失败",error)
        return None



def parse_result(response: dict) -> tuple[str, list[str]]:
    """从响应里提取翻译和总结。"""
    translation = response["translation"]
    bullet_points = response["summary"]

    return translation, bullet_points


def print_result(translation: str, bullet_points: list[str]) -> None:
    """格式化输出结果。"""
    pass


def main() -> None:
    text = input("请输入英文文本：").strip()

    if not text:
        print("输入不能为空。")
        return

    prompt = build_prompt(text)
    print("\n=== Prompt ===")
    print(prompt)

    response = mock_openai_response(text)
    translation, bullet_points = parse_result(response)

    print_result(translation, bullet_points)

#单一的上下文的保存

conversation = []

conversation.append(
    {
        "role" : "user",
        "content" : "xxxxxxx",

    }
)

conversation.append(
    {
        "role" : "assistant",
        "content" : "xxxxxxx",
    }
)

for message in conversation:
    print(message['role'],message['content'])

#让模型记住上下文

history = []

def chat_with_history(history,user_text):
    history.append(
        {
            "role" : "user",
            "content" : [{
                "type" : "input_text","text" : user_text
            }],
        }
    )

    response = client.responses.create(
        model = "GPT-5.6"
        input = history
    )

    history.append(
        {
            "role" : "assistant",
            "content" : [{
                "type" : "output_text","text" : response.output_text
            }],
        }
    )

    return response.output_text

reply = chat_with_history(history, "我叫小明")
print(reply)

reply = chat_with_history(history, "我叫什么？")
print(reply)

#只有当直接python运行该文件时才调用main函数，不能import

if __name__ == "__main__":
    main()