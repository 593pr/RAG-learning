from langchain_core.tools import tool

@tool
def calculator(expression:str)->str:
    """计算数学表达式，比如1+1=2，必须是纯数学表达式"""
    return str(eval(expression))

@tool
def weather(city:str)->str:
    """查询指定城市的天气"""
    return f"{city}今天晴天，25℃"

tools = [calculator,weather]