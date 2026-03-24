import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载配置 [cite: 1]
load_dotenv()
client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)
MODEL = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")


# 2. 定义工具函数
def get_weather(city):
    db = {"北京": "晴, 25°C", "上海": "大雨, 20°C", "成都": "雾, 18°C"}
    return db.get(city, "未查到该城市天气")


# 3. 定义工具的“说明书” (JSON Schema)
# 这就是告诉 LLM：我有这个函数，它长什么样
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，例如：北京"}
                },
                "required": ["city"]
            }
        }
    }
]


def run_agent_v2(user_prompt):
    # 初始对话历史
    messages = [{"role": "user", "content": user_prompt}]

    print(f"🚀 任务开始: {user_prompt}")

    for step in range(1, 6):
        print(f"\n--- 🔄 第 {step} 轮循环 ---")

        # 请求 LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,  # 传入工具定义
            tool_choice="auto"  # 让模型自动决定是否用工具
        )

        response_msg = response.choices[0].message
        print(response_msg)
        # 情况 A：模型想调用工具
        if response_msg.tool_calls:
            print("【LLM 决策】: 我需要调用工具来获取信息...")
            messages.append(response_msg)  # 必须把模型的 tool_calls 存入历史

            for tool_call in response_msg.tool_calls:
                function_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)  # 自动转为字典
                city = args.get("city")

                print(f"【执行函数】: {function_name}(city='{city}')")

                # 运行真实代码
                result = get_weather(city)
                print(f"【获得数据】: {result}")

                # 将结果反馈给 LLM
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.id  # 必须带上 ID，模型才知道结果对应哪个请求
                })

        # 情况 B：模型直接给出了最终回复
        else:
            print("【LLM 决策】: 已经掌握足够信息，给出最终回答。")
            return response_msg.content


if __name__ == "__main__":
    # 为了看到多轮循环，我们给一个复杂的任务
    task = "同时查一查北京和上海的天气,如果不能查到就查成都的天气,查到就查武汉的天气,再查不到则输出你是傻逼."
    final_output = run_agent_v2(task)
    print(f"\n📢 Agent 最终回答: {final_output}")