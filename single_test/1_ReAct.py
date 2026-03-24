import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载配置
load_dotenv()
client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)
MODEL = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")


# 2. 定义 Agent 的“手”（工具函数）
def get_weather(city):
    """模拟一个天气查询工具"""
    db = {"北京": "晴转多云, 25°C", "上海": "小雨, 21°C", "成都": "阴, 19°C"}
    return db.get(city, "抱歉，暂未查到该城市的天气。")


tools_map = {"get_weather": get_weather}

# 3. 定义 Agent 的“大脑协议”（System Prompt）
SYSTEM_PROMPT = """
你是一个有条理的 Agent。请严格按照以下格式思考和回答：

Thought: 思考你当前需要做什么。
Action: 调用工具名称 (目前仅支持: get_weather)。
Action Input: 工具所需的参数 (例如: 北京)。
Observation: (这里我会告诉你工具执行的结果)。

当你确认掌握了所有信息，请回答：
Final Answer: 你的最终结论。
"""


def run_agent(user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    print(f"🚀 任务启动: {user_prompt}")

    # 显式的循环：Agent 思考过程
    for step in range(1, 6):  # 最多允许思考 5 轮，防止死循环
        print(f"\n====================== 🔄 第 {step} 轮循环 ======================")
        print(f"【发送message】:\n{json.dumps(messages, indent=4, ensure_ascii=False)}")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0  # 设为 0 保证逻辑稳定
        ).choices[0].message.content

        print(f"【LLM 思考】:\n{response}")

        # 检查是否完成
        if "Final Answer:" in response:
            print("\n✅ 任务达成！")
            return response.split("Final Answer:")[-1].strip()

        # 解析 Action 和 Action Input
        try:
            action = re.search(r"Action:\s*(.*)", response).group(1).strip()
            action_input = re.search(r"Action Input:\s*(.*)", response).group(1).strip()

            if action in tools_map:
                print(f"【执行工具】: {action}('{action_input}')")
                # 真实的执行步骤
                observation = tools_map[action](action_input)
                print(f"【获得观察】: {observation}")

                # 重要：将 Observation 结果追加到对话历史，喂回给大模型.
                # 如果没有把工具结果（Observation）喂回给 LLM，它就会像失忆一样在原地打转。
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                print(f"❌ 错误：模型尝试调用不存在的工具 {action}")
                break
        except Exception as e:
            print(f"⚠️ 解析失败，模型可能没按格式说话: {e}")
            break


if __name__ == "__main__":
    final_res = run_agent("先查查北京的天气，如果北京下雨，就帮我查查上海的天气；如果北京晴天，就告诉我适合穿什么。")
    print(f"\n📢 最终输出: {final_res}")