import os
import json
from typing import Annotated, Union, TypedDict, List
from dotenv import load_dotenv

# LangChain & LangGraph 核心组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. 加载环境变量
load_dotenv()


# --- 第一步：定义工具 (使用 @tool 装饰器自动生成 Schema) ---
@tool
def get_weather(city: str):
    """获取指定城市的实时天气。"""
    # 模拟逻辑：设定上海正在下雨
    if "上海" in city:
        return "大雨, 20°C，不建议室外活动。"
    return "晴, 25°C，天气非常舒适。"


@tool
def search_hotels(location: str):
    """在指定地点搜索高评分酒店。"""
    return f"{location}附近的『和平饭店』目前有房，评分 4.9，坐拥外滩江景。"


@tool
def search_museum(museum_name: str):
    """查询博物馆的开馆时间和预约状态。"""
    return f"{museum_name}目前正常开放，建议提前通过公众号预约，室内空调舒适。"


tools = [get_weather, search_hotels, search_museum]
# ToolNode 是 LangGraph 提供的标准节点，专门用来执行被调用的工具
tool_node = ToolNode(tools)


# --- 第二步：定义状态 (State) ---
class AgentState(TypedDict):
    # Annotated[类型, 累加函数]
    # lambda x, y: x + y 表示新消息会追加到旧消息列表后，而不是覆盖
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]


# --- 第三步：定义逻辑节点函数 ---

# 初始化模型并绑定工具
model = ChatOpenAI(
    model=os.getenv("MOONSHOT_MODEL"),
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
    temperature=0
).bind_tools(tools)


def call_model(state: AgentState):
    """节点：调用大模型进行思考"""
    response = model.invoke(state["messages"])
    # 只要返回新增的消息，LangGraph 会自动通过 reducer 累加到 state 中
    return {"messages": [response]}


def should_continue(state: AgentState):
    """条件边：判断是去调工具还是直接结束"""
    last_message = state["messages"][-1]
    # 如果模型返回的消息中包含 tool_calls，说明它想用工具
    if last_message.tool_calls:
        return "continue"
    # 否则，任务完成
    return "end"


# --- 第四步：构建图 (Graph) 结构 ---

workflow = StateGraph(AgentState)

# 1. 添加节点
workflow.add_node("agent", call_model)  # 思考节点
workflow.add_node("action", tool_node)  # 执行工具节点

# 2. 设置起点
workflow.set_entry_point("agent")

# 3. 添加条件边 (从 agent 出发)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",  # 如果返回 continue，去 action 节点
        "end": END  # 如果返回 end，直接结束
    }
)

# 4. 添加普通边 (从 action 出发)
# 工具执行完后，必须回到 agent 让模型看一眼 Observation 并决定下一步
workflow.add_edge("action", "agent")

# 5. 编译
app = workflow.compile()

# --- 第五步：测试运行 ---

if __name__ == "__main__":
    task = "我想去上海玩，帮我查查天气。如果天气好就搜外滩酒店，下雨就搜上海博物馆信息。最后给我建议。"

    inputs = {"messages": [HumanMessage(content=task)]}

    print(f"🚀 任务开始: {task}\n")

    # 使用 stream 模式可以看到状态在节点间流转的过程
    for output in app.stream(inputs):
        for node_name, state_update in output.items():
            print(f"【进入节点】: {node_name}")

            # 打印当前节点产生的最新一条消息
            last_msg = state_update["messages"][-1]

            if last_msg.content:
                print(f"  内容: {last_msg.content}")

            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tool in last_msg.tool_calls:
                    print(f"  ⚙️ 决定调用工具: {tool['name']}, 参数: {tool['args']}")
        print("-" * 30)

    # 将你的 app 编译结果导出为图片
    try:
        with open("graph.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("已生成工作流结构图: graph.png")
    except Exception:
        # 如果缺少绘图库，也可以打印 Mermaid 语法字符串，粘贴到在线编辑器查看
        print(app.get_graph().draw_mermaid())