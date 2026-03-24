import os
import sqlite3
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. 加载环境变量
load_dotenv()


# --- 第一步：定义工具 ---
@tool
def get_weather(city: str):
    """获取指定城市的实时天气。"""
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
tool_node = ToolNode(tools)


# --- 第二步：定义状态 (State) ---
class AgentState(TypedDict):
    # 【关键修改 1】：使用 add_messages 替代原来的 lambda 函数
    # 这能确保在读取数据库记忆时，消息能根据 ID 正确去重和合并，而不会无限叠加
    messages: Annotated[list, add_messages]


# --- 第三步：定义逻辑节点函数 ---
model = ChatOpenAI(
    model=os.getenv("MOONSHOT_MODEL"),
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
    temperature=0
).bind_tools(tools)


def call_model(state: AgentState):
    """节点：调用大模型进行思考"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    """条件边：判断是去调工具还是直接结束"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"


# --- 第四步：构建图 (Graph) 结构 ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END}
)
workflow.add_edge("action", "agent")

# --- 第五步：引入 SQLite 记忆并编译 ---
# 【关键修改 2】：连接本地 SQLite 数据库文件
conn = sqlite3.connect("agent_memory_short.db", check_same_thread=False)
memory = SqliteSaver(conn)

# 【关键修改 3】：在编译时传入 checkpointer
app = workflow.compile(checkpointer=memory)

# --- 第六步：测试多轮对话运行 ---
if __name__ == "__main__":
    # 【关键修改 4】：配置 thread_id，Agent 会根据这个 ID 来存取记忆
    config = {"configurable": {"thread_id": "user_langgraph_test_01"}}

    # 第一轮对话：提出初步需求
    task_1 = "我想去上海玩，帮我查查天气。如果下雨就搜上海博物馆信息。"
    print(f"--- 第一轮对话开始：{task_1} ---")
    inputs_1 = {"messages": [HumanMessage(content=task_1)]}

    for output in app.stream(inputs_1, config=config):
        for node_name, state_update in output.items():
            last_msg = state_update["messages"][-1]
            if last_msg.content:
                print(f"[节点 {node_name} 回复]: {last_msg.content}")
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for t in last_msg.tool_calls:
                    print(f"[节点 {node_name} 调用工具]: {t['name']}")

    print("\n" + "=" * 50 + "\n")

    # 第二轮对话：直接追问，测试 Agent 是否记得“上海”和“下雨”的上下文
    task_2 = "那我不想去博物馆了，既然天气不好，帮我找个附近的酒店躺着吧。"
    print(f"--- 第二轮对话开始：{task_2} ---")
    # 注意这里不需要把之前的对话传进去，只传新的一句话
    inputs_2 = {"messages": [HumanMessage(content=task_2)]}

    for output in app.stream(inputs_2, config=config):
        for node_name, state_update in output.items():
            last_msg = state_update["messages"][-1]
            if last_msg.content:
                print(f"[节点 {node_name} 回复]: {last_msg.content}")
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for t in last_msg.tool_calls:
                    print(f"[节点 {node_name} 调用工具]: {t['name']}")