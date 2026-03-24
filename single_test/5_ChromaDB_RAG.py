import os
import uuid
import sqlite3
import json
import hashlib
from typing import Annotated, TypedDict, List, Literal

# LangChain & LangGraph 核心组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# 向量数据库与本地 Embedding
import chromadb
from chromadb.utils import embedding_functions

# 1. 加载环境变量
from dotenv import load_dotenv

load_dotenv()

# --- 第一步：初始化长期记忆 (ChromaDB) ---
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
collection = chroma_client.get_or_create_collection(
    name="long_term_memory_v2",
    embedding_function=local_ef
)


# --- 会话 ID 计数器管理器 ---
def get_next_session_id(conn: sqlite3.Connection):
    """从数据库读取并递增会话 ID"""
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS session_counter (id INTEGER PRIMARY KEY, count INTEGER)")
    cursor.execute("SELECT count FROM session_counter WHERE id = 1")
    row = cursor.fetchone()

    if row is None:
        cursor.execute("INSERT INTO session_counter (id, count) VALUES (1, 1)")
        conn.commit()
        return "session_001"
    else:
        new_count = row[0] + 1
        cursor.execute("UPDATE session_counter SET count = ? WHERE id = 1", (new_count,))
        conn.commit()
        return f"session_{new_count:03d}"


# --- 第二步：数据预灌装逻辑 (MD5 锁) ---
def ingest_external_knowledge(user_id: str):
    """使用 MD5 哈希作为“锁”，确保内容变更时才更新"""
    mock_data = [
        {"content": "用户是软件工程专业大二学生，专注智能软件开发。", "category": "A"},
        {"content": "ClearNotify 是一个利用 FastAPI 和 Vue 实现的 AI 公告解析项目。", "category": "B"},
        {"content": "SQLite 负责存储短期 Thread 记忆，ChromaDB 存储长期知识。", "category": "C"},
        {"content": "用户在 2025 年完成了单周期 CPU 的 Verilog 设计项目。", "category": "B"}
    ]

    data_str = json.dumps(mock_data, sort_keys=True)
    current_hash = hashlib.md5(data_str.encode('utf-8')).hexdigest()

    existing_lock = collection.get(ids=["data_lock_id"])
    last_hash = existing_lock['metadatas'][0].get("hash_lock") if existing_lock['metadatas'] else None

    if current_hash != last_hash:
        print(f"--- 检测到内容变更，正在同步长期记忆库 (User: {user_id}) ---")
        if collection.count() > 0:
            collection.delete(where={"user_id": user_id})

        for item in mock_data:
            collection.add(
                documents=[item["content"]],
                metadatas=[{"category": item["category"], "user_id": user_id}],
                ids=[str(uuid.uuid4())]
            )

        collection.upsert(
            ids=["data_lock_id"],
            documents=["HASH_LOCK_MARKER"],
            metadatas=[{"hash_lock": current_hash, "user_id": user_id}]
        )
        print(f"--- 成功同步背景知识，哈希锁已更新 ---")
    else:
        print(f"--- 长期记忆库内容未变，已跳过导入 ---")


# --- 第三步：定义显式工具 (让 Agent 自主调用) ---

@tool
def query_long_term_memory(query: str):
    """当需要获取用户的项目背景、历史经验或技术偏好时，调用此工具检索长期记忆。"""
    # 此处 user_id 固定为演示 ID
    results = collection.query(query_texts=[query], n_results=3, where={"user_id": "slump_student_2024"})
    context = "\n".join(results['documents'][0]) if results['documents'][0] else "未找到相关记忆。"
    return f"【长期记忆库检索结果】:\n{context}"


@tool
def parse_local_file(file_name: str):
    """当用户提到需要分析某个文件（如校赛通知、PDF、文本）时，调用此工具模拟解析内容。"""
    if "校赛" in file_name:
        return "【文件解析结果】: 该公告要求 2026 年计算机设计大赛作品需在 5 月 1 日前提交，必须包含源码、演示视频和 Redis 环境配置说明。"
    return f"【文件解析结果】: 未找到文件 {file_name} 的具体内容。"


tools = [query_long_term_memory, parse_local_file]
tool_node = ToolNode(tools)


# --- 第四步：定义 LangGraph 状态与节点 ---

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


model = ChatOpenAI(
    model=os.getenv("MOONSHOT_MODEL"),
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url=os.getenv("MOONSHOT_BASE_URL"),
    temperature=0
).bind_tools(tools)


def call_model(state: AgentState):
    """节点：Agent 决策。现在不再预注入 Context，由 Agent 自己决定是否查记忆。"""
    system_msg = SystemMessage(
        content="你是一个具备自主能力的 AI 助手。你可以查阅长期记忆，也可以解析文件。请根据用户需求灵活调用工具。")
    response = model.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    """逻辑分支：判断是去执行工具还是去提炼记忆"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "summarize"


def summarize_and_store(state: AgentState):
    """节点：对话结束后提炼新知识并存入长期记忆"""
    summary_prompt = SystemMessage(
        content="分析对话，提取 1-2 条关于用户的新信息。如果没有新信息，请回复 NONE。只输出内容。")
    response = model.invoke([summary_prompt] + state["messages"])
    content = response.content.strip()

    if content != "NONE":
        print(f"\n--- [系统自动提炼新记忆]: {content} ---")
        collection.add(
            documents=[content],
            metadatas=[{"category": "AUTO_EXTRACTED", "user_id": state["user_id"]}],
            ids=[str(uuid.uuid4())]
        )
        # 更新哈希锁强制下次同步
        collection.upsert(
            ids=["data_lock_id"],
            documents=["HASH_LOCK_MARKER"],
            metadatas=[{"hash_lock": "FORCE_UPDATE_STALE", "user_id": state["user_id"]}]
        )
    return state


# --- 第五步：构建循环图结构 (ReAct 架构) ---

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("summarize", summarize_and_store)

workflow.set_entry_point("agent")
# 条件边：agent -> action (如果需要工具) 或 summarize (如果回答完毕)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "action", "summarize": "summarize"}
)
workflow.add_edge("action", "agent")  # 工具执行完必须回大脑思考
workflow.add_edge("summarize", END)

# 持久化
conn = sqlite3.connect("short_term_memory.db", check_same_thread=False)
saver = SqliteSaver(conn)
app = workflow.compile(checkpointer=saver)

# --- 第六步：执行综合挑战测试 ---

if __name__ == "__main__":
    USER_ID = "slump_student_2024"
    THREAD_ID = get_next_session_id(conn)
    print(f"--- 当前启动会话: {THREAD_ID} ---")
    ingest_external_knowledge(USER_ID)

    config = {"configurable": {"thread_id": THREAD_ID}}

    # 构造一个需要“多步思考”的问题
    complex_prompt = "根据我目前做的那个项目的技术选型，对比‘校赛通知.pdf’的要求，我还需要在 Redis 方面补充什么配置？"

    print(f"\n{'=' * 20} 挑战开始 {'=' * 20}")
    print(f"[用户]: {complex_prompt}")

    # 这里的 stream 会显示 Agent 往返于 agent 和 action 之间的过程
    for output in app.stream({"messages": [HumanMessage(content=complex_prompt)], "user_id": USER_ID}, config=config):
        for node_name, state_update in output.items():
            print(f"\n>>> [动作追踪]: 进入节点 <{node_name}>")
            last_msg = state_update["messages"][-1]

            if node_name == "agent":
                if last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"--- [Agent 决策]: 意识到需要调用 [{tc['name']}]，参数: {tc['args']}")
                if last_msg.content:
                    print(f"[AI 最终回复]: {last_msg.content}")

            elif node_name == "action":
                print(f"--- [系统执行]: 工具运行完毕，结果已存入消息流 ---")

            elif node_name == "summarize":
                print(f"--- [系统复盘]: 正在进行后台知识归档 ---")

    print(f"\n{'=' * 50}\n测试结束。")

    # 将你的 app 编译结果导出为图片
    try:
        with open("graph.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("已生成工作流结构图: lang_graph.png")
    except Exception:
        # 如果缺少绘图库，也可以打印 Mermaid 语法字符串，粘贴到在线编辑器查看
        print(app.get_graph().draw_mermaid())