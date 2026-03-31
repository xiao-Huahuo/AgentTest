import json
import sqlite3
from typing import Annotated, TypedDict, Any, AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from agent.config import AgentConfig
from agent.memory import LongTermMemory, long_term_memory
from agent.tools import tools, tool_node


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
class AgentCore:
    model: Any
    memory_engine: LongTermMemory
    workflow: StateGraph
    conn: sqlite3.Connection
    saver: SqliteSaver
    app: CompiledStateGraph

    def __init__(self):
        # 调用共享长期记忆引擎，获取数据库连接实例 [cite: 2026-03-31]
        self.memory_engine = long_term_memory

        # 初始化模型并绑定工具
        self.model = ChatOpenAI(
            model=AgentConfig.LLM_MODEL,
            api_key=AgentConfig.LLM_API_KEY,
            base_url=AgentConfig.LLM_URL_BASE,
            temperature=AgentConfig.LLM_TEMPERATURE,
            timeout=AgentConfig.LLM_TIMEOUT
        ).bind_tools(tools)

        # 循环图结构(ReAct架构)
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("agent", self._call_model)
        self.workflow.add_node("action", tool_node)
        self.workflow.add_node("summarize", self._summarize_and_store)
        self.workflow.set_entry_point("agent")
        self.workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            path_map={"tools": "action", "summarize": "summarize"}
        ) # 条件边：agent -> action (如果需要工具) 或 summarize (如果回答完毕)
        self.workflow.add_edge("action", "agent")  # 工具执行完必须回大脑思考
        self.workflow.add_edge("summarize", END)

        # 持久化
        self.conn = sqlite3.connect(AgentConfig.RELATIONAL_DB_PATH, check_same_thread=False)
        self.saver = SqliteSaver(self.conn)
        self.app = self.workflow.compile(checkpointer=self.saver)

    def _call_model(self, state: AgentState):
        """节点：Agent 决策。"""
        system_msg = SystemMessage(content=AgentConfig.SYSTEM_PROMPT)
        response = self.model.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    def _should_continue(self, state: AgentState):
        """逻辑分支：判断是去执行工具还是结束并提炼记忆。"""
        last_message = state["messages"][-1]
        # 如果有工具调用，去 tools；否则去 summarize 提炼记忆 [cite: 2026-03-31]
        return "tools" if last_message.tool_calls else "summarize"

    def _summarize_and_store(self, state: AgentState):
        """节点：对话结束后提炼新知识。 [cite: 2026-03-31]"""
        summary_prompt = SystemMessage(
            content="分析对话，提取 1-2 条关于用户的新信息（偏好、习惯或背景）。如果没有新信息，请回复 NONE。只输出内容。")

        # 1. 仅通过模型提炼信息
        response = self.model.invoke([summary_prompt] + state["messages"])
        content = response.content.strip()

        # 2. 委托给 memory_engine 处理复杂的存储和锁逻辑
        self.memory_engine.summarize_and_store_knowledge(
            user_id=state["user_id"],
            content=content
        )

        return {}

    def stream_run(self, prompt: str, user_id: str, thread_id: str) -> AsyncGenerator[str, None]:
        """
        异步生成器：驱动图运行并输出 SSE 数据流。 [cite: 2026-03-31]

        使用示例 (FastAPI):
        ------------------
        @app.get("/stream")
        async def stream(q: str, uid: str, tid: str):
            return StreamingResponse(agent.astream_run(q, uid, tid), media_type="text/event-stream")

        使用示例 (Python Client):
        -----------------------
        async for chunk in agent.astream_run("你好", "user_1", "thread_1"):
            print(chunk)
        """
        inputs = {
            "messages": [HumanMessage(content=prompt)],
            "user_id": user_id
        }
        config = {"configurable": {"thread_id": thread_id}}

        print(f"\n{'=' * 20} 任务开始 (Thread: {thread_id}) {'=' * 20}")
        print(f"[用户输入]: {prompt}")

        # 使用 astream 迭代每一个节点的更新
        # 自动去 SQLite 数据库中检索该 ID 对应的所有历史消息并注入到 state["messages"] 中,
        # 此时得到的 event 已经是结合了“历史消息 + 当前输入”后的运行结果,即实现了"短期记忆"上下文 (Context)
        for event in self.app.stream(inputs, config=config, stream_mode="updates"):
            for node_name, state_update in event.items():
                # 增加判空保护，防止 summarize 节点返回 None 导致崩溃
                if state_update is None:
                    continue
                # 1. 终端追踪打印 (原版风格)
                print(f"\n>>> [动作追踪]: 进入节点 <{node_name}>")

                if "messages" not in state_update:
                    continue

                last_msg = state_update["messages"][-1]

                # 处理不同节点的终端显示逻辑
                if node_name == "agent":
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            print(f"--- [内核决策]: 意识到需要调用工具 [{tc['name']}]")
                    if last_msg.content:
                        print(f"[内核回复]: {last_msg.content}")

                elif node_name == "action":
                    print(f"--- [系统执行]: 工具执行完毕，结果已更新至上下文")

                elif node_name == "summarize":
                    print(f"--- [知识复盘]: 正在提炼并归档新发现的个人偏好")

                # 2. SSE 数据流推送 (微服务风格) [cite: 2026-03-31]
                payload = {
                    "node": node_name,
                    "user_id": user_id,
                    "content": last_msg.content if last_msg.content else "",
                    "tool_calls": [
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in getattr(last_msg, "tool_calls", [])
                    ] if hasattr(last_msg, "tool_calls") else []
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        print(f"\n{'=' * 20} 任务结束 {'=' * 20}")
        yield "data: [DONE]\n\n"

    def close(self):
        """释放数据库连接。 [cite: 2026-03-31]"""
        if self.conn:
            self.conn.close()