from agent.config import AgentConfig
from agent.agent import AgentCore
from agent.conversations import SQLiteConversationStore


def agent_full_workflow():
    # --- 1. 核心参数配置 ---
    AgentConfig.setup(
        BASE_DATA_DIR="./store",
        VECTOR_DB_PATH="chroma_storage",
        COLLECTION_NAME="long_term_memory_v2",
        EMBEDDING_MODEL="paraphrase-multilingual-MiniLM-L12-v2",
        RELATIONAL_DB_PATH="agent_relational.db",
        RAG_RAW_FILE_PATH="knowledge.json",
        RAG_CHUNK_SIZE=500,
        RAG_METADATA_EXTRAS=["category", "source"],
        RAG_FORCE_UPDATE=False,
        RAG_TOP_K=3,
        RAG_SCORE_THRESHOLD=0.7,
        LLM_MODEL="moonshot-v1-8k",
        LLM_API_KEY="sk-323ztWwkJNGkZD0ZR2yNgLYlgfrcsFOIVNC1az6EJG1G3QGQ",
        LLM_URL_BASE="https://api.moonshot.cn/v1",
        LLM_TEMPERATURE=0,
        LLM_TIMEOUT=60,
        SYSTEM_PROMPT=(
            "你是一个具备自主能力的 AI 助手。你可以查阅长期记忆，也可以解析文件。"
            "请根据用户需求灵活调用工具并提供帮助。"
        )
    )

    # --- 2. 初始化组件 ---
    # 实例化 Agent 内核（会自动连接 database.db 和 chroma_storage）
    agent = AgentCore()
    # 实例化会话管理器
    session_manager = SQLiteConversationStore()

    # --- 3. 模拟业务流程：创建并获取 Thread ID ---
    USER_ID = "slump_student_2024"
    # 使用管理器创建正式会话，MD5 ID 会存入 session_history 表
    new_session = session_manager.create_session(user_id=USER_ID, session_name="短期记忆交叉验证")
    THREAD_ID = new_session.session_id

    print(f"\n{'=' * 20} 测试开始 {'=' * 20}")
    print(f"[会话信息]: ID={THREAD_ID}, User={USER_ID}")

    try:
        # --- 4. 第一轮：注入信息 ---
        prompt_1 = "你好，我是张三。我正在研究如何使用 LangGraph 构建多 Agent 系统。"
        print(f"\n[用户]: {prompt_1}")

        # 驱动运行并实时观察节点流转
        for chunk in agent.stream_run(prompt_1, USER_ID, THREAD_ID):
            pass  # 内部已包含 print 追踪

        # --- 5. 第二轮：确认短期记忆 ---
        # 此时 Agent 应该能从 SqliteSaver 加载第一轮的对话上下文
        prompt_2 = "你还记得我叫什么名字吗？我刚才提到的研究方向是什么？"
        print(f"\n[用户]: {prompt_2}")

        for chunk in agent.stream_run(prompt_2, USER_ID, THREAD_ID):
            pass

        # --- 6. 验证长期记忆提炼 ---
        # 在 agent.py 的流程中，第二轮结束后会进入 summarize 节点
        # 你可以在控制台观察是否有 "--- [系统自动提炼新记忆] ---" 的日志输出

    finally:
        # --- 7. 资源清理 ---
        session_manager.close()
        agent.close()
        print(f"\n{'=' * 20} 测试结束 {'=' * 20}")


if __name__ == "__main__":
    agent_full_workflow()