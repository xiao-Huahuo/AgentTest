
## 插件概述

该插件是一个基于 LangGraph 构建的 Agent 持久化解决方案，旨在通过模块化的方式为 AI 助手提供短期对话记忆（SQLite）与长期知识库（ChromaDB）的集成能力。它支持同步流式输出，并内置了基于 MD5 哈希锁的知识库增量更新机制。

---

## 全局配置参数 (AgentConfig)

在启动插件前，必须通过 `AgentConfig.setup()` 方法注入以下参数。系统将根据这些参数初始化数据库连接与模型加载。

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| BASE_DATA_DIR | str | 所有数据存储的根路径 |
| VECTOR_DB_PATH | str | 向量数据库存储目录名 |
| RELATIONAL_DB_PATH | str | 关系型数据库文件名 |
| EMBEDDING_MODEL | str | Embedding 模型名称或本地绝对路径 |
| COLLECTION_NAME | str | 向量库集合（Collection）名称 |
| RAG_RAW_FILE_PATH | str | 初始知识库 JSON 文件路径 |
| RAG_CHUNK_SIZE | int | 文本切分块大小，默认 500 |
| RAG_METADATA_EXTRAS | list | 向量库中需要保留的额外元数据字段 |
| RAG_FORCE_UPDATE | bool | 是否强制刷新向量库，跳过哈希校验 |
| RAG_TOP_K | int | 检索时返回的相关文档数量 |
| RAG_SCORE_THRESHOLD | float | 检索相关性分数阈值，默认 0.7 |
| LLM_MODEL | str | 大语言模型名称 |
| LLM_API_KEY | str | LLM 服务商的 API 密钥 |
| LLM_URL_BASE | str | LLM 服务的 API 基础地址 |
| LLM_TEMPERATURE | float | 模型采样温度 |
| LLM_TIMEOUT | int | 请求超时时间（秒） |
| SYSTEM_PROMPT | str | Agent 的核心系统提示词 |

---

## 完整接口方法说明

### 1. 会话管理类 (SQLiteConversationStore)
负责处理业务层的会话生命周期管理。

* **create_session(user_id: str, session_name: str = "新对话") -> SessionRecord**
    创建并持久化一个新的会话记录，自动生成基于时间戳和用户 ID 的 MD5 哈希作为 session_id。
* **delete_session(session_id: str) -> bool**
    从数据库中物理删除指定的会话记录。
* **update_session_name(session_id: str, new_name: str) -> bool**
    更新指定会_id 的会话显示名称。
* **get_session(session_id: str) -> Optional[SessionRecord]**
    根据 ID 获取单条会话的详细结构体数据。
* **get_user_sessions(user_id: str) -> List[SessionRecord]**
    获取指定用户的所有历史对话列表，按创建时间倒序排列。
* **close()**
    关闭与关系型数据库的连接。

### 2. Agent 内核类 (AgentCore)
负责执行决策逻辑、工具调用及记忆流转。

* **stream_run(prompt: str, user_id: str, thread_id: str) -> Generator[str, None, None]**
    核心运行接口。这是一个同步生成器，驱动图运行并实时产生 SSE 格式的数据流。它会自动根据 thread_id 恢复短期记忆上下文。
* **close()**
    安全释放内核占用的所有数据库资源，包括 SqliteSaver 检查点连接。

### 3. 长期记忆类 (LongTermMemory)
负责基于向量库的知识存储与检索。

* **rag_ingest(user_id: str, raw_data: Optional[List[Dict[str, str]]] = None)**
    同步初始知识库。该方法会对比本地文件 MD5 码，仅在变动时更新向量库。
* **rag_query_tok_k(query: str, user_id: str, rag_top_k: Optional[int] = None) -> List[str]**
    根据查询语句检索最相关的长期记忆片段。
* **summarize_and_store_knowledge(user_id: str, content: str)**
    将 Agent 提炼的新知识存入向量库，并标记哈希锁为过期状态，以确保下次同步时强制刷新。

---

## 完整搭建流程与示例代码



以下是整合上述所有接口搭建 Agent 的完整流程：

```python
from agent.config import AgentConfig
from agent.agent import AgentCore
from agent.conversations import SQLiteConversationStore
from agent.config import AgentConfig
from agent.agent import AgentCore
from agent.conversations import SQLiteConversationStore


def agent_full_workflow():
    # --- 1. 核心参数配置 ---
    AgentConfig.setup(
        BASE_DATA_DIR="./agent",
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
```