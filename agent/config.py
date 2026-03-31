import os


class AgentConfig:
    """
    Agent 核心配置常量。
    必须给所有大写变量赋予初始值（哪怕是空值），否则类对象无法直接访问。 [cite: 2026-03-31]
    """
    # --- 基础路径配置 ---
    BASE_DATA_DIR = "."
    VECTOR_DB_PATH = ""
    RELATIONAL_DB_PATH = ""

    # --- 向量库与 Embedding 配置 ---
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    COLLECTION_NAME = "agent_memory"

    # --- RAG 检索与灌装配置 ---
    RAG_RAW_FILE_PATH = ""
    RAG_CHUNK_SIZE = 500
    RAG_METADATA_EXTRAS = []
    RAG_FORCE_UPDATE = False
    RAG_TOP_K = 5
    RAG_SCORE_THRESHOLD = 0.7

    # --- LLM 核心配置 ---
    LLM_MODEL = ""
    LLM_API_KEY = ""
    LLM_URL_BASE = ""
    LLM_TEMPERATURE = 0.0
    LLM_TIMEOUT = 60
    SYSTEM_PROMPT = ""

    @classmethod
    def setup(cls, **kwargs):
        """
        使用 cls.变量名 确保直接修改类属性。 [cite: 2026-03-31]
        """
        cls.BASE_DATA_DIR = kwargs.get("BASE_DATA_DIR", ".")
        if not os.path.exists(cls.BASE_DATA_DIR):
            os.makedirs(cls.BASE_DATA_DIR, exist_ok=True)

        # 路径类参数需要基于 BASE_DATA_DIR 动态拼接 [cite: 2026-03-31]
        cls.VECTOR_DB_PATH = os.path.join(cls.BASE_DATA_DIR, kwargs.get("VECTOR_DB_PATH", "chroma_db_storage"))
        cls.RELATIONAL_DB_PATH = os.path.join(cls.BASE_DATA_DIR, kwargs.get("RELATIONAL_DB_PATH", "database.db"))
        cls.RAG_RAW_FILE_PATH = os.path.join(cls.BASE_DATA_DIR, kwargs.get("RAG_RAW_FILE_PATH", "knowledge_file.json"))

        # 其他参数直接赋值 [cite: 2026-03-31]
        cls.COLLECTION_NAME = kwargs.get("COLLECTION_NAME", "agent_memory")
        cls.EMBEDDING_MODEL = kwargs.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        cls.RAG_CHUNK_SIZE = int(kwargs.get("RAG_CHUNK_SIZE", 500))

        # 元数据处理 [cite: 2026-03-31]
        extras = kwargs.get("RAG_METADATA_EXTRAS", ["category", "source"])
        cls.RAG_METADATA_EXTRAS = extras if isinstance(extras, list) else extras.split(",")

        cls.RAG_FORCE_UPDATE = kwargs.get("RAG_FORCE_UPDATE", False)
        cls.RAG_TOP_K = int(kwargs.get("RAG_TOP_K", 5))
        cls.RAG_SCORE_THRESHOLD = float(kwargs.get("RAG_SCORE_THRESHOLD", 0.7))

        cls.LLM_MODEL = kwargs.get("LLM_MODEL", "moonshot-v1-8k")
        cls.LLM_API_KEY = kwargs.get("LLM_API_KEY", "")
        cls.LLM_URL_BASE = kwargs.get("LLM_URL_BASE", "https://api.moonshot.cn/v1")
        cls.LLM_TEMPERATURE = float(kwargs.get("LLM_TEMPERATURE", 0))
        cls.LLM_TIMEOUT = int(kwargs.get("LLM_TIMEOUT", 60))
        cls.SYSTEM_PROMPT = kwargs.get("SYSTEM_PROMPT", "你是一个具备自主能力的 AI 助手。")