import os
import json
import uuid
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional

# 导入配置类，直接访问其内部定义的常量
from agent.config import AgentConfig


class LongTermMemory:
    """
    长期记忆模块。
    不再接收外部 config 实例，直接硬编码引用 AgentConfig 的类常量。 
    """
    # 成员字段声明
    local_ef: embedding_functions.SentenceTransformerEmbeddingFunction
    chroma_client: chromadb.ClientAPI
    collection: chromadb.Collection
    def __init__(self):
        """
        初始化向量库连接。
        完全不接收任何参数，直接从 AgentConfig 获取常量。 
        """
        # 直接使用 AgentConfig 的类常量进行初始化
        self.local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=AgentConfig.EMBEDDING_MODEL
        )

        # 初始化 ChromaDB 持久化客户端
        self.chroma_client = chromadb.PersistentClient(
            path=AgentConfig.VECTOR_DB_PATH
        )

        # 获取或创建集合
        self.collection = self.chroma_client.get_or_create_collection(
            name=AgentConfig.COLLECTION_NAME,
            embedding_function=self.local_ef
        )

    def _calculate_md5(self, data: List[Dict[str, str]]) -> str:
        """
        内部方法：计算预灌装数据内容的 MD5 哈希值，用于版本锁定。
        """
        # 确保字典键排序一致，从而保证相同内容的哈希值固定
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()

    def rag_ingest(self, user_id: str, raw_data: Optional[List[Dict[str, str]]] = None):
        """
        数据预灌装逻辑：
        1. 优先从 AgentConfig.RAG_RAW_FILE_PATH 读取本地 JSON。
        2. 合并函数参数传入的 raw_data。
        3. 利用 MD5 锁机制，仅在内容发生变更或强制更新时才同步向量库。
        """
        final_ingest_data = []

        # --- 第一步：加载本地配置文件中的知识 [cite: 2026-03-31] ---
        if os.path.exists(AgentConfig.RAG_RAW_FILE_PATH):
            try:
                with open(AgentConfig.RAG_RAW_FILE_PATH, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        final_ingest_data.extend(file_data)
                        print(f"--- [长期记忆] 已从本地文件 {AgentConfig.RAG_RAW_FILE_PATH} 加载 {len(file_data)} 条初始知识 ---")
            except Exception as e:
                print(f"--- [长期记忆] 读取本地知识文件失败: {e} ---")

        # --- 第二步：合并传入的裸数据 ---
        if raw_data:
            final_ingest_data.extend(raw_data)

        if not final_ingest_data:
            print("--- [长期记忆] 无任何数据需要同步 ---")
            return

        # --- 第三步：MD5 校验与同步逻辑 ---
        # 计算合并后总数据的 MD5 哈希
        current_hash = self._calculate_md5(final_ingest_data)

        # 获取旧哈希锁
        lock_id = f"data_lock_{user_id}"
        existing_lock = self.collection.get(ids=[lock_id])

        last_hash = None
        if existing_lock and existing_lock.get('metadatas') and len(existing_lock['metadatas']) > 0:
            last_hash = existing_lock['metadatas'][0].get("hash_lock")

        # 判断是否同步 (MD5 变化或 FORCE_UPDATE 为 True)
        if current_hash != last_hash or AgentConfig.RAG_FORCE_UPDATE:
            print(f"--- [长期记忆] 检测到内容变更，正在同步知识库 (User: {user_id}) ---")

            # 清理旧数据
            if self.collection.count() > 0:
                self.collection.delete(where={"user_id": user_id})

            documents = []
            metadatas = []
            ids = []

            for item in final_ingest_data:
                # 提取正文
                documents.append(item["content"])

                # 构建元数据
                meta = {"user_id": user_id}
                for key, value in item.items():
                    if key != "content":
                        meta[key] = value

                metadatas.append(meta)
                ids.append(str(uuid.uuid4()))

            # 批量写入
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

            # 更新哈希锁
            self.collection.upsert(
                ids=[lock_id],
                documents=["HASH_LOCK_MARKER"],
                metadatas=[{"hash_lock": current_hash, "user_id": user_id}]
            )
            print(f"--- [长期记忆] 知识同步完成，当前共计 {len(documents)} 条文档 ---")
        else:
            print(f"--- [长期记忆] 内容未变且未触发强制更新，跳过灌装 ---")

    def rag_query_tok_k(self, query: str, user_id: str, rag_top_k: Optional[int] = None) -> List[str]:
        """
        检索接口：根据查询语句返回最相关的记忆片段。
        """
        # 优先级：函数参数 > AgentConfig.RAG_TOP_K 常量 
        top_k = rag_top_k if rag_top_k is not None else AgentConfig.RAG_TOP_K

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"user_id": user_id}
        )

        # 提取检索到的文本文档
        if results and results.get('documents') and len(results['documents']) > 0:
            return results['documents'][0]
        return []

    def summarize_and_store_knowledge(self, user_id: str, content: str):
        """
        核心逻辑：将提炼出的新信息存入向量库，并强制更新哈希锁。
        """
        if not content or content.upper() == "NONE":
            return

        print(f"\n--- [系统自动提炼新记忆]: {content} ---")

        # 1. 直接存入向量库
        self.collection.add(
            documents=[content],
            metadatas=[{"category": "AUTO_EXTRACTED", "user_id": user_id}],
            ids=[str(uuid.uuid4())]
        )

        # 2. 更新该用户的哈希锁，标记状态为 STALE（陈旧）
        # 这将确保下次执行 rag_ingest 时，即便原始文件没变，也会触发重对齐。
        lock_id = f"data_lock_{user_id}"
        self.collection.upsert(
            ids=[lock_id],
            documents=["HASH_LOCK_MARKER"],
            metadatas=[{"hash_lock": "FORCE_UPDATE_STALE", "user_id": user_id}]
        )

    def close(self):
        """
        释放或关闭相关资源。 
        """
        pass

# 全局单例变量,确保在AgentConfig初始化之后才会初始化chromaDB,以防出现根目录出现未定义文件名向量库文件夹
_global_memory_instance = None

def get_long_term_memory():
    """
    延迟实例化函数：确保在调用时 AgentConfig 已经完成了 setup() [cite: 2026-03-31]
    """
    global _global_memory_instance
    if _global_memory_instance is None:
        # 此时 AgentConfig.VECTOR_DB_PATH 已经被 setup 修改过
        _global_memory_instance = LongTermMemory()
    return _global_memory_instance

# 共享LongTermMemory实例，确保全局唯一
# long_term_memory = LongTermMemory()