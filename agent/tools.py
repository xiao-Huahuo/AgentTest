from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from agent.memory import LongTermMemory,get_long_term_memory

# TODO: 用户需在这里手动写Agent可调用的工具和逻辑

@tool
def query_long_term_memory(query: str):
    """当需要获取用户的项目背景、历史经验或技术偏好时，调用此工具检索长期记忆。"""
    long_term_memory = get_long_term_memory()
    # 此处 user_id 固定为演示 ID
    results = long_term_memory.collection.query(query_texts=[query], n_results=3, where={"user_id": "slump_student_2024"})
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