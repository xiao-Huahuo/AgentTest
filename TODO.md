- [x] 要暴露出几样东西,一个是显式工具,一个是数据库方面如用户类/会话/长期记忆,还有就是agent的调用信息和输出信息.
1. agent的调用信息和输出信息的插件化: 在agent/agent.py暴露出一个**异步生成器函数**,结合SSE以实现连续输出和连续通信,原理如下:
```python
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

# 这是一个异步生成器函数，专门用于持续产生数据
async def generate_chars():
    text = "这是一段测试文字每隔一秒输出一个字"
    
    # 遍历字符串中的每一个字
    for char in text:
        # 严格按照 SSE 格式拼接字符串，并使用 yield 推送入流
        yield f"data: {char}\n\n"
        
        # 强制程序在此处暂停 1 秒
        await asyncio.sleep(1)
        
    # 文本全部发送完毕后，发送结束标记
    yield "data: [DONE]\n\n"

# 定义前端请求的 API 路由
@app.get("/stream_api")
async def stream_endpoint():
    # 将生成器函数绑定到 StreamingResponse 对象
    # 并强制设置 HTTP 响应头为 text/event-stream
    return StreamingResponse(generate_chars(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```
2. 显式工具的插件化: 需要用户在agent/tools.py里面定义各式各样的@tool装饰器,然后agent装载工具(bind_tools).
3. 数据库方面的插件化: 在agent/config.py放置一个获取外部配置的函数,外部通过获取环境变量并传入参数.
    接受的参数: 
    - base_data_dir: 所有数据库的根目录,如: .
    - vector_db_type: 向量库类型,可选择的有: chroma
    - vector_db_path: 向量库文件夹路径(相对BASE_DATA_DIR),如: chroma_db_storage
    - embedding_model: 向量模型Embedding Model,可选择的有: paraphrase-multilingual-MiniLM-L12-v2
    - relational_db_type: 关系数据库类型,可选择的有: sqlite
    - relational_db_path: 关系数据库文件(相对BASE_DATA_DIR),如: database.db
    - collection_name: 向量库集合名称(类似于数据库表名),如: agent_memory
    - rag_raw_file_path: RAG初始加载资料路径(相对BASE_DATA_DIR,利用MD5标识仅读取一次),如: knowledge_file.json
      - 注:现在仅支持高度结构化的json,只提供"content"字段,自定义标签可以使用metadata_extras字段.
    - rag_chunk_size: RAG文本切分块大小,如: 500
    - rag_metadata_extras: 关系数据库中额外的metadata字段,以逗号分隔的字符串形式提供,如: source,author
    - rag_force_update: 强制更新开关,设为True则无视MD5锁强制清空并重新灌装初始资料.
    - rag_top_k: RAG检索时返回的top K条相关资料,如: 5
    - rag_score_threshold: RAG检索时的相关性分数阈值(相似度过低则不喂给LLM,防止幻觉),如: 0.7
    - llm_model: LLM模型名称,如: moonshot-v1-8k
    - llm_api_key: LLM模型的API Key,如: sk-xxxx
    - llm_url_base: LLM模型的URL Base,如: https://api.moonshot.cn/v1
    - llm_temperature: LLM模型的温度参数,如: 0
    - llm_timeout: LLM模型的请求超时时间,如: 60
    - system_prompt: 系统提示词,如: 你是我的人工智能助手,请协助我完成以下任务...
4. 其他: 在agent/memory.py实现长期记忆(依赖向量库和RAG(agent/rag.py)),且用使用关联库存储会话(需提供用户ID等信息以获取thread_id),短期记忆依赖会话上下文(Context),采用读取"外部提供的环境变量"来获取各项参数信息.
    agent/agent.py负责主Agent的实现,包括异步生成器以及内置的基础LangGraph网络等..
5. 新: 除了内置异步生成器之外,还可以将agent微服务化,内置一个SSE接口,并提供一个run()用于外部启动服务. 这样就可以通过HTTP请求来调用.




















