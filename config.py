"""
GraphRAG 智能问答系统 - 核心配置模块
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 大模型配置
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.deepseek.com/v1')
    LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-chat')
    
    # 向量检索配置
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))
    TOP_K_VECTOR = int(os.getenv('TOP_K_VECTOR', 3))
    TOP_K_GRAPH = int(os.getenv('TOP_K_GRAPH', 3))
    
    # 图谱配置
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    
    # 生成配置
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2000))
    MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', 4000))
    
    # 目录配置
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
    
    for dir_path in [DATA_DIR, UPLOAD_DIR]:
        os.makedirs(dir_path, exist_ok=True)
