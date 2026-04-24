import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# 1. 快速验证数据文件
print("=" * 60)
print("Step 1: 验证数据文件")
print("=" * 60)
import json
try:
    movies = json.load(open('data/raw_data/movie_data.json', 'r', encoding='utf-8'))
    print(f"[OK] 电影数据: {len(movies)} 条")
    print(f"    示例: {movies[0].get('title', '?')} | {movies[0].get('content', '')[:60]}...")
except Exception as e:
    print(f"[ERROR] 电影数据读取失败: {e}")

try:
    df_chunks = json.load(open('data/raw_data/movie_chunks.json', 'r', encoding='utf-8'))
    print(f"[OK] 电影chunks: {len(df_chunks)} 条")
except Exception as e:
    print(f"[WARN] movie_chunks.json: {e}")

print()

# 2. 验证依赖包
print("=" * 60)
print("Step 2: 验证依赖包")
print("=" * 60)
packages = [
    'streamlit', 'networkx', 'chromadb', 'langchain_openai', 'langchain_ollama',
    'langchain_text_splitters', 'sentence_transformers', 'pandas', 'numpy',
    'plotly', 'openai', 'langchain_community', 'langchain_core', 'neo4j'
]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg}")
    except ImportError as e:
        print(f"  [MISSING] {pkg}: {e}")
print()

# 3. 验证环境变量
print("=" * 60)
print("Step 3: 验证环境变量")
print("=" * 60)
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv('OPENAI_API_KEY', '')
api_base = os.getenv('OPENAI_API_BASE', '')
llm_model = os.getenv('LLM_MODEL', '')
embed_model = os.getenv('EMBEDDING_MODEL', '')
print(f"  API_KEY: {'已配置' if api_key else '未配置'} (key={'*'+api_key[-6:] if api_key else 'None'})")
print(f"  API_BASE: {api_base}")
print(f"  LLM_MODEL: {llm_model}")
print(f"  EMBEDDING_MODEL: {embed_model}")
print()

# 4. 初始化LLM服务
print("=" * 60)
print("Step 4: 初始化LLM服务 (DeepSeek API)")
print("=" * 60)
from core.llm_service import LLMService
llm = LLMService(
    api_key=api_key,
    api_base=api_base,
    model_name=llm_model,
    local_model="qwen2.5:7b-instruct",
    timeout=30
)
print(f"  prefer_local=False, skip_ollama")
if api_key:
    ok = llm.initialize(prefer_local=False)
    if ok:
        print(f"  [OK] LLM初始化成功, 模式: {llm.mode}, 模型: {llm.model_name if llm.mode=='api' else llm.local_model}")
        # 测试调用
        try:
            resp = llm.generate_answer("你好，请用一句话介绍你自己", max_retries=1)
            print(f"  [OK] 测试回复: {resp[:100]}")
        except Exception as e:
            print(f"  [WARN] 测试调用失败: {e}")
    else:
        print(f"  [ERROR] LLM初始化失败")
else:
    print(f"  [SKIP] 未配置API_KEY")

print()
print("=" * 60)
print("基础环境验证完成！")
print("=" * 60)
