import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

print("=== 环境快速检查 ===")

# 1. 数据文件
print("\n[1] 数据文件检查:")
import os
files = {
    'data/raw_data/movie_data.json': '电影数据',
    'ls_jingqutousu - 去隐私.xlsx': '投诉Excel',
    '.env': '环境变量',
}
for f, desc in files.items():
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f"  {'[OK]' if exists else '[MISSING]'} {desc}: {f} ({size/1024:.1f}KB)")

# 2. 依赖包
print("\n[2] 依赖包检查:")
packages = [
    ('streamlit', 'Streamlit'),
    ('networkx', 'NetworkX'),
    ('chromadb', 'ChromaDB'),
    ('langchain_openai', 'LangChain OpenAI'),
    ('langchain_text_splitters', 'LangChain TextSplitters'),
    ('sentence_transformers', 'SentenceTransformers'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('plotly', 'Plotly'),
    ('openai', 'OpenAI'),
    ('neo4j', 'Neo4j'),
    ('jieba', 'Jieba'),
    ('docx', 'python-docx'),
    ('pypdf', 'PyPDF'),
]
for pkg, name in packages:
    try:
        __import__(pkg)
        print(f"  [OK] {name} ({pkg})")
    except ImportError as e:
        print(f"  [MISSING] {name} ({pkg})")

# 3. 电影数据JSON
print("\n[3] 电影数据检查:")
try:
    import json
    movies = json.load(open('data/raw_data/movie_data.json', 'r', encoding='utf-8'))
    print(f"  [OK] 电影数据: {len(movies)} 条")
    m = movies[0]
    keys = list(m.keys())
    print(f"  字段: {keys}")
    title = m.get('title', m.get('name', '?'))
    content = m.get('content', m.get('text', m.get('description', '')))
    print(f"  示例: {title} | {content[:80]}...")
except Exception as e:
    print(f"  [ERROR] {e}")

# 4. 投诉Excel
print("\n[4] 投诉Excel检查:")
try:
    import pandas as pd
    df = pd.read_excel('ls_jingqutousu - 去隐私.xlsx')
    print(f"  [OK] 投诉数据: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"  列名: {list(df.columns)}")
except Exception as e:
    print(f"  [ERROR] {e}")

# 5. 环境变量
print("\n[5] 环境变量检查:")
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv('OPENAI_API_KEY', '')
api_base = os.getenv('OPENAI_API_BASE', '')
llm_model = os.getenv('LLM_MODEL', '')
embed_model = os.getenv('EMBEDDING_MODEL', '')
print(f"  API_KEY: {'已配置 ' + '*'*(len(api_key)-8) + api_key[-6:] if api_key else '[未配置]'}")
print(f"  API_BASE: {api_base}")
print(f"  LLM_MODEL: {llm_model}")
print(f"  EMBEDDING_MODEL: {embed_model}")

print("\n=== 检查完成 ===")
