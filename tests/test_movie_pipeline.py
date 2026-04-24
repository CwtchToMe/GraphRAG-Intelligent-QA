# -*- coding: utf-8 -*-
"""快速测试电影数据处理管线"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from core.movie_data import load_movie_data, movie_to_chunks, get_available_datasets

# 1. 加载
movies = load_movie_data('data/raw_data/movie_data.json')
print(f"[OK] 加载电影数据: {len(movies)} 条")

# 2. 转 chunks
chunks = movie_to_chunks(movies)
print(f"[OK] 转为文本块: {len(chunks)} 块")
print(f"示例: {chunks[0]['id']} | {chunks[0]['text'][:80]}...")

# 3. 数据集探测
datasets = get_available_datasets()
print(f"[OK] 可用数据集: {list(datasets.keys())}")
for key, info in datasets.items():
    print(f"  - {key}: {info['name']} ({info['count']} 条)")

# 4. 验证 Prompt 存在
from core.movie_data import MOVIE_NER_RE_PROMPT
print(f"[OK] 电影领域 NER/RE Prompt 长度: {len(MOVIE_NER_RE_PROMPT)} 字")

print("\n[ALL OK] 电影数据管线验证通过！")
