# -*- coding: utf-8 -*-
import json

with open('data/raw_data/movie_data.json', 'r', encoding='utf-8') as f:
    movies = json.load(f)

# 统计分类
from collections import Counter
cats = Counter(m['category'] for m in movies)
print('=== 分类统计 ===')
for c, n in cats.most_common():
    print(f'  {c}: {n}')

print(f'\n总计: {len(movies)} 条\n')

# 看各分类的样例
for cat in list(dict(cats.most_common(3)).keys()):
    print(f'=== {cat} 样例 ===')
    samples = [m for m in movies if m['category'] == cat][:3]
    for s in samples:
        print(f'  [{s["id"]}] {s["title"]}: {s["content"][:120]}')
    print()
