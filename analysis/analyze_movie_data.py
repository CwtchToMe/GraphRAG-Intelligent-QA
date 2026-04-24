# -*- coding: utf-8 -*-
import json

# 分析 movie_data.json 结构
with open('data/raw_data/movie_data.json', 'r', encoding='utf-8') as f:
    movies = json.load(f)

print('Type:', type(movies))
print('Count:', len(movies) if isinstance(movies, list) else 'not a list')
if isinstance(movies, list) and len(movies) > 0:
    print('First item keys:', list(movies[0].keys()))
    print('First item:', json.dumps(movies[0], ensure_ascii=False, indent=2)[:800])
    print()
    # 看最后几条
    for m in movies[-3:]:
        print('Keys:', list(m.keys()))
        text = m.get('text', '') or m.get('content', '') or str(m)
        print('Text length:', len(text))
        print('---')
