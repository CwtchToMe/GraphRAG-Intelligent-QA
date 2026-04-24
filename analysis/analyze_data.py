# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_excel('ls_jingqutousu - 去隐私.xlsx')

print('Rows:', len(df))
print()

# 所有列的非空数量和样例
for col in df.columns:
    non_null = df[col].dropna()
    print(f'{col}: {len(non_null)} non-null, examples: {list(non_null.head(3))}')

print()
print('=== sfd (诉求人) ===')
print(sorted(df['sfd'].dropna().unique()))

print()
print('=== 处理时限(bjsx) ===')
print(df['bjsx'].dropna().head(10))

print()
print('=== 回复内容(sfhf) ===')
print(df['sfhf'].dropna().unique())

print()
print('=== 工单标题样例(gdbt) ===')
for i, row in df[df['gdbt'].notna()].head(5).iterrows():
    print(f'  {row["ldr"]}: {row["gdbt"]}')

print()
print('=== 内容长度统计 ===')
lengths = df['zynr'].dropna().str.len()
print(f'min: {lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.0f}')

print()
print('=== 工单类型(sjlx) ===')
for v in df['sjlx'].dropna().unique():
    parts = v.split('|')
    for p in parts:
        print(f'  {p}')

# 看一条最长内容
longest_idx = lengths.idxmax()
longest = df.loc[longest_idx]
print()
print('=== 最长内容 ===')
print('gdbh:', longest['gdbh'])
print('ldr:', longest['ldr'])
print('blbm:', longest['blbm'])
print('zynr:', longest['zynr'][:500])

# 看具体关系内容
print()
print('=== 详细内容分析 ===')
for i in [1, 4, 18, 40, 62]:
    row = df.iloc[i]
    print(f'--- Row {i} ---')
    print('ldr:', row['ldr'])
    print('blbm:', row['blbm'])
    print('yjfl:', row['yjfl'], '/', row['ejfl'], '/', row['sjfl'])
    print('content:', row['zynr'][:200])
    print()
