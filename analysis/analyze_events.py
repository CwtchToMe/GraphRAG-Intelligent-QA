# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_excel('ls_jingqutousu - 去隐私.xlsx')

# 分析内容中的具体事件
print('=== 每条工单的具体事件分析 ===')
for i, row in df.iterrows():
    gdbh = row['gdbh']
    ldr = row['ldr']
    lrsj = str(row['lrsj'])[:19] if pd.notna(row['lrsj']) else ''
    blbm = row['blbm']
    content = str(row['zynr']) if pd.notna(row['zynr']) else ''
    print(f'--- 工单{gdbh} ---')
    print(f'来电: {ldr} @ {lrsj}')
    print(f'部门: {blbm}')
    print(f'内容: {content}')
    print()
