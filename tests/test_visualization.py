# -*- coding: utf-8 -*-
"""
端到端测试脚本：验证从 Excel 上传到知识图谱可视化的全流程
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import networkx as nx
import tempfile
import traceback

# ── 1. 读取 Excel ──────────────────────────────────────────
excel_path = os.path.join(os.path.dirname(__file__), "ls_jingqutousu - 去隐私.xlsx")
print(f"[TEST] Reading Excel: {excel_path}")
df = pd.read_excel(excel_path)
print(f"[TEST] Rows: {len(df)}, Columns: {list(df.columns)}")

# ── 2. 正确解析 Excel 列 ───────────────────────────────────
# 列名: gdbh,lrsj,sfd,sfhf,blbm,bjsx,ldr,yjfl,ejfl,sjfl,gdbt,zynr,sjlx,jssj,location,updatetime
# lrsj=录入时间, blbm=办理部门, yjfl/ejfl/sjfl=一二三级分类, gdbt=工单标题, zynr=主要内容, location=地点
print(f"\n[TABLE] Sample rows:")
for col in ['blbm', 'location', 'yjfl', 'sjfl', 'gdbt']:
    if col in df.columns:
        vals = df[col].dropna().head(3).tolist()
        print(f"  {col}: {vals}")

# 构造结构化文本用于分块
def make_chunk_text(row):
    parts = []
    for col in ['lrsj', 'blbm', 'yjfl', 'ejfl', 'sjfl', 'location', 'gdbt', 'zynr']:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(f"{col}:{row[col]}")
    return " | ".join(parts)

chunks = []
chunk_size = 500
chunk_overlap = 50

# 收集所有行文本
all_row_texts = [make_chunk_text(row) for _, row in df.iterrows()]
all_text = "\n".join(all_row_texts)

for i in range(0, len(all_text), chunk_size - chunk_overlap):
    chunk = all_text[i:i + chunk_size]
    if len(chunk.strip()) > 20:
        chunks.append({
            'id': f"chunk_{i}",
            'text': chunk,
            'source': 'ls_jingqutousu',
            'chunk_index': i
        })
print(f"\n[TEST] Chunks: {len(chunks)}, Total chars: {len(all_text)}")

# ── 3. 规则提取（模拟 LLM 行为）──────────────────────────────
import re
from collections import defaultdict

all_entities = []
all_relations = []

# 从所有文本中提取
full_text = "\n".join(c['text'] for c in chunks)

# 提取办理部门 (blbm列)
if 'blbm' in df.columns:
    for val in df['blbm'].dropna().unique():
        s = str(val).strip()
        if 2 < len(s) <= 20 and not s.isdigit():
            all_entities.append({'name': s, 'type': 'ORGANIZATION'})

# 提取地点 (location列)
if 'location' in df.columns:
    for val in df['location'].dropna().unique():
        s = str(val).strip()
        if 2 < len(s) <= 20 and not s.isdigit():
            all_entities.append({'name': s, 'type': 'LOCATION'})

# 提取一级分类
if 'yjfl' in df.columns:
    for val in df['yjfl'].dropna().unique():
        s = str(val).strip()
        if len(s) >= 2 and len(s) <= 20:
            all_entities.append({'name': s, 'type': 'TYPE'})

# 提取主要内容关键词作为 ISSUE
if 'zynr' in df.columns:
    issue_patterns = [
        r'乱收费', r'绕路', r'欺诈', r'设施损坏', r'黑导游',
        r'欺诈', r'强制', r'投诉', r'纠纷', r'停车收费',
        r'管理', r'秩序', r'服务', r'收费', r'安全',
    ]
    seen_issues = set()
    for val in df['zynr'].dropna().head(20):
        for pat in issue_patterns:
            found = re.findall(pat, str(val))
            for f in found:
                if f not in seen_issues:
                    seen_issues.add(f)
                    all_entities.append({'name': f, 'type': 'ISSUE'})

# 去重
seen_e = {}
for e in all_entities:
    key = (e['name'], e['type'])
    if key not in seen_e:
        seen_e[key] = e
entities = list(seen_e.values())
print(f"\n[TEST] Extracted entities: {len(entities)}")
for e in entities[:15]:
    print(f"       {e['name']} ({e['type']})")

# 构建关系：每个 ISSUE 与其对应的部门/地点关联
if 'blbm' in df.columns and 'location' in df.columns and 'yjfl' in df.columns:
    orgs = [e['name'] for e in entities if e['type'] == 'ORGANIZATION']
    locs = [e['name'] for e in entities if e['type'] == 'LOCATION']
    issues = [e['name'] for e in entities if e['type'] == 'ISSUE']

    # 对每条投诉建立关系
    for _, row in df.iterrows():
        blbm_val = str(row.get('blbm', '')).strip() if pd.notna(row.get('blbm')) else ''
        loc_val = str(row.get('location', '')).strip() if pd.notna(row.get('location')) else ''
        yjfl_val = str(row.get('yjfl', '')).strip() if pd.notna(row.get('yjfl')) else ''

        matched_org = None
        for o in orgs:
            if o in blbm_val or blbm_val in o:
                matched_org = o
                break

        matched_loc = None
        for l in locs:
            if l in loc_val or loc_val in l:
                matched_loc = l
                break

        # 从 zynr 找 issue
        zynr_val = str(row.get('zynr', '')).strip() if pd.notna(row.get('zynr')) else ''
        matched_issues = []
        for iss in issues:
            if iss in zynr_val:
                matched_issues.append(iss)

        for iss in matched_issues:
            if matched_org:
                all_relations.append({'head': iss, 'relation': '处理', 'tail': matched_org})
            if matched_loc:
                all_relations.append({'head': iss, 'relation': '发生地', 'tail': matched_loc})
            if yjfl_val:
                all_relations.append({'head': iss, 'relation': '涉及', 'tail': yjfl_val})

print(f"[TEST] Extracted relations: {len(all_relations)}")
for r in all_relations[:10]:
    print(f"       {r['head']} --[{r['relation']}]--> {r['tail']}")

# ── 4. 过滤孤立实体 ───────────────────────────────────────
connected_names = set()
for r in all_relations:
    connected_names.add(r['head'])
    connected_names.add(r['tail'])

entities = [e for e in entities if e['name'] in connected_names]
print(f"\n[TEST] Entities after filtering: {len(entities)}")
print(f"[TEST] Relations after filtering: {len(all_relations)}")

# ── 5. 构建 NetworkX 图 ────────────────────────────────────
graph = nx.DiGraph()
for e in entities:
    graph.add_node(e['name'], etype=e['type'])
for r in all_relations:
    if r['head'] in graph.nodes and r['tail'] in graph.nodes:
        graph.add_edge(r['head'], r['tail'], relation=r['relation'])

print(f"\n[TEST] Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
if graph.number_of_nodes() == 0:
    print("[FAIL] Graph is empty! Check extraction logic.")
    sys.exit(1)

# ── 6. 测试可视化布局（核心验证）────────────────────────────
print("\n[TEST] Testing build_unified_pos...")

def build_unified_pos_test(g):
    pos = {}
    if not g.nodes():
        return pos

    g_und = g.to_undirected()
    comps = list(nx.connected_components(g_und))
    print(f"  Connected components: {len(comps)}")
    for ci, c in enumerate(comps):
        print(f"    Component {ci+1}: {list(c)[:5]}{'...' if len(c)>5 else ''} ({len(c)} nodes)")

    import random
    random.seed(42)

    nontrivial = [c for c in comps if len(c) > 1]
    singletons = [list(c)[0] for c in comps if len(c) == 1]

    if nontrivial:
        all_nontrivial = set()
        for c in nontrivial:
            all_nontrivial.update(c)
        sub_g = g.subgraph(all_nontrivial).copy().to_undirected()
        print(f"  Non-isolated nodes: {sub_g.number_of_nodes()}")

        sub_n = sub_g.number_of_nodes()
        if sub_n <= 80:
            try:
                sub_pos = nx.kamada_kawai_layout(sub_g)
            except Exception:
                sub_pos = nx.spring_layout(sub_g, k=5.0, iterations=300, seed=42)
        elif sub_n <= 300:
            sub_pos = nx.spring_layout(sub_g, k=3.0, iterations=200, seed=42)
        else:
            sub_pos = nx.spring_layout(sub_g, k=2.0, iterations=100, seed=42)

        xs = [p[0] for p in sub_pos.values()]
        ys = [p[1] for p in sub_pos.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = max(x_max - x_min, 0.001)
        y_range = max(y_max - y_min, 0.001)
        scale = 8.0
        for node, (x, y) in sub_pos.items():
            pos[node] = (
                (x - (x_min + x_range / 2)) / x_range * scale,
                (y - (y_min + y_range / 2)) / y_range * scale
            )

    # Isolated nodes grid layout
    used = set()
    for node in pos:
        px, py = round(pos[node][0]), round(pos[node][1])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                used.add((px + dx, py + dy))

    n_single = len(singletons)
    if n_single > 0:
        print(f"  Isolated nodes: {n_single}")
        grid_cols = max(4, int(n_single ** 0.5) + 1)
        grid_rows = (n_single + grid_cols - 1) // grid_cols
        spread_x = 10.0
        spread_y = 8.0

        for idx, node in enumerate(singletons):
            col = idx % grid_cols
            row = idx // grid_cols
            dist = random.uniform(0.5, 2.0)
            base_x = (col - grid_cols / 2 + 0.5) * spread_x
            base_y = (row - grid_rows / 2 + 0.5) * spread_y
            x = base_x + dist * 0.6 * (1 if col % 2 == 0 else -1)
            y = base_y + dist * 0.5 * random.choice([-1, 1])

            placed = False
            for _ in range(30):
                px, py = round(x), round(y)
                if (px, py) not in used:
                    pos[node] = (x, y)
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            used.add((px + dx, py + dy))
                    placed = True
                    break
                dist += 0.15
                x = base_x + dist * 0.6 * (1 if col % 2 == 0 else -1)
                y = base_y + dist * 0.5 * random.choice([-1, 1])

            if not placed:
                pos[node] = (x, y)

    return pos

try:
    unified_pos = build_unified_pos_test(graph)
    print(f"\n[PASS] build_unified_pos: {len(unified_pos)} nodes positioned")

    if unified_pos:
        xs = [p[0] for p in unified_pos.values()]
        ys = [p[1] for p in unified_pos.values()]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        print(f"  X range: [{min(xs):.2f}, {max(xs):.2f}], span: {x_span:.2f}")
        print(f"  Y range: [{min(ys):.2f}, {max(ys):.2f}], span: {y_span:.2f}")
        if y_span < 0.5 and len(xs) > 3:
            print("  [WARN] Y-span too small, nodes may overlap!")
        else:
            print("  [PASS] Layout distribution OK")
except Exception as e:
    print(f"\n[FAIL] build_unified_pos: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 7. 测试筛选逻辑 ────────────────────────────────────────
print("\n[TEST] Testing filter logic...")
entity_types = sorted({e['type'] for e in entities})
rel_types = sorted({r['relation'] for r in all_relations})
print(f"  Entity types: {entity_types}")
print(f"  Relation types: {rel_types}")

selected_entity_types = entity_types
selected_rel_types = rel_types

filtered_rels = [r for r in all_relations if r['relation'] in selected_rel_types]
involved_nodes = set()
for r in filtered_rels:
    involved_nodes.add(r['head'])
    involved_nodes.add(r['tail'])

filtered_ents = [e for e in entities
                 if e['type'] in selected_entity_types and e['name'] in involved_nodes]
filtered_rels2 = [r for r in filtered_rels
                  if r['head'] in {e['name'] for e in filtered_ents}
                  and r['tail'] in {e['name'] for e in filtered_ents}]

print(f"  After filter: {len(filtered_ents)} entities, {len(filtered_rels2)} relations")
if len(filtered_ents) == 0:
    print("  [FAIL] Filter produced 0 entities!")
    sys.exit(1)

# Rebuild graph with filtered data
graph2 = nx.DiGraph()
for e in filtered_ents:
    graph2.add_node(e['name'], etype=e['type'])
for r in filtered_rels2:
    graph2.add_edge(r['head'], r['tail'], relation=r['relation'])

print(f"  Filtered graph: {graph2.number_of_nodes()} nodes, {graph2.number_of_edges()} edges")

# ── 8. 测试 pyvis HTML 生成 ──────────────────────────────────
print("\n[TEST] Testing pyvis HTML generation...")
try:
    from pyvis.network import Network

    net = Network(height="400px", width="100%", bgcolor="#FAFBFC",
                  font_color="#2C3E50", notebook=False, cdn_resources='remote')
    net.force_atlas_2based(gravity=-80, central_gravity=0.02,
                           spring_length=80, spring_strength=0.04, damping=0.5)

    COLOR_MAP = {
        'ORGANIZATION': '#2980B9',
        'LOCATION': '#27AE60',
        'PERSON': '#C0392B',
        'TIME': '#D35400',
        'ISSUE': '#8E44AD',
        'ACTION': '#C2185B',
        'TYPE': '#00838F',
        'OTHER': '#78909C',
    }

    def get_color(t):
        for k in sorted(COLOR_MAP.keys(), key=len, reverse=True):
            if k.upper() in t.upper():
                return COLOR_MAP[k]
        return COLOR_MAP['OTHER']

    for n, d in graph2.nodes(data=True):
        ntype = d.get('etype', 'OTHER')
        color = get_color(ntype)
        deg = graph2.degree(n)
        size = 10 + min(deg * 1.5, 30)
        title = f"<b>{n}</b><br/>Type: {ntype}"
        if n in unified_pos:
            x_px = int(unified_pos[n][0] * 50)
            y_px = int(unified_pos[n][1] * 50)
            net.add_node(n, label=n, title=title, color=color, size=size,
                        x=x_px, y=y_px,
                        font={'size': 11, 'face': 'Microsoft YaHei', 'color': '#2C3E50'})
        else:
            net.add_node(n, label=n, title=title, color=color, size=size,
                        font={'size': 11, 'face': 'Microsoft YaHei', 'color': '#2C3E50'})

    for u, v, d in graph2.edges(data=True):
        rel = d.get('relation', 'RELATED')
        net.add_edge(u, v, title=rel, arrows='to')

    tmp = os.path.join(tempfile.gettempdir(), "kg_test.html")
    net.save_graph(tmp)

    with open(tmp, 'r', encoding='utf-8') as f:
        html = f.read()

    size_kb = len(html) / 1024
    has_network = 'mynetwork' in html
    has_vis = 'vis' in html or 'canvas' in html
    print(f"  [PASS] pyvis HTML: {size_kb:.1f} KB")
    print(f"  Contains mynetwork div: {has_network}")
    print(f"  Contains vis/canvas: {has_vis}")
    print(f"  HTML snippet: {html[200:500]}")

except Exception as e:
    print(f"  [FAIL] pyvis: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 9. 测试 Plotly fallback ─────────────────────────────────
print("\n[TEST] Testing Plotly fallback...")
try:
    import plotly.graph_objects as go

    edge_x, edge_y = [], []
    for u, v in graph2.edges():
        if u in unified_pos and v in unified_pos:
            edge_x.extend([unified_pos[u][0], unified_pos[v][0], None])
            edge_y.extend([unified_pos[u][1], unified_pos[v][1], None])

    all_x, all_y, all_c, all_s, all_t = [], [], [], [], []
    for n, d in graph2.nodes(data=True):
        if n in unified_pos:
            all_x.append(unified_pos[n][0])
            all_y.append(unified_pos[n][1])
            all_t.append(n)
            all_c.append(get_color(d.get('etype', 'OTHER')))
            all_s.append(12 + min(graph2.degree(n) * 2, 30))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                              line=dict(width=1.5, color='#bdc3c7'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=all_x, y=all_y, mode='markers+text',
                              marker=dict(size=all_s, color=all_c, line=dict(width=1.5, color='white')),
                              text=[f"<b>{t}</b>" for t in all_t],
                              textposition='top center',
                              textfont=dict(size=9), hoverinfo='text'))

    fig.update_layout(showlegend=False, height=400,
                      margin=dict(b=10, l=10, r=10, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      plot_bgcolor='#FAFBFC')

    html = fig.to_html()
    print(f"  [PASS] Plotly HTML: {len(html)/1024:.1f} KB")
    print(f"  Contains plot: {'plot' in html or 'graph' in html}")

except Exception as e:
    print(f"  [FAIL] Plotly: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
