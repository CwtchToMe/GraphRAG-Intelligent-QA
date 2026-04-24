#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""完整功能测试"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd
import json
import time
from datetime import datetime
from core.llm_service import LLMService
from core.vector_store import VectorStore
from core.neo4j_kg import Neo4jKnowledgeGraph
from config import Config

print("=" * 70)
print(" GraphRAG 完整功能测试")
print(f" 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# 1. 读取Excel
print("\n[步骤1] 读取Excel文件...")
excel_path = r"d:\file_University\大三下\图机器学习\实验\ls_jingqutousu - 去隐私.xlsx"
df = pd.read_excel(excel_path)
print(f"[OK] Excel读取成功: {len(df)}行")

# 提取文本
documents = []
for idx, row in df.iterrows():
    text_parts = [str(row[col]) for col in df.columns if pd.notna(row[col])]
    text = ' '.join(text_parts)
    if len(text.strip()) > 10:
        documents.append({'id': str(idx), 'text': text.strip(), 'metadata': {'row': idx + 1}})

print(f"[OK] 提取文档: {len(documents)}条")

# 使用全部数据
test_docs = documents
print(f"[INFO] 使用全部 {len(test_docs)} 条数据")

# 2. 初始化LLM
print("\n[步骤2] 初始化LLM...")
llm = LLMService(
    api_key=Config.OPENAI_API_KEY,
    api_base=Config.OPENAI_API_BASE,
    model_name=Config.LLM_MODEL,
    local_model="qwen2.5:7b-instruct",
    temperature=0.1,
    max_tokens=2048
)

if not llm.initialize(prefer_local=True):
    print("[ERROR] LLM初始化失败")
    sys.exit(1)
print(f"[OK] LLM初始化成功")

# 3. 构建向量库
print("\n[步骤3] 构建向量库...")
vector_store = VectorStore(embedding_model=Config.EMBEDDING_MODEL, persist_directory="./test_chroma_db")
if not vector_store.initialize():
    print("[ERROR] 向量库初始化失败")
    sys.exit(1)

start_time = time.time()
chunks = [{'text': doc['text'], 'id': doc['id'], 'metadata': doc.get('metadata', {})} for doc in test_docs]
vs_count = vector_store.add_documents(chunks)
vs_elapsed = time.time() - start_time

vs_success = vs_count > 0
vs_stats = vector_store.get_statistics()
print(f"[OK] 向量库构建: {vs_count}条, 耗时{vs_elapsed:.1f}秒")

# 4. 构建知识图谱
print("\n[步骤4] 构建知识图谱...")
kg = Neo4jKnowledgeGraph(llm_service=llm)
kg.vector_store = vector_store

start_time = time.time()
kg_success = kg.build_knowledge_graph_from_documents(chunks=chunks, progress_callback=None)
kg_elapsed = time.time() - start_time

stats = kg.get_statistics()
print(f"[OK] 知识图谱构建完成")
print(f"     耗时: {kg_elapsed:.1f}秒")
print(f"     实体数: {stats.get('unique_entities', 0)}")
print(f"     关系数: {stats.get('total_relations', 0)}")

# 5. 显示结果
print("\n[步骤5] 提取结果...")

if kg.local_entities:
    print(f"\n实体列表 (共{len(kg.local_entities)}个):")
    for i, e in enumerate(kg.local_entities[:20]):
        print(f"  {i+1}. [{e.type}] {e.name}")
    if len(kg.local_entities) > 20:
        print(f"  ... 还有 {len(kg.local_entities) - 20} 个实体")

if kg.local_relations:
    print(f"\n关系列表 (共{len(kg.local_relations)}个):")
    for i, r in enumerate(kg.local_relations[:20]):
        print(f"  {i+1}. ({r.head}) --[{r.relation}]--> ({r.tail})")
    if len(kg.local_relations) > 20:
        print(f"  ... 还有 {len(kg.local_relations) - 20} 个关系")

# 6. 测试向量检索
print("\n[步骤6] 测试向量检索...")
results = vector_store.similarity_search("崂山风景区停车问题", top_k=3)
if results:
    print(f"[OK] 检索到 {len(results)} 条结果")

# 最终报告
print("\n" + "=" * 70)
print(" 测试报告")
print("=" * 70)
print(f"""
【测试数据】
  - Excel文件: ls_jingqutousu - 去隐私.xlsx
  - 总行数: {len(df)}
  - 测试文档数: {len(test_docs)}

【向量库构建】
  - 状态: {'成功' if vs_success else '失败'}
  - 耗时: {vs_elapsed:.1f}秒
  - 向量数: {vs_stats.get('total_documents', 0)}

【知识图谱构建】
  - 状态: {'成功' if kg_success else '失败'}
  - 耗时: {kg_elapsed:.1f}秒 ({kg_elapsed/60:.1f}分钟)
  - 实体数: {stats.get('unique_entities', 0)}
  - 关系数: {stats.get('total_relations', 0)}
  - 社团数: {stats.get('communities', 0)}
""")

if stats.get('unique_entities', 0) > 0:
    print(f"[OK] 测试成功！")
else:
    print(f"[ERROR] 未提取到实体！")
