"""
============================================================
GraphRAG 智能问答系统 - UI层（纯展示，无业务逻辑）
============================================================

架构设计：
- app.py: UI层（Streamlit界面）
- core/: 业务逻辑层
  - document.py: 文档处理
  - knowledge_graph.py: NetworkX基础图谱
  - neo4j_kg.py: Neo4j增强知识图谱（LLM提取+社团划分+综合查询）
  - llm_service.py: 大模型服务
  - retriever.py: 检索引擎（支持综合查询）
  - vector_store.py: 向量数据库

原则：高内聚、低耦合

v2.2 新增：
- Neo4j知识图谱集成
- 基于LLM的实体关系提取
- 多层级图谱（社团划分+抽象关键词）
- 图谱节点与向量库关联
- 综合查询方法（向量库+Neo4j融合）
"""
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
import json
import os
from pathlib import Path

import sys
sys.path.insert(0, '.')
from core.document import process_document, batch_process_documents
from core.knowledge_graph import KnowledgeGraph, create_sample_graph
from core.llm_service import LLMService
from core.retriever import Retriever
from core.vector_store import VectorStore
from core.neo4j_kg import Neo4jKnowledgeGraph, create_neo4j_knowledge_graph
from config import Config


st.set_page_config(
    page_title="GraphRAG 智能问答系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """初始化会话状态"""
    if 'kg' not in st.session_state:
        st.session_state.kg = KnowledgeGraph()
    
    if 'llm' not in st.session_state:
        st.session_state.llm = LLMService(
            api_key=Config.OPENAI_API_KEY,
            api_base=Config.OPENAI_API_BASE,
            model_name=Config.LLM_MODEL,
            local_model="qwen2.5:7b-instruct",
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        init_success = st.session_state.llm.initialize(prefer_local=True)
        
        if init_success:
            mode = st.session_state.llm.mode
            if mode == 'local':
                st.sidebar.success(f"[OK] LLM已启动 (本地模式 - {st.session_state.llm.local_model})")
            else:
                st.sidebar.success(f"[OK] LLM已启动 (API模式 - {Config.LLM_MODEL})")
        else:
            st.sidebar.warning("[WARN] LLM未初始化，知识图谱构建将不可用")
            st.sidebar.info("请确保 Ollama 服务正在运行 (ollama serve)")
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'neo4j_kg' not in st.session_state:
        st.session_state.neo4j_kg = None
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = Retriever(st.session_state.kg)
    
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []


init_session_state()


def render_knowledge_base_tab():
    """渲染知识库构建标签页"""
    st.header("知识库构建")

    st.subheader("步骤1: 上传文档")
    uploaded_files = st.file_uploader(
        "**支持格式**: PDF | TXT | DOCX | XLSX",
        type=['pdf', 'txt', 'docx', 'xlsx'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"已选择 {len(uploaded_files)} 个文件")

    st.divider()

    st.subheader("步骤2: 配置参数")
    col1, col2 = st.columns(2)

    with col1:
        chunk_size = st.slider("文本块大小", 100, 2000, 500, 100)
        chunk_overlap = st.slider("重叠大小", 0, 500, 50, 10)

    with col2:
        show_details = st.checkbox("显示详细进度", value=True)
        use_llm_extraction = st.checkbox("使用LLM提取实体关系", value=True)
        build_kg_graph = st.checkbox("构建知识图谱", value=True)

    st.divider()

    st.subheader("步骤2.5: 知识图谱模式")
    prompt_mode = st.radio(
        "选择知识图谱构建模式",
        options=["通用模式", "投诉模式"],
        captions=[
            "适用于书籍介绍、技术文档、新闻报道等任何领域的文本",
            "专门针对崂山景区投诉工单数据优化（预置实体/关系类型和别名映射）"
        ],
        horizontal=True,
        index=0
    )

    st.divider()

    st.subheader("步骤3: 构建知识库")

    col_build, col_sample = st.columns(2)

    with col_build:
        if st.button("开始构建", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("请先上传文档")
            else:
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                try:
                    status_placeholder.info("正在初始化向量数据库...")

                    vector_store = VectorStore(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

                    init_success = vector_store.initialize()
                    if not init_success:
                        status_placeholder.error("向量数据库初始化失败，请检查依赖安装")
                        st.stop()

                    progress_bar.progress(0.1)

                    total_chunks = 0
                    all_triples = []

                    for idx, file in enumerate(uploaded_files):
                        status_placeholder.info(f"处理文件 [{idx+1}/{len(uploaded_files)}]: {file.name}")

                        chunks = vector_store.load_document(uploaded_file=file)
                        count = vector_store.add_documents(chunks)
                        total_chunks += count

                        progress_bar.progress(0.1 + (idx + 1) / len(uploaded_files) * 0.3)

                    st.session_state.vector_store = vector_store
                    st.session_state.documents = vector_store.documents

                    progress_bar.progress(0.4)

                    kg_stats = None
                    if build_kg_graph:
                        if not st.session_state.llm.is_initialized:
                            status_placeholder.warning("LLM未初始化，正在尝试连接...")
                            init_ok = st.session_state.llm.initialize(prefer_local=True)
                            if not init_ok:
                                status_placeholder.error("LLM初始化失败！请确保Ollama服务正在运行")
                                st.info("请运行: ollama serve")
                                st.stop()

                        status_placeholder.info(
                            f"正在构建知识图谱...\n"
                            f"  文档块总数: {len(vector_store.documents)}\n"
                            f"  预估耗时: 30秒-2分钟"
                        )

                        neo4j_kg = Neo4jKnowledgeGraph(
                            uri="bolt://localhost:7687",
                            user="neo4j",
                            password="password",
                            llm_service=st.session_state.llm if use_llm_extraction else None
                        )
                        neo4j_kg.set_prompt_mode("universal" if prompt_mode == "通用模式" else "complaint")

                        neo4j_kg.connect()
                        neo4j_kg.set_vector_store(vector_store)

                        def update_progress(current, total, message):
                            progress = 0.4 + (current / total) * 0.5
                            progress_bar.progress(min(progress, 0.9))
                            if show_details:
                                status_placeholder.info(message)

                        kg_stats = neo4j_kg.build_knowledge_graph_from_documents(
                            chunks=vector_store.documents,
                            progress_callback=update_progress if show_details else None
                        )

                        st.session_state.neo4j_kg = neo4j_kg

                        # 将 Neo4j 图谱数据同步到 session_state.kg（供 fallback 可视化使用）
                        if neo4j_kg.local_entities and neo4j_kg.local_relations:
                            for rel in neo4j_kg.local_relations:
                                st.session_state.kg.add_triples([{
                                    'head': rel.head,
                                    'relation': rel.relation,
                                    'tail': rel.tail
                                }])

                    st.session_state.retriever = Retriever(
                        st.session_state.kg,
                        vector_store,
                        neo4j_kg=st.session_state.neo4j_kg
                    )

                    progress_bar.progress(1.0)
                    status_placeholder.empty()

                    st.success("构建完成！")

                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("文档数", len(uploaded_files))
                    with col_stat2:
                        st.metric("文本块", total_chunks)
                    with col_stat3:
                        vs_stats = vector_store.get_statistics()
                        st.metric("向量数", vs_stats.get('total_documents', 0))
                    with col_stat4:
                        if kg_stats:
                            st.metric("实体数", kg_stats.get('unique_entities', 0))
                        else:
                            st.metric("知识图谱", "未构建")

                    if show_details and kg_stats:
                        with st.expander("知识图谱详情"):
                            st.json(kg_stats)
                        elapsed = kg_stats.get('elapsed_seconds', 0)
                        total_input = kg_stats.get('total_chunks_input', 0)
                        llm_calls = kg_stats.get('llm_calls_made', 1)

                        if total_input > 0:
                            st.info(
                                f"构建完成！\n"
                                f"  处理文档块: {total_input} 个\n"
                                f"  LLM调用次数: {llm_calls} 次\n"
                                f"  实际耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)"
                            )

                except Exception as e:
                    import traceback
                    tb_str = traceback.format_exc()
                    status_placeholder.error(f"构建失败: {str(e)}")
                    with st.expander("错误详情（技术信息）", expanded=True):
                        st.code(tb_str, language="python")

    with col_sample:
        if st.button("加载示例数据", use_container_width=True):
            with st.spinner("加载中..."):
                sample_kg = create_sample_graph()
                st.session_state.kg = sample_kg
                st.session_state.retriever = Retriever(sample_kg)

                stats = sample_kg.get_statistics()
                st.success(f"已加载示例数据！")
                st.metric("节点数", stats['nodes'])
                st.metric("边数", stats['edges'])


def render_qa_tab():
    """渲染智能问答标签页"""
    st.header("基于大模型的智能问答")
    
    st.subheader("步骤1: 配置大模型API")
    
    with st.expander("API配置", expanded=not st.session_state.llm.is_initialized):
        col_cfg1, col_cfg2 = st.columns([3, 2])
        
        with col_cfg1:
            api_key = st.text_input("**API Key**",
                value=st.session_state.llm.api_key,
                type="password")
            
            api_base = st.text_input("**API Base URL**",
                value=st.session_state.llm.api_base)
            
            model_name = st.selectbox("**模型**",
                options=['deepseek-chat', 'gpt-3.5-turbo', 'gpt-4'],
                index=0)
        
        with col_cfg2:
            temperature = st.slider("**Temperature**", 0.0, 1.0, 
                                     st.session_state.llm.temperature, 0.1)
            
            if st.button("测试连接", use_container_width=True, type="primary"):
                st.session_state.llm.update_config(
                    api_key=api_key,
                    api_base=api_base,
                    model_name=model_name,
                    temperature=temperature
                )
                
                success, message = st.session_state.llm.test_connection()
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    st.divider()
    
    st.subheader("步骤2: 输入问题")
    
    user_query = st.text_area("**您的问题**:",
        placeholder="例如：《大话西游》的导演是谁？",
        height=80)
    
    # ── 检索模式 ──
    retrieval_mode = st.radio(
        "**检索方式**",
        options=["知识图谱", "向量检索", "综合查询"],
        captions=[
            "仅使用知识图谱结构化检索",
            "仅使用向量语义相似度检索",
            "知识图谱 + 向量融合（两路并行，取长补短）"
        ],
        horizontal=True,
        index=0,
        label_visibility="collapsed"
    )

    use_llm = st.checkbox("使用大模型生成答案", value=True)

    mode_label = {
        "知识图谱": "纯知识图谱检索",
        "向量检索": "纯向量检索",
        "综合查询": "综合查询（图谱+向量）",
    }[retrieval_mode]
    st.caption(f"当前模式: {mode_label}")
    
    if st.button("提交问题", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("请输入问题")
        elif not st.session_state.documents and not st.session_state.kg.triples:
            st.warning("请先上传文档并构建知识库")
        elif retrieval_mode == "知识图谱" and not st.session_state.neo4j_kg and not st.session_state.kg.triples:
            st.warning("请先构建知识图谱（切换到「知识库构建」标签页）")
        elif use_llm and not st.session_state.llm.is_initialized:
            st.warning("请先配置API并测试连接")
        else:
            with st.spinner("检索和生成中..."):
                # ── Step 1: 执行检索 ──────────────────────────────
                if retrieval_mode == "知识图谱":
                    # 单独使用知识图谱，不走向量
                    retrieval_result = st.session_state.retriever.hybrid_retrieve(
                        user_query,
                        st.session_state.documents,
                        use_vector=False,
                        use_graph=True
                    )
                    context = st.session_state.retriever.format_context(retrieval_result)
                    source = "[知识图谱检索]"

                elif retrieval_mode == "综合查询":
                    # 综合查询：同时使用知识图谱 + 向量
                    if st.session_state.neo4j_kg:
                        retrieval_result = st.session_state.retriever.comprehensive_query(
                            user_query,
                            documents=st.session_state.documents
                        )
                        context = st.session_state.retriever.format_comprehensive_context(retrieval_result)
                        has_graph = bool(retrieval_result.get('graph_results'))
                        has_vector = bool(retrieval_result.get('vector_results'))
                        if has_graph and has_vector:
                            source = "[综合查询]（图谱+向量）"
                        elif has_graph:
                            source = "[综合查询]（图谱）"
                        else:
                            source = "[综合查询]（向量）"
                    else:
                        # 没有 Neo4j KG，走混合检索
                        retrieval_result = st.session_state.retriever.hybrid_retrieve(
                            user_query,
                            st.session_state.documents,
                            use_vector=True,
                            use_graph=True
                        )
                        context = st.session_state.retriever.format_context(retrieval_result)
                        has_graph = bool(retrieval_result.get('graph_results'))
                        has_vector = bool(retrieval_result.get('vector_results'))
                        if has_graph and has_vector:
                            source = "[综合查询]（图谱+向量）"
                        elif has_graph:
                            source = "[综合查询]（图谱）"
                        else:
                            source = "[综合查询]（向量）"

                else:
                    # 单独使用向量检索，不走图谱
                    retrieval_result = st.session_state.retriever.hybrid_retrieve(
                        user_query,
                        st.session_state.documents,
                        use_vector=True,
                        use_graph=False
                    )
                    context = st.session_state.retriever.format_context(retrieval_result)
                    source = "[向量检索]"

                # ── Step 2: LLM 生成答案 ─────────────────────────
                if use_llm and st.session_state.llm.is_initialized:
                    try:
                        answer = st.session_state.llm.generate_answer(user_query, context)
                        source = "[大模型生成]"
                    except Exception as e:
                        answer = f"生成失败: {str(e)}\n\n原始结果:\n{context}"
                        source = "[生成失败]"
                else:
                    # 不使用LLM时，直接展示检索到的内容
                    if retrieval_result.get('graph_results'):
                        answer = "根据知识图谱:\n\n"
                        for r in retrieval_result['graph_results'][:3]:
                            answer += f"- {r['head']} {r['relation']} {r['tail']}\n"
                    elif retrieval_result.get('vector_results'):
                        answer = "根据向量检索结果:\n\n"
                        for r in retrieval_result['vector_results'][:3]:
                            text = r.get('content', '')[:200]
                            answer += f"- {text}\n\n"
                    else:
                        answer = "未找到相关信息。"
                        source = "[无结果]"

                st.session_state.conversation_history.append({
                    'question': user_query,
                    'answer': answer,
                    'source': source,
                    'graph_count': retrieval_result.get('graph_count', len(retrieval_result.get('graph_results', []))),
                    'vector_count': retrieval_result.get('vector_count', len(retrieval_result.get('vector_results', []))),
                    'retrieval_result': retrieval_result,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })

                st.divider()
                st.subheader("答案")
                st.markdown(answer)
                st.caption(f"来源: {source}")

                # ────────── P1-3: 溯源可视化区域 ──────────
                with st.expander("查看知识溯源与推理过程", expanded=True):
                    tabs_src = st.tabs(["推理链路径", "相关三元组", "原始文档片段"])

                    # Tab 1: 推理链路径
                    with tabs_src[0]:
                        chain = retrieval_result.get('reasoning_chain', [])
                        if chain:
                            st.success(f"检测到多跳推理！共 {len(chain)} 条推理路径")
                            step_groups = {}
                            for step in chain:
                                s = step.get('step', 1)
                                if s not in step_groups:
                                    step_groups[s] = []
                                step_groups[s].append(step)
                            for s, steps in sorted(step_groups.items()):
                                st.markdown(f"**第 {s} 跳（共 {len(steps)} 条）**")
                                for step_item in steps:
                                    st.markdown(
                                        f"  {step_item.get('from', '?')} "
                                        f"--[{step_item.get('relation', '?')}]--> "
                                        f"{step_item.get('to', '?')}"
                                    )
                        else:
                            st.info("无需多跳推理，使用直接检索结果")

                        # 展示子图（如果有）
                        sub_nodes = retrieval_result.get('subgraph_nodes', [])
                        sub_edges = retrieval_result.get('subgraph_edges', [])
                        if sub_nodes and sub_edges:
                            st.markdown("**推理相关子图：**")
                            import networkx as nx
                            G_sub = nx.DiGraph()
                            for n in sub_nodes:
                                G_sub.add_node(n.get('id', ''))
                            for e in sub_edges:
                                G_sub.add_edge(
                                    e.get('from', ''), e.get('to', ''),
                                    label=e.get('label', '')
                                )
                            if G_sub.number_of_nodes() > 0:
                                pos_sub = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)
                                node_x, node_y, node_texts = [], [], []
                                for node in G_sub.nodes():
                                    node_x.append(pos_sub[node][0])
                                    node_y.append(pos_sub[node][1])
                                    node_texts.append(node)
                                edge_x, edge_y, edge_labels = [], [], []
                                for u, v, d in G_sub.edges(data=True):
                                    x0, y0 = pos_sub[u]
                                    x1, y1 = pos_sub[v]
                                    edge_x += [x0, x1, None]
                                    edge_y += [y0, y1, None]
                                    edge_labels.append(d.get('label', ''))

                                fig_sub = go.Figure()
                                fig_sub.add_trace(go.Scatter(
                                    x=edge_x, y=edge_y, mode='lines',
                                    line=dict(width=1.5, color='#888'),
                                    hoverinfo='none'
                                ))
                                fig_sub.add_trace(go.Scatter(
                                    x=node_x, y=node_y, mode='markers+text',
                                    marker=dict(size=20, color='#3498DB'),
                                    text=node_texts, textposition="top center",
                                    textfont=dict(size=11), hoverinfo='text'
                                ))
                                fig_sub.update_layout(
                                    title="推理路径子图",
                                    width=600, height=350,
                                    showlegend=False,
                                    xaxis=dict(visible=False),
                                    yaxis=dict(visible=False)
                                )
                                st.plotly_chart(fig_sub, use_container_width=True)
                            else:
                                st.info("子图节点不足，无法渲染")

                    # Tab 2: 相关三元组
                    with tabs_src[1]:
                        graph_res = retrieval_result.get('graph_results', [])
                        if graph_res:
                            rows = []
                            for g in graph_res[:10]:
                                path_str = g.get('path', '')
                                if '->' in path_str:
                                    parts = path_str.split('->')
                                    if len(parts) >= 3:
                                        rows.append({
                                            '头实体': parts[0].strip(),
                                            '关系': g.get('relations', [''])[0] if g.get('relations') else '',
                                            '尾实体': parts[-1].strip(),
                                            '跳数': g.get('length', 1)
                                        })
                                    elif len(parts) == 2:
                                        rows.append({
                                            '头实体': parts[0].strip(),
                                            '关系': g.get('relations', [''])[0] if g.get('relations') else '',
                                            '尾实体': parts[1].strip(),
                                            '跳数': 1
                                        })
                                elif 'head' in g:
                                    rows.append({
                                        '头实体': g.get('head', ''),
                                        '关系': g.get('relation', ''),
                                        '尾实体': g.get('tail', ''),
                                        '跳数': 1
                                    })
                            if rows:
                                import pandas as pd
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                            else:
                                st.info("当前检索结果中无三元组详情")
                        else:
                            st.info("无图谱检索结果")

                    # Tab 3: 原始文档片段
                    with tabs_src[2]:
                        vec_res = retrieval_result.get('vector_results', [])
                        if vec_res:
                            for i, v in enumerate(vec_res[:5], 1):
                                score = v.get('score', 0)
                                content = v.get('content', '')[:300]
                                src = v.get('source', '未知来源')
                                st.markdown(
                                    f"**{i}.** [{score:.3f}] *({src})*\n> {content}..."
                                )
                                st.divider()
                        else:
                            st.info("无向量检索结果")
    
    st.divider()
    
    if st.session_state.conversation_history:
        st.subheader("步骤3: 对话历史")
        
        for conv in reversed(st.session_state.conversation_history[-5:]):
            with st.expander(f"Q: {conv['question'][:60]}..."):
                st.markdown(f"**问题:** {conv['question']}")
                st.markdown(f"**答案:** {conv['answer'][:300]}...")
                st.caption(
                    f"{conv['source']} | 图谱:{conv.get('graph_count', 0)} | "
                    f"向量:{conv.get('vector_count', 0)} | {conv['timestamp']}"
                )


def render_prompt_tab():
    """渲染Prompt模板标签页"""
    st.header("核心Prompt模板")
    
    prompts = st.session_state.llm.get_prompts()
    
    st.subheader("1. 系统Prompt（问答生成）")
    st.code(prompts['system'], language="markdown")
    
    st.divider()
    
    st.subheader("2. NER Prompt（实体识别）")
    st.code(prompts['ner'], language="markdown")
    
    st.divider()
    
    st.subheader("3. RE Prompt（关系抽取）")
    st.code(prompts['re'], language="markdown")


def render_visualization_tab():
    """渲染知识图谱可视化 - 单一画布、完整布局、支持缩放拖拽"""
    import networkx as nx

    st.header("知识图谱可视化溯源")

    # ─── 获取数据 ───
    if st.session_state.neo4j_kg and (st.session_state.neo4j_kg.is_connected or st.session_state.neo4j_kg.local_entities):
        kg = st.session_state.neo4j_kg
        entities = kg.local_entities
        relations = kg.local_relations
    elif st.session_state.kg and st.session_state.kg.triples:
        name_to_type = {}
        for t in st.session_state.kg.triples:
            for key in ('head', 'tail'):
                val = t.get(key)
                if val and val not in name_to_type:
                    name_to_type[val] = 'OTHER'
        from core.neo4j_kg import Entity, Relation
        entities = [Entity(name=k, type=v) for k, v in name_to_type.items()]
        relations = [Relation(head=t['head'], relation=t.get('relation', 'RELATED_TO'), tail=t['tail'])
                     for t in st.session_state.kg.triples]
    else:
        st.info("请先上传 Excel 文件并构建知识图谱")
        return

    if not entities:
        st.warning("当前图谱没有实体数据")
        return

    # ─── 颜色映射 ───
    # ─── 根据 Prompt 模式选择颜色方案 ───
    prompt_mode = "universal"
    if st.session_state.neo4j_kg:
        prompt_mode = getattr(st.session_state.neo4j_kg, '_prompt_mode', 'universal')
    elif st.session_state.kg and st.session_state.kg.triples:
        prompt_mode = "universal"

    if prompt_mode == "complaint":
        COLOR_MAP = {
            'COMPLAINT':  '#1A237E',
            'PERSON':      '#C0392B',
            'LOCATION':    '#27AE60',
            'ORGANIZATION': '#2980B9',
            'ISSUE':       '#8E44AD',
            'EVENT':       '#E67E22',
            'VEHICLE':     '#16A085',
            'TYPE':        '#00838F',
            'AREA':        '#6A1B9A',
            'OTHER':       '#78909C',
        }
        REL_COLORS = {
            '处理':    '#2980B9', '涉及部门': '#2980B9',
            '发生地':  '#27AE60',
            '反映人':  '#8E44AD',
            '具体事件':'#E67E22',
            '涉及问题':'#C0392B',
            '乘坐':    '#16A085',
            '属于':    '#00838F',
            '区域':    '#6A1B9A',
            '关联':    '#95A5A6',
        }
    else:
        COLOR_MAP = {
            'PERSON':       '#E74C3C',
            'ORGANIZATION': '#3498DB',
            'LOCATION':     '#27AE60',
            'EVENT':        '#E67E22',
            'CONCEPT':      '#9B59B6',
            'WORK':         '#F39C12',
            'PRODUCT':      '#1ABC9C',
            'TECHNOLOGY':   '#2980B9',
            'TIME':         '#D35400',
            'ABSTRACT':     '#8E44AD',
            'OBJECT':       '#16A085',
            'AWARD':        '#F1C40F',
            'LAW':          '#7F8C8D',
            'MONEY':        '#27AE60',
            'NATIONALITY':   '#E91E63',
            'MOVEMENT':     '#00BCD4',
            'DISEASE':      '#C0392B',
            'DRUG':         '#E67E22',
            'OTHER':        '#78909C',
        }
        REL_COLORS = {
            '创作': '#E74C3C', '主演': '#E74C3C', '导演': '#E74C3C',
            '编剧': '#E74C3C', '属于': '#3498DB', '类型': '#3498DB',
            '发生于': '#27AE60', '位于': '#27AE60',
            '导致': '#E67E22', '引起': '#E67E22', '影响': '#E67E22',
            '参与': '#9B59B6', '任职于': '#3498DB', '创始人': '#E74C3C',
            '获奖': '#F1C40F', '使用': '#2980B9', '应用于': '#2980B9',
            '毕业于': '#27AE60', '研究': '#9B59B6',
            '描写': '#F39C12', '讲述': '#F39C12',
            '合作': '#16A085', '竞争': '#C0392B', '对立': '#C0392B',
            '相似': '#95A5A6', '包含': '#95A5A6',
            '发行': '#E67E22', '翻译': '#2980B9',
            '代表': '#8E44AD', '改进': '#27AE60',
            '关联': '#95A5A6',
        }

    def get_color(t):
        for k in sorted(COLOR_MAP.keys(), key=len, reverse=True):
            if k.upper() in t.upper():
                return COLOR_MAP[k]
        return COLOR_MAP['OTHER']

    def get_rel_color(r):
        for k in sorted(REL_COLORS.keys(), key=len, reverse=True):
            if k.upper() in r.upper():
                return REL_COLORS[k]
        return '#95A5A6'

    # ─── 筛选控制 ───
    entity_types = sorted({e.type for e in entities})
    rel_types = sorted({r.relation for r in relations})

    with st.expander("筛选与布局控制", expanded=True):
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            selected_entity_types = st.multiselect(
                "实体类型", entity_types, default=entity_types)
        with col_c2:
            selected_rel_types = st.multiselect(
                "关系类型", rel_types, default=rel_types)

    # ─── 构建全量 NetworkX 图 ───
    all_graph = nx.DiGraph()
    for e in entities:
        all_graph.add_node(e.name, etype=e.type)
    for r in relations:
        if r.head in all_graph.nodes and r.tail in all_graph.nodes:
            all_graph.add_edge(r.head, r.tail, relation=r.relation)

    # ─── 按筛选条件过滤 ───
    # 第一步：按关系类型过滤
    filtered_rels = [r for r in relations if r.relation in selected_rel_types]
    involved_nodes = set()
    for r in filtered_rels:
        involved_nodes.add(r.head)
        involved_nodes.add(r.tail)

    # 第二步：按实体类型过滤
    filtered_ents = [e for e in entities
                     if e.type in selected_entity_types and e.name in involved_nodes]

    # 第三步：过滤后重新得到关系（头尾都在 filtered_ents 里）
    filtered_rels2 = [r for r in filtered_rels
                      if r.head in {e.name for e in filtered_ents}
                      and r.tail in {e.name for e in filtered_ents}]

    # 构建筛选后的图
    graph = nx.DiGraph()
    name_to_etype = {e.name: e.type for e in filtered_ents}
    for e in filtered_ents:
        graph.add_node(e.name, etype=e.type)
    for r in filtered_rels2:
        graph.add_edge(r.head, r.tail, relation=r.relation)

    n_entities = graph.number_of_nodes()
    n_relations = graph.number_of_edges()
    undirected = graph.to_undirected()

    if n_entities == 0:
        st.warning("当前筛选条件下没有实体数据，请调整筛选条件")
        return

    try:
        components = sorted(nx.connected_components(undirected), key=len, reverse=True)
    except Exception:
        components = [set(graph.nodes())]

    st.caption(f"展示 {n_entities} 个有关联实体，{n_relations} 条关系，共 {len(components)} 个连通簇（孤立节点已排除）")

    # ─── 统一布局：分量级 spring 布局 + 分量网格排列 ───
    import random
    random.seed(42)

    def build_unified_pos(g):
        """每个连通分量独立 spring 布局，再按网格排列到全局坐标"""
        pos = {}
        if not g.nodes():
            return pos

        g_und = g.to_undirected()
        comps = list(nx.connected_components(g_und))
        # 只处理有边的分量，孤立节点直接忽略
        comps = [c for c in comps if len(c) > 1]
        if not comps:
            return pos

        # ── 第一步：每个分量内部独立布局 ──
        comp_centers = []   # 每个分量的 (节点x列表, 节点y列表)
        comp_positions = []  # 每个分量的 {node: (x,y)} 列表

        for c in comps:
            sub = g.subgraph(c).to_undirected()
            n = len(c)
            # spring 参数随节点数调整
            k = min(3.0, 0.5 + n * 0.05)
            iters = max(50, 200 - n)
            try:
                sub_pos = nx.kamada_kawai_layout(sub)
            except Exception:
                sub_pos = nx.spring_layout(sub, k=k, iterations=iters, seed=42)

            xs = [p[0] for p in sub_pos.values()]
            ys = [p[1] for p in sub_pos.values()]
            rng_x = max(xs) - min(xs) if len(set(xs)) > 1 else 1.0
            rng_y = max(ys) - min(ys) if len(set(ys)) > 1 else 1.0
            rng = max(rng_x, rng_y, 0.001)

            # 归一化到 [-1, 1]
            cx = (max(xs) + min(xs)) / 2
            cy = (max(ys) + min(ys)) / 2
            norm_pos = {
                node: ((x - cx) / rng, (y - cy) / rng)
                for node, (x, y) in sub_pos.items()
            }
            comp_positions.append(norm_pos)
            comp_centers.append(cx)

        # ── 第二步：将各分量排列到网格 ──
        n_comp = len(comp_positions)
        cols = max(1, int(n_comp ** 0.5))
        rows = (n_comp + cols - 1) // cols
        gap_x, gap_y = 5.0, 4.0  # 分量之间的间距

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n_comp:
                    break
                cpos = comp_positions[idx]
                # 全局坐标 = 网格偏移 + 分量内相对位置 * 分量大小
                ox = (c - cols / 2) * gap_x
                oy = (r - rows / 2) * gap_y
                for node, (x, y) in cpos.items():
                    pos[node] = (ox + x * 2.5, oy + y * 2.5)
                idx += 1

        return pos

    unified_pos = build_unified_pos(graph)

    # ─── 图例 ───
    shown_types = sorted({d.get('etype', 'OTHER') for _, d in graph.nodes(data=True)})
    legend_items = "  ".join(
        f'<span style="display:inline-block;width:14px;height:14px;background:{get_color(t)};'
        f'border-radius:50%;margin-right:4px;vertical-align:middle;"></span>{t}'
        for t in shown_types
    )
    st.markdown(
        f"<div style='font-size:12px;color:#555;padding:4px 0;'>{legend_items}</div>",
        unsafe_allow_html=True
    )

    # ─── 可视化（Plotly 静态）──────────────────
    canvas_h = min(max(600, n_entities * 6), 900)
    components = list(nx.connected_components(graph.to_undirected()))

    import plotly.graph_objects as go

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        if u in unified_pos and v in unified_pos:
            edge_x.extend([unified_pos[u][0], unified_pos[v][0], None])
            edge_y.extend([unified_pos[u][1], unified_pos[v][1], None])

    all_x, all_y, all_c, all_s, all_t, all_etype = [], [], [], [], [], []
    for n, d in graph.nodes(data=True):
        if n in unified_pos:
            all_x.append(unified_pos[n][0])
            all_y.append(unified_pos[n][1])
            all_t.append(n)
            all_etype.append(d.get('etype', 'OTHER'))
            all_c.append(get_color(d.get('etype', 'OTHER')))
            all_s.append(12 + min(graph.degree(n) * 2, 30))

    fig = go.Figure()

    # 边
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1.5, color='#bdc3c7'),
        hoverinfo='none',
        showlegend=False
    ))

    # 节点
    fig.add_trace(go.Scatter(
        x=all_x, y=all_y,
        mode='markers+text',
        marker=dict(
            size=all_s,
            color=all_c,
            line=dict(width=1.5, color='white')
        ),
        text=[f"<b>{t}</b>" for t in all_t],
        textposition='top center',
        textfont=dict(size=9, family='Microsoft YaHei', color='#2C3E50'),
        hoverinfo='text',
        hovertext=[f"<b>{t}</b><br/>类型: {e}<br/>度数: {graph.degree(t)}"
                   for t, e in zip(all_t, all_etype)],
        showlegend=False
    ))

    fig.update_layout(
        showlegend=False,
        height=canvas_h,
        margin=dict(b=10, l=10, r=10, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[min(all_x)-2 if all_x else -5, max(all_x)+2 if all_x else 5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[min(all_y)-2 if all_y else -5, max(all_y)+2 if all_y else 5]),
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='#FAFBFC',
        annotations=[
            dict(
                text=f"{n_entities} 实体 | {n_relations} 关系 | {len(components)} 连通分量",
                x=0.5, y=1.02,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=11, color='#555'),
                xanchor='center'
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── 三元组表格 ───
    st.divider()
    st.subheader("三元组列表")
    rows = [
        {'头实体': r.head, '关系': r.relation, '尾实体': r.tail}
        for r in filtered_rels2
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True,
                 height=min(400, 28 * min(len(rows), 18) + 60))

    # ─── 统计 ───
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("实体数", n_entities)
    with c2: st.metric("关系数", n_relations)
    with c3:
        d = nx.density(graph) if n_entities > 1 else 0
        st.metric("密度", f"{d:.3f}")
    with c4: st.metric("连通分量", len(components))
    with c5: st.metric("实体类型", len(shown_types))



def render_test_cases_tab():
    """渲染测试案例标签页"""
    st.header("测试案例对比")
    
    st.info("对比纯向量RAG与GraphRAG的效果差异")
    
    st.subheader("案例1: 简单查询")
    st.markdown("""
| 方法 | 答案 | 准确性 |
|------|------|--------|
| **纯向量RAG** | 可能是刘镇伟 | 3星 |
| **GraphRAG** | 刘镇伟 | 5星 |

**推理过程**: 用户问题 -> 实体识别 -> 图谱查询 -> 答案
""")
    
    st.divider()
    
    st.subheader("案例2: 多跳推理")
    st.markdown("""
| 方法 | 答案 | 准确性 |
|------|------|--------|
| **纯向量RAG** | 无法准确回答 | 1星 |
| **GraphRAG** | 《都市情缘》、《天长地久》等 | 5星 |

**推理过程**:
```
步骤1: 查询《大话西游》导演 -> 刘镇伟
步骤2: 查询刘镇伟的其他作品 -> 都市情缘...
```
""")
    
    st.divider()
    
    st.subheader("案例3: 边界案例")
    st.markdown("""
| 方法 | 答案 | 准确性 |
|------|------|--------|
| **纯向量RAG** | 可能编造票房数据 | 1星 |
| **GraphRAG** | 无法回答此问题 | 5星 |

**优势**: GraphRAG能诚实地说"我不知道"，避免幻觉
""")
    
    st.divider()
    
    st.subheader("功能对比总结")
    
    comparison_data = {
        '能力维度': ['简单查询', '多跳推理', '避免幻觉', '知识溯源', '可解释性'],
        '纯向量RAG': ['3星', '1星', '1星', '困难', '低'],
        'GraphRAG': ['5星', '5星', '5星', '清晰', '高']
    }
    
    st.dataframe(comparison_data, use_container_width=True)


def main():
    st.title("GraphRAG 智能问答系统")
    st.markdown("基于**大模型+知识图谱**的增强型RAG问答系统")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "知识库构建",
        "智能问答",
        "Prompt模板",
        "知识图谱可视化",
        "测试案例"
    ])
    
    with tab1:
        render_knowledge_base_tab()
    
    with tab2:
        render_qa_tab()
    
    with tab3:
        render_prompt_tab()
    
    with tab4:
        render_visualization_tab()
    
    with tab5:
        render_test_cases_tab()
    
    st.divider()
    
    st.subheader("系统状态监控")
    
    # 统一显示Neo4j知识图谱的数据（如果有的话）
    if st.session_state.neo4j_kg and (st.session_state.neo4j_kg.is_connected or st.session_state.neo4j_kg.local_entities):
        kg_stats = st.session_state.neo4j_kg.get_statistics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("文档数", len(st.session_state.documents))
        
        with col2:
            st.metric("实体数", kg_stats.get('unique_entities', 0))
        
        with col3:
            st.metric("关系数", kg_stats.get('total_relations', 0))
        
        with col4:
            st.metric("社团数", kg_stats.get('communities', 0))
        
        with col5:
            if st.session_state.llm.is_initialized:
                st.metric("LLM", "[OK] 已连接")
            else:
                st.metric("LLM", "[--] 未配置")
        
        # 向量库状态
        if st.session_state.vector_store and st.session_state.vector_store.is_initialized:
            vs_stats = st.session_state.vector_store.get_statistics()
            st.metric("向量库", f"[OK] {vs_stats.get('total_documents', 0)}条向量")
        else:
            st.metric("向量库", "[--] 未构建")
        
        # 存储模式
        mode = "Neo4j" if kg_stats.get('is_connected') else "内存"
        st.metric("存储模式", mode)
        
    else:
        # 如果没有Neo4j图谱，显示基础图谱
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("文档数", len(st.session_state.documents))
        
        with col2:
            st.metric("三元组", len(st.session_state.kg.triples))
        
        with col3:
            st.metric("节点数", st.session_state.kg.graph.number_of_nodes())
        
        with col4:
            st.metric("边数", st.session_state.kg.graph.number_of_edges())
        
        with col5:
            if st.session_state.llm.is_initialized:
                st.metric("LLM", "[OK] 已连接")
            else:
                st.metric("LLM", "[--] 未配置")
        
        if st.session_state.vector_store and st.session_state.vector_store.is_initialized:
            vs_stats = st.session_state.vector_store.get_statistics()
            st.metric("向量库", f"[OK] {vs_stats.get('total_documents', 0)}条向量")
        else:
            st.metric("向量库", "[--] 未构建")
    
    st.success("[OK] **GraphRAG v2.2 (Neo4j增强版)** - LLM提取 + 社团划分 + 综合查询")


if __name__ == "__main__":
    main()
