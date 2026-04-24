"""
检索引擎模块
负责：向量检索、图谱检索、混合检索、混合推理（含多跳）

增强功能：
- 支持Neo4j知识图谱（多层级图谱+社团划分）
- 综合查询方法（向量库+Neo4j融合）
- 节点-向量关联查询
- 多跳推理引擎（支持两步及以上推理问题）
"""
import re
from typing import List, Dict, Optional
from core.knowledge_graph import KnowledgeGraph
from core.vector_store import VectorStore
from core.neo4j_kg import Neo4jKnowledgeGraph


# ============================================================
# 辅助函数：投诉领域格式化
# ============================================================

def _is_complaint_id(text: str) -> bool:
    """判断字符串是否为工单ID（纯数字，15位以上）。"""
    cleaned = re.sub(r'^工单', '', text).strip()
    return bool(cleaned) and cleaned.isdigit() and len(cleaned) >= 12


def _format_complaint_entity(entity: str) -> str:
    """
    将图谱实体还原为可读格式。

    - 工单ID "工单160724181838968317" → "工单160724181838968317"
    - 带时间的投诉人 "刘先生@2016年7月14日" → "刘先生（2016年7月14日）"
    - 还原事件摘要中的多余符号
    """
    if not entity:
        return entity

    # 工单ID：加上"工单"前缀（如果没有）
    if _is_complaint_id(entity):
        if not entity.startswith("工单"):
            return f"工单{entity}"
        return entity

    # 带时间的投诉人 "姓名@日期" → "姓名（日期）"
    if '@' in entity:
        name, date = entity.split('@', 1)
        return f"{name}（{date}）"

    # 还原过于简短的EVENT实体（补全句号、标点）
    # 例如 "大巴票收130元" → 保持原样（已经是可读句子）

    return entity


def _normalize_relation(rel: str) -> str:
    """
    关系词中文友好化（兼容 Neo4jKG 输出的英文别名和中文原名）。
    """
    mapping = {
        'REPORTER': '反映人', '反映': '反映人',
        'PROCESSES': '处理', '处理': '处理',
        'LOCATION': '发生地', '发生地': '发生地',
        'ISSUES': '涉及问题', '涉及问题': '涉及问题',
        'EVENTS': '具体事件', '具体事件': '具体事件',
        'RIDE': '乘坐', '乘坐': '乘坐',
        'CATEGORY': '属于', '属于': '属于',
        'DEPT': '涉及部门', '涉及部门': '涉及部门',
        'INVOLVES': '涉及人', '涉及人': '涉及人',
        'AREA': '区域', '区域': '区域',
        'ACTOR': '演员', '导演': '导演',
        'RELATED': '关联', '关联': '关联',
    }
    return mapping.get(rel, rel)


# ============================================================
# 多跳推理引擎
# ============================================================

# 问题模式 → 推理路径配置
MULTI_HOP_PATTERNS = [
    {
        "keywords": ["还", "其他", "还有", "另外"],
        "pattern": "HOP_THEN_FIND",
        "description": "A还做了X？ → 先找A，再找A相关的其他事物",
        "steps": [
            {"action": "find_entity", "desc": "找到问题中的核心实体"},
            {"action": "hop_1", "desc": "通过关系找到关联实体"},
            {"action": "hop_2", "desc": "再找该实体关联的其他实体"},
        ]
    },
    {
        "keywords": ["谁", "哪个", "哪些"],
        "pattern": "FILTER_THEN_MATCH",
        "description": "谁在电影X中出现？ → 先找电影，再找演员",
        "steps": [
            {"action": "find_entity", "desc": "找到目标类型"},
            {"action": "hop_1", "desc": "通过关系过滤"},
        ]
    },
    {
        "keywords": ["导演", "演员", "主演", "出演"],
        "pattern": "RELATION_NAVIGATE",
        "description": "X导演的/主演的 → 关系导航",
        "steps": [
            {"action": "find_entity", "desc": "找到人物或电影"},
            {"action": "hop_1", "desc": "沿关系找目标"},
        ]
    },
]


class MultiHopReasoner:
    """
    多跳推理器
    根据用户问题自动识别推理模式，然后执行多跳查询
    """

    def __init__(self, neo4j_kg: Neo4jKnowledgeGraph = None,
                 knowledge_graph: KnowledgeGraph = None,
                 vector_store: VectorStore = None):
        self.neo4j_kg = neo4j_kg
        self.kg = knowledge_graph
        self.vector_store = vector_store

    def detect_hop_type(self, query: str) -> Dict:
        """识别问题是否需要多跳推理，返回推理配置"""
        query_lower = query.lower()

        for p in MULTI_HOP_PATTERNS:
            for kw in p["keywords"]:
                if kw in query_lower or kw in query:
                    return {"needs_hop": True, "pattern": p["pattern"], "config": p}

        # 额外检测：问"xxx的其他xxx"结构
        if re.search(r'的\s*其他', query) or re.search(r'还\s*[\u4e00-\u9fa5]{1,5}', query):
            return {"needs_hop": True, "pattern": "HOP_THEN_FIND", "config": MULTI_HOP_PATTERNS[0]}

        return {"needs_hop": False, "pattern": None, "config": None}

    def extract_core_entities(self, query: str) -> List[str]:
        """从问题中提取关键实体名（用于图谱查询）"""
        # 去掉常见疑问词
        clean = re.sub(r'[吗呢吧呀啊？?。,\s]', '', query)
        # 提取人名（先生/女士/导演/演员）
        persons = re.findall(r'[\u4e00-\u9fa5]{2,4}(?:先生|女士|导演|演员)', clean)
        # 提取电影名（用《》）
        movies = re.findall(r'《([^》]+)》', query)
        # 提取裸人名（2-4字，开头不是"其他"/"还有"）
        naked = re.findall(r'^([\u4e00-\u9fa5]{2,4})(?:的|导演|演员|出演|主演|还|其他)', query)
        # 提取关键词（长词优先）
        candidates = persons + movies + naked
        return list(dict.fromkeys(candidates))

    # ----- 电影领域多跳查询 -----

    def movie_multi_hop(self, query: str, entity: str) -> Dict:
        """
        电影领域多跳推理
        例如："周星驰还主演了哪些电影？" → PERSON → MOVIE → (找其他MOVIE)
        """
        results = {"reasoning_chain": [], "entities": [], "relations": [], "answer_context": ""}

        if not self.neo4j_kg:
            return results

        # 模式A：问某人的其他电影（"xxx还/其他/导演/主演了什么"）
        m1 = re.search(r'([\u4e00-\u9fa5]{2,4})(?:还|其他)?(?:导演|主演|出演|拍了|拍过)?了?哪些?电影', query)
        if m1:
            person_name = m1.group(1)
            # 1跳：找该人导演的电影
            directed = self.neo4j_kg.search_relations(
                head=person_name, relation="导演"
            )
            # 2跳：找该人演员的电影
            acted = self.neo4j_kg.search_relations(
                head=person_name, relation="演员"
            )

            for r in directed:
                results["reasoning_chain"].append({
                    "step": 1,
                    "from": person_name,
                    "relation": "导演",
                    "to": r["tail"]
                })
            for r in acted:
                results["reasoning_chain"].append({
                    "step": 2,
                    "from": person_name,
                    "relation": "演员",
                    "to": r["tail"]
                })

            results["entities"] = list({r["tail"] for r in results["reasoning_chain"]})

        # 模式B：问某电影的其他演员
        m2 = re.search(r'《([^》]+)》的?其他?演员', query)
        if m2:
            movie_name = f"《{m2.group(1)}》"
            actors = self.neo4j_kg.search_relations(head=movie_name, relation="演员")
            results["reasoning_chain"] = [
                {"step": 1, "from": movie_name, "relation": "演员", "to": a["tail"]}
                for a in actors
            ]
            results["entities"] = [a["tail"] for a in actors]

        # 模式C：问某演员还和谁合作过
        m3 = re.search(r'([\u4e00-\u9fa5]{2,4})(?:还|也)?和谁合作', query)
        if m3:
            person_name = m3.group(1)
            co_actors = self.neo4j_kg.search_relations(head=person_name, relation="同演员")
            results["reasoning_chain"] = [
                {"step": 1, "from": person_name, "relation": "同演员", "to": r["tail"]}
                for r in co_actors
            ]
            results["entities"] = [r["tail"] for r in co_actors]

        # 模式D：问某类型的所有电影
        m4 = re.search(r'([\u4e00-\u9fa5]{2,4})(?:类型|类型|风格)的电影', query)
        if m4:
            genre_name = m4.group(1)
            movies = self.neo4j_kg.search_relations(head=genre_name, relation="类型")
            if not movies:
                movies = self.neo4j_kg.search_relations(head=genre_name, relation=None)
            results["reasoning_chain"] = [
                {"step": 1, "from": movie, "relation": "类型", "to": genre_name}
                for movie in [r["head"] for r in movies]
            ]
            results["entities"] = list({r["head"] for r in movies})

        # 构建答案上下文
        if results["reasoning_chain"]:
            ctx_parts = ["【多跳推理结果】"]
            ctx_parts.append(f"从问题「{query}」推理得到：")
            for step in results["reasoning_chain"]:
                ctx_parts.append(f"  第{step['step']}跳: {step['from']} --[{step['relation']}]--> {step['to']}")
            results["answer_context"] = "\n".join(ctx_parts)

        return results

    # ----- 投诉领域多跳查询 -----

    def complaint_multi_hop(self, query: str, entity: str) -> Dict:
        """
        投诉领域多跳推理
        例如："大巴乱收费的投诉人还投诉了什么？" → EVENT → COMPLAINT → PERSON → 其他COMPLAINT
        """
        results = {"reasoning_chain": [], "entities": [], "relations": [], "answer_context": ""}

        if not self.neo4j_kg:
            return results

        # 模式A：问某投诉人的其他投诉
        m1 = re.search(r'([\u4e00-\u9fa5]{2,4}@[\d\-: ]+)(?:还|其他)?投诉了?什么', query)
        if not m1:
            # 也支持不带时间的格式
            m1 = re.search(r'([\u4e00-\u9fa5]{2,4})(?:还|其他)?投诉了?什么', query)
        if m1:
            person_name = m1.group(1)
            # 1跳：找该人所有投诉
            complaints = self.neo4j_kg.search_relations(
                head=person_name, relation="反映人"
            )
            # 收集所有投诉
            all_complaints = [c["tail"] for c in complaints]
            for r in complaints:
                results["reasoning_chain"].append({
                    "step": 1,
                    "from": person_name,
                    "relation": "反映人",
                    "to": r["tail"]
                })
            # 2跳：找这些投诉涉及的问题
            for comp in all_complaints[:5]:
                issues = self.neo4j_kg.search_relations(head=comp, relation="涉及问题")
                for issue in issues:
                    results["reasoning_chain"].append({
                        "step": 2,
                        "from": comp,
                        "relation": "涉及问题",
                        "to": issue["tail"]
                    })
            results["entities"] = all_complaints

        # 模式B：问某问题的所有投诉
        m2 = re.search(r'([\u4e00-\u9fa5]{2,8})(?:投诉|乱收费|问题)', query)
        if m2:
            issue_name = m2.group(1)
            complaints = self.neo4j_kg.search_relations(head=issue_name, relation="涉及问题")
            if not complaints:
                complaints = self.neo4j_kg.search_relations(head=issue_name, relation=None)
            results["reasoning_chain"] = [
                {"step": 1, "from": c.get("head", ""), "relation": "涉及问题", "to": issue_name}
                for c in complaints
            ]
            results["entities"] = list({c.get("head", "") for c in complaints})

        if results["reasoning_chain"]:
            ctx_parts = ["【多跳推理结果】"]
            ctx_parts.append(f"从问题「{query}」推理得到：")
            step_results = {}
            for step in results["reasoning_chain"]:
                s = step["step"]
                if s not in step_results:
                    step_results[s] = []
                step_results[s].append(f"{step['from']} --[{step['relation']}]--> {step['to']}")
            for s, lines in step_results.items():
                ctx_parts.append(f"  第{s}跳:")
                for line in lines[:3]:
                    ctx_parts.append(f"    {line}")
            results["answer_context"] = "\n".join(ctx_parts)

        return results

    def query(self, query: str, domain: str = "movie") -> Dict:
        """
        主入口：判断是否多跳 → 执行对应推理

        Args:
            query: 用户问题
            domain: 领域类型 ("movie" 或 "complaint")

        Returns:
            {
                "needs_hop": bool,
                "hop_type": str,
                "reasoning_chain": [...],
                "entities": [...],
                "answer_context": str,
                "subgraph_nodes": [...],
                "subgraph_edges": [...]
            }
        """
        hop_info = self.detect_hop_type(query)
        needs_hop = hop_info["needs_hop"]

        if not needs_hop:
            return {
                "needs_hop": False,
                "hop_type": None,
                "reasoning_chain": [],
                "entities": [],
                "answer_context": "",
                "subgraph_nodes": [],
                "subgraph_edges": []
            }

        # 根据领域选择推理器
        if domain == "movie":
            raw_results = self.movie_multi_hop(query, "")
        else:
            raw_results = self.complaint_multi_hop(query, "")

        # 构建子图（用于可视化）
        subgraph_nodes = []
        subgraph_edges = []
        seen_nodes = set()
        seen_edges = set()

        for step in raw_results.get("reasoning_chain", []):
            for node in [step["from"], step["to"]]:
                if node and node not in seen_nodes:
                    seen_nodes.add(node)
                    subgraph_nodes.append({"id": node, "label": node})
            edge_key = f"{step['from']}|{step['relation']}|{step['to']}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                subgraph_edges.append({
                    "from": step["from"],
                    "to": step["to"],
                    "label": step["relation"],
                    "step": step["step"]
                })

        return {
            "needs_hop": needs_hop,
            "hop_type": hop_info["pattern"],
            "reasoning_chain": raw_results.get("reasoning_chain", []),
            "entities": raw_results.get("entities", []),
            "answer_context": raw_results.get("answer_context", ""),
            "subgraph_nodes": subgraph_nodes,
            "subgraph_edges": subgraph_edges,
        }


class Retriever:
    """
    检索引擎类
    
    功能：
    - 向量检索（语义匹配 - 使用真正的Embedding模型）
    - NetworkX图检索（基础版）
    - Neo4j知识图谱检索（增强版 - 支持多跳、社团）
    - 混合检索（结果融合）
    - 综合查询（向量库 + Neo4j深度融合）
    - 上下文格式化
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph = None, 
                 vector_store: VectorStore = None,
                 neo4j_kg: Neo4jKnowledgeGraph = None):
        self.kg = knowledge_graph  # NetworkX基础图谱
        self.vector_store = vector_store  # ChromaDB向量库
        self.neo4j_kg = neo4j_kg  # Neo4j增强知识图谱
        self.vector_weight = 0.5
        self.graph_weight = 0.5
    
    def set_vector_store(self, vector_store: VectorStore):
        """设置向量数据库"""
        self.vector_store = vector_store
        
        # 如果有Neo4j KG，同步设置vector_store引用
        if self.neo4j_kg:
            self.neo4j_kg.set_vector_store(vector_store)
    
    def set_neo4j_kg(self, neo4j_kg: Neo4jKnowledgeGraph):
        """设置Neo4j知识图谱"""
        self.neo4j_kg = neo4j_kg

        # 建立与向量库的关联
        if self.vector_store:
            neo4j_kg.set_vector_store(self.vector_store)

    def multi_hop_query(self, query: str, domain: str = "movie") -> Dict:
        """
        多跳推理查询

        Args:
            query: 用户问题
            domain: 领域类型 ("movie" 或 "complaint")

        Returns:
            {
                "needs_hop": bool,
                "hop_type": str,
                "reasoning_chain": [...],
                "entities": [...],
                "answer_context": str,
                "subgraph_nodes": [...],
                "subgraph_edges": [...]
            }
        """
        reasoner = MultiHopReasoner(
            neo4j_kg=self.neo4j_kg,
            knowledge_graph=self.kg,
            vector_store=self.vector_store
        )
        return reasoner.query(query, domain=domain)

    def _detect_domain(self, query: str, documents: List[Dict] = None) -> str:
        """根据查询或文档内容推断领域"""
        movie_keywords = ["电影", "导演", "演员", "主演", "出演", "票房", "获奖", "科幻", "爱情", "喜剧", "港片"]
        complaint_keywords = ["投诉", "举报", "乱收费", "景区", "大巴", "投诉人", "反映", "工单", "处理"]

        q_lower = query
        for kw in movie_keywords:
            if kw in q_lower:
                return "movie"
        for kw in complaint_keywords:
            if kw in q_lower:
                return "complaint"

        if documents:
            doc_text = " ".join([d.get('text', '')[:100] for d in documents[:3]])
            for kw in movie_keywords:
                if kw in doc_text:
                    return "movie"
            for kw in complaint_keywords:
                if kw in doc_text:
                    return "complaint"

        return "movie"  # 默认电影领域

    def vector_search(self, query: str, documents: List[Dict] = None, top_k: int = None) -> List[Dict]:
        """
        真正的向量相似度检索

        使用预训练Embedding模型 + ChromaDB进行向量搜索

        Args:
            query: 查询文本
            documents: 文档列表（兼容旧接口，优先使用vector_store）
            top_k: 返回数量，None 表示返回全部结果

        Returns:
            results: 检索结果列表
        """
        # 优先使用VectorStore进行真正的向量检索
        if self.vector_store and self.vector_store.is_initialized:
            try:
                results = self.vector_store.similarity_search(query, top_k or 1000)
                if results:
                    print(f"[OK] 使用ChromaDB向量检索成功: {len(results)}条结果")
                    return results
            except Exception as e:
                print(f"[WARN] ChromaDB检索失败，回退到基础方法: {e}")

        # 回退方案：如果没有VectorStore或检索失败，使用文档列表做简单匹配
        if documents:
            keywords = re.findall(r'[\u4e00-\u9fa5]{2,10}|《[^》]+》', query)

            results = []
            for doc in documents:
                match_count = sum(1 for kw in keywords if kw in doc.get('text', ''))

                if match_count > 0:
                    score = match_count / len(keywords) if keywords else 0
                    results.append({
                        'type': 'vector_fallback',
                        'content': doc['text'][:200],
                        'source': doc.get('source', ''),
                        'score': score,
                        'match_keywords': match_count,
                        'note': '使用关键词匹配（回退模式）'
                    })

            results.sort(key=lambda x: x['score'], reverse=True)
            return results if top_k is None else results[:top_k]

        return []

    def graph_search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        图谱检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            results: 检索结果列表
        """
        if not self.kg:
            return []
        
        return self.kg.search(query, top_k)
    
    def hybrid_retrieve(self, query: str, documents: List[Dict],
                       use_vector: bool = True, use_graph: bool = True,
                       top_k: int = None) -> Dict:
        """
        混合检索

        Args:
            query: 查询文本
            documents: 文档列表
            use_vector: 是否使用向量检索
            use_graph: 是否使用图谱检索
            top_k: 返回数量，None 表示返回全部结果

        Returns:
            retrieval_result: 包含所有检索结果的字典
        """
        vector_results = []
        graph_results = []

        # 向量检索
        if use_vector and documents:
            vector_results = self.vector_search(query, documents, top_k)

        # 图谱检索
        if use_graph and self.kg:
            graph_results = self.graph_search(query, top_k)

        # 合并结果
        merged_results = []

        for r in vector_results:
            merged_results.append({**r, 'search_type': 'vector'})

        for r in graph_results:
            merged_results.append({**r, 'search_type': 'graph'})

        # 按分数排序
        merged_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        # top_k=None 时返回全部
        if top_k is None:
            merged_slice = merged_results
        else:
            merged_slice = merged_results[:top_k * 2]

        return {
            'query': query,
            'vector_results': vector_results,
            'graph_results': graph_results,
            'merged_results': merged_slice,
            'vector_count': len(vector_results),
            'graph_count': len(graph_results),
            'total_count': len(merged_slice)
        }
    
    def format_context(self, retrieval_result: Dict) -> str:
        """
        格式化上下文（供LLM使用）
        
        Args:
            retrieval_result: 检索结果
        
        Returns:
            context: 格式化的上下文字符串
        """
        parts = ["=== 知识图谱检索结果 ==="]
        
        for i, result in enumerate(retrieval_result.get('graph_results', []), 1):
            parts.append(f"{i}. {result['head']} --[{result['relation']}]--> {result['tail']}")
        
        parts.append("\n=== 向量检索结果 ===")
        
        for i, result in enumerate(retrieval_result.get('vector_results', []), 1):
            parts.append(f"{i}. {result.get('content', '')[:150]}...")
        
        return "\n".join(parts) if parts else "未找到相关信息"
    
    def comprehensive_query(self, query: str, documents: List[Dict] = None,
                           top_k: int = None) -> Dict:
        """
        综合查询方法：深度融合向量库和Neo4j知识图谱 + 多跳推理

        查询策略：
        1. 自动检测是否需要多跳推理
        2. 如果有Neo4j KG → 使用hybrid_query（多跳+向量+社团）
        3. 多跳推理结果注入到结果中
        4. 否则 → 使用传统hybrid_retrieve（NetworkX+向量）

        Args:
            query: 用户问题
            documents: 文档列表（兼容旧接口）
            top_k: 返回数量，None 表示返回全部结果

        Returns:
            result: 综合查询结果（含 reasoning_chain, subgraph_nodes/edges）
        """
        domain = self._detect_domain(query, documents)

        # 先尝试多跳推理
        hop_result = self.multi_hop_query(query, domain=domain)

        # 再执行常规检索
        if self.neo4j_kg and (self.neo4j_kg.is_connected or self.neo4j_kg.local_entities):
            print("[QUERY] 使用Neo4j增强型综合查询...")
            result = self.neo4j_kg.hybrid_query(
                query=query,
                top_k_graph=top_k if top_k is not None else 1000,
                top_k_vector=top_k if top_k is not None else 1000
            )
        else:
            print("[QUERY] 使用传统混合检索（NetworkX + 向量库）...")
            result = self.hybrid_retrieve(
                query=query,
                documents=documents or [],
                use_vector=True,
                use_graph=True,
                top_k=top_k
            )

        # 注入多跳推理结果
        if hop_result["needs_hop"]:
            result["needs_hop"] = True
            result["hop_type"] = hop_result["hop_type"]
            result["reasoning_chain"] = hop_result["reasoning_chain"]
            result["subgraph_nodes"] = hop_result["subgraph_nodes"]
            result["subgraph_edges"] = hop_result["subgraph_edges"]
            if hop_result["answer_context"]:
                result["answer_context"] = hop_result["answer_context"]
        else:
            result["needs_hop"] = False

        return result
    
    def format_comprehensive_context(self, result: Dict) -> str:
        """
        格式化综合查询结果（包含社团信息、推理过程）
        
        Args:
            result: comprehensive_query或hybrid_query的返回结果
        
        Returns:
            context: 增强的上下文字符串
        """
        parts = []
        
        # 基本信息
        parts.append(f"=== 综合检索结果 ===")
        parts.append(f"查询: {result.get('query', 'N/A')}")
        
        # 图谱路径（如果存在）
        graph_results = result.get('graph_results', [])
        if graph_results:
            parts.append(f"\n【知识图谱路径】(共{len(graph_results)}条)")
            for i, path in enumerate(graph_results[:3], 1):
                path_str = path.get('path', '')
                length = path.get('length', 0)
                source = path.get('source_entity', '')
                parts.append(f"{i}. [长度:{length}跳] {path_str}")
                if source:
                    parts.append(f"   起始实体: {source}")
        
        # 向量检索结果
        vector_results = result.get('vector_results', [])
        if vector_results:
            parts.append(f"\n【语义相关片段】(共{len(vector_results)}条)")
            for i, v_res in enumerate(vector_results[:3], 1):
                score = v_res.get('score', 0)
                content = v_res.get('content', '')[:120]
                source = v_res.get('source', '未知来源')
                parts.append(f"{i}. [相似度:{score:.3f}] ({source})")
                parts.append(f"   内容: {content}...")
        
        # 社团上下文（Neo4j增强版特有）
        community_ctx = result.get('community_context', {})
        if community_ctx:
            parts.append(f"\n【社群分析】(共{len(community_ctx)}个相关社群)")
            for comm_id, ctx in list(community_ctx.items())[:2]:
                members = ctx.get('members', [])[:5]
                size = ctx.get('size', 0)
                parts.append(f"- {comm_id}: {size}个成员")
                parts.append(f"  核心成员: {', '.join(members)}")
        
        # 多跳推理链
        reasoning_chain = result.get('reasoning_chain', [])
        if reasoning_chain:
            parts.append(f"\n【多跳推理链】(共{len(reasoning_chain)}条推理路径)")
            step_groups = {}
            for step in reasoning_chain:
                s = step.get('step', 1)
                if s not in step_groups:
                    step_groups[s] = []
                step_groups[s].append(f"{step.get('from', '')} --[{step.get('relation', '')}]--> {step.get('to', '')}")
            for s, lines in sorted(step_groups.items()):
                parts.append(f"  第{s}跳:")
                for line in lines[:5]:
                    parts.append(f"    {line}")

        # 推理说明
        explanation = result.get('explanation', '')
        if explanation:
            parts.append(f"\n【推理过程】\n{explanation}")
        
        return "\n".join(parts)
