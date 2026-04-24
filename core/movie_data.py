"""
电影数据处理模块
支持：加载 movie_data.json (500条)、文本向量化、知识图谱构建
"""
import json
import os
from typing import List, Dict, Any, Optional
from core.document import process_document
from core.vector_store import VectorStore
from core.llm_service import LLMService


# ============================================================
# 电影领域专用 NER/RE Prompt
# ============================================================
MOVIE_NER_RE_PROMPT = """## 任务

从下面的电影简介文本中提取实体和关系，**只输出JSON**，不要输出任何解释。

## 文本结构

每条文本是一部电影的简介，包含：电影名称、导演、演员、上映年份、类型、剧情简介。

示例：
"《大话西游之大圣娶亲》是周星驰主演的古装爱情电影，1995年上映。刘镇伟执导，周星驰、吴孟达共同出演。影片讲述了至尊宝为救白晶晶而穿越回五百年前，遇到紫霞仙子发生的一系列故事。"

## 实体类型（共6种）

| 类型 | 说明 | 示例 |
|------|------|------|
| MOVIE | 电影名称，用《》包裹 | 《大话西游之大圣娶亲》 |
| PERSON | 导演、演员、编剧等人物 | 周星驰、刘镇伟、吴孟达 |
| ORGANIZATION | 制片公司、出品方 | 华纳兄弟、中国电影集团 |
| LOCATION | 故事发生地、拍摄地 | 香港、北京、美国 |
| GENRE | 电影类型 | 爱情、动作、科幻、喜剧 |
| AWARD | 获奖信息 | 金像奖、金马奖、奥斯卡 |

## 关系类型（共8种）

| 关系 | 格式 | 说明 | 必/选 |
|------|------|------|------|
| 导演 | (MOVIE) --[导演]--> (PERSON) | 电影的导演 | 必 |
| 演员 | (MOVIE) --[演员]--> (PERSON) | 电影的演员（主要演员必选，配角可选） | 必 |
| 上映 | (MOVIE) --[上映]--> (LOCATION) | 首映/上映地点 | 选 |
| 类型 | (MOVIE) --[类型]--> (GENRE) | 电影所属类型 | 必 |
| 出品 | (MOVIE) --[出品]--> (ORGANIZATION) | 出品/制片公司 | 选 |
| 获奖 | (MOVIE) --[获奖]--> (AWARD) | 电影获奖情况 | 选 |
| 同演员 | (PERSON) --[同演员]--> (PERSON) | 两位演员共同出演过同一部电影 | 选 |
| 合作 | (PERSON) --[合作]--> (PERSON) | 导演和演员的合作关系 | 选 |

## 核心规则

1. **每部电影必须提取为 MOVIE 实体**，片名作为 name（如"《大话西游之大圣娶亲》"）
2. **每部 MOVIE 至少建立导演、演员、类型3条关系**
3. 人物实体提取"姓名"，去掉头衔（去掉"先生/女士/导演"等后缀）
4. 同一演员出演多部电影：LLM 只建立 MOVIE--演员-->PERSON 关系，跨电影连接由程序自动完成
5. 如果电影介绍中提到演员之间的关系（如"周星驰和吴孟达是黄金搭档"），建立 PERSON--同演员-->PERSON 关系
6. 禁止创建类型错配关系（如 MOVIE--导演-->MOVIE）
7. 实体名规范：2-30字、中文为主、不是纯数字

## 示例

输入：
"《大话西游之大圣娶亲》是周星驰主演的古装爱情电影，1995年上映。刘镇伟执导，周星驰、吴孟达共同出演。影片讲述了至尊宝为救白晶晶而穿越回五百年前，遇到紫霞仙子发生的一系列故事。该片由彩星电影公司出品，曾获得香港电影金像奖最佳编剧提名。"

正确输出：
```json
{
  "entities": [
    {"name": "《大话西游之大圣娶亲》", "type": "MOVIE"},
    {"name": "周星驰", "type": "PERSON"},
    {"name": "刘镇伟", "type": "PERSON"},
    {"name": "吴孟达", "type": "PERSON"},
    {"name": "古装爱情", "type": "GENRE"},
    {"name": "彩星电影公司", "type": "ORGANIZATION"},
    {"name": "香港电影金像奖", "type": "AWARD"},
    {"name": "香港", "type": "LOCATION"}
  ],
  "relations": [
    {"head": "《大话西游之大圣娶亲》", "relation": "导演", "tail": "刘镇伟"},
    {"head": "《大话西游之大圣娶亲》", "relation": "演员", "tail": "周星驰"},
    {"head": "《大话西游之大圣娶亲》", "relation": "演员", "tail": "吴孟达"},
    {"head": "《大话西游之大圣娶亲》", "relation": "类型", "tail": "古装爱情"},
    {"head": "《大话西游之大圣娶亲》", "relation": "出品", "tail": "彩星电影公司"},
    {"head": "《大话西游之大圣娶亲》", "relation": "获奖", "tail": "香港电影金像奖"},
    {"head": "《大话西游之大圣娶亲》", "relation": "上映", "tail": "香港"},
    {"head": "周星驰", "relation": "同演员", "tail": "吴孟达"},
    {"head": "周星驰", "relation": "合作", "tail": "刘镇伟"}
  ]
}
```

## 待处理文本
"""


# ============================================================
# 数据加载器
# ============================================================

def load_movie_data(json_path: str) -> List[Dict]:
    """
    加载电影JSON数据文件

    Args:
        json_path: movie_data.json 的路径

    Returns:
        电影列表，每条包含 id, title, content, source, category
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        movies = json.load(f)
    return movies


def movie_to_chunks(movies: List[Dict]) -> List[Dict]:
    """
    将电影数据转换为文本块（供向量化和图谱构建使用）

    每部电影构建一条描述文本，格式为：
    "[电影名称]：[简介内容]"

    Returns:
        chunks: [{'id', 'text', 'source', 'chunk_index'}]
    """
    chunks = []
    for m in movies:
        movie_id = m.get('id', 0)
        title = m.get('title', '未知电影')
        content = m.get('content', '')
        source = m.get('source', 'movie_data')
        category = m.get('category', '电影')

        text = f"《{title}》：{content}"
        chunks.append({
            'id': f"movie_{movie_id}",
            'text': text,
            'source': source,
            'chunk_index': movie_id,
            'title': title,
            'category': category,
        })
    return chunks


def process_movie_data(
    json_path: str,
    llm_service: Optional[LLMService] = None,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    batch_size: int = 10,
    progress_callback=None
) -> Dict[str, Any]:
    """
    一站式处理电影数据：加载 → 向量化 → 图谱构建

    Args:
        json_path: movie_data.json 路径
        llm_service: LLM服务（用于图谱构建，可选）
        embedding_model: Embedding 模型名称
        batch_size: 批处理大小
        progress_callback: 进度回调函数 fn(current, total, message)

    Returns:
        {
            'chunks': 文本块列表,
            'vector_store': 向量库,
            'kg_entities': 实体列表,
            'kg_relations': 关系列表,
            'stats': 统计信息
        }
    """
    # 1. 加载数据
    movies = load_movie_data(json_path)
    total = len(movies)

    # 2. 转换为文本块
    chunks = movie_to_chunks(movies)

    # 3. 向量化
    vector_store = VectorStore(embedding_model=embedding_model)
    vs_ok = vector_store.initialize()
    if not vs_ok:
        raise Exception("向量数据库初始化失败，请检查 ChromaDB 和 embedding 模型")
    for i, chunk in enumerate(chunks):
        vector_store.add_documents([chunk])
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, f"向量化: {i}/{total}")

    # 4. 构建知识图谱（如果提供了 LLM）
    entities = []
    relations = []
    if llm_service is not None:
        from core.neo4j_kg import Neo4jKnowledgeGraph, Entity, Relation

        # 创建独立的电影领域 KG
        kg = Neo4jKnowledgeGraph(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            llm_service=llm_service
        )
        kg.connect()
        kg.set_vector_store(vector_store)

        # 临时注入电影 prompt（通过私有属性）
        kg._ner_re_prompt = MOVIE_NER_RE_PROMPT

        # 添加关系类型映射
        if not hasattr(kg, '_extra_relation_types'):
            kg._extra_relation_types = {}
        kg._extra_relation_types.update({
            '导演': '导演', 'DAXTOR': '导演',
            '演员': '演员', 'ACTOR': '演员',
            '上映': '上映', 'RELEASE': '上映',
            '类型': '类型', 'GENRE': '类型',
            '出品': '出品', 'PRODUCED_BY': '出品',
            '获奖': '获奖', 'AWARD': '获奖',
            '同演员': '同演员', 'CO_ACTOR': '同演员',
            '合作': '合作', 'COLLABORATED': '合作',
        })

        def pg_callback(current, total, message):
            if progress_callback:
                progress_callback(current, total, message)

        kg_stats = kg.build_knowledge_graph_from_documents(
            chunks=chunks,
            progress_callback=pg_callback
        )

        entities = kg.local_entities
        relations = kg.local_relations

        return {
            'chunks': chunks,
            'vector_store': vector_store,
            'kg_entities': entities,
            'kg_relations': relations,
            'stats': {
                'total_movies': total,
                'total_chunks': len(chunks),
                'total_entities': len(entities),
                'total_relations': len(relations),
                'build_stats': kg_stats,
            }
        }

    return {
        'chunks': chunks,
        'vector_store': vector_store,
        'kg_entities': [],
        'kg_relations': [],
        'stats': {
            'total_movies': total,
            'total_chunks': len(chunks),
            'total_entities': 0,
            'total_relations': 0,
        }
    }


def get_available_datasets() -> Dict[str, Dict]:
    """
    返回所有可用数据集的信息

    Returns:
        {
            'complaint': {'name': ..., 'path': ..., 'domain': ..., 'count': ...},
            'movie': {'name': ..., 'path': ..., 'domain': ..., 'count': ...},
            ...
        }
    """
    base = os.path.dirname(os.path.dirname(__file__))
    datasets = {}

    # 电影数据集
    movie_path = os.path.join(base, 'data', 'raw_data', 'movie_data.json')
    if os.path.exists(movie_path):
        movies = load_movie_data(movie_path)
        datasets['movie'] = {
            'name': '电影数据集（500部）',
            'path': movie_path,
            'domain': 'movie',
            'description': f'包含 {len(movies)} 部电影简介，涵盖港片、华语片等多种类型',
            'count': len(movies),
            'chunk_count': len(movies),  # 每部电影一个chunk
        }

    # 投诉数据集（如果 Excel 文件存在）
    import glob as _glob
    excel_files = _glob.glob(os.path.join(base, '**', '*.xlsx'), recursive=True)
    for fpath in excel_files:
        fname = os.path.basename(fpath)
        if '去隐私' in fname or 'jingqutousu' in fname.lower():
            datasets['complaint'] = {
                'name': '崂山景区投诉数据集',
                'path': fpath,
                'domain': 'complaint',
                'description': '崂山景区投诉工单，包含投诉人、问题、涉及部门等信息',
                'count': None,  # 动态读取
                'chunk_count': None,
            }
            break

    return datasets
