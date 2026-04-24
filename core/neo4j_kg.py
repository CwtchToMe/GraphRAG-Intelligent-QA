"""
Neo4j知识图谱模块
负责：
1. 基于LLM的实体关系提取
2. Neo4j图数据库操作
3. 多层级图谱构建（社团划分+抽象关键词）
4. 图谱节点与向量库向量的关联
5. 综合查询方法（图谱+向量库融合）

技术栈：
- Neo4j: 图数据库
- LangChain LLM: 实体关系提取
- NetworkX: 社团划分算法
- ChromaDB: 向量库关联
"""
import json
import re
import sys
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime


def safe_print(msg: str = ""):
    """跨平台安全输出，兼容 Windows 控制台 Unicode 字符"""
    try:
        print(msg)
    except OSError:
        try:
            sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
        except Exception:
            pass


from collections import defaultdict


@dataclass
class Entity:
    """实体数据结构"""
    name: str
    type: str
    properties: Dict = field(default_factory=dict)
    vector_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.type,
            'properties': self.properties,
            'vector_id': self.vector_id
        }


@dataclass
class Relation:
    """关系数据结构"""
    head: str
    relation: str
    tail: str
    properties: Dict = field(default_factory=dict)
    confidence: float = 0.8
    source_chunk_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'head': self.head,
            'relation': self.relation,
            'tail': self.tail,
            'properties': self.properties,
            'confidence': self.confidence,
            'source_chunk_id': self.source_chunk_id
        }


class Neo4jKnowledgeGraph:
    """
    Neo4j知识图谱类
    
    功能：
    - 连接Neo4j数据库
    - 基于LLM的实体关系提取
    - 图谱构建与管理
    - 多层级图谱（社团划分）
    - 节点与向量的关联
    - 综合查询
    """
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 llm_service=None):
        """
        初始化Neo4j知识图谱
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.llm_service = llm_service
        
        self.driver = None
        self.is_connected = False
        
        self.local_entities: List[Entity] = []
        self.local_relations: List[Relation] = []
        
        self.vector_store = None
        
        # === 投诉数据专用的实体关系抽取 Prompt ===
        # 每行Excel = 一条工单 = 一个投诉簇，簇间通过共同实体连接
        self._complaint_ner_re_prompt = """## 任务

从下面的投诉工单文本中提取实体和关系，**只输出JSON**，不要输出任何解释。

## 文本结构

每条文本是一份完整的投诉工单，由以下字段按顺序拼接：
"工单编号 时间 来电人 [诉求人身份] 反映 [投诉内容，包括具体事件] [交通工具] [联系方式]。该投诉由[部门]处理。"

## 实体类型（共9种）

| 类型 | 说明 | 示例 |
|------|------|------|
| COMPLAINT | 每条工单 = 一个投诉簇，用工单编号唯一标识 | 工单160714142327592834 |
| PERSON | 投诉人/反映人，**必须附加时间以区分同名人物** | 刘先生@2016年7月14日 |
| LOCATION | 景区名称/景点/具体地点 | 崂山景区、流清河、仰口、大河东客服中心、停车场 |
| ORGANIZATION | 负责处理或涉及的部门/机构 | 崂山风景区管理局、崂山旅游稽查大队、派出所 |
| ISSUE | 投诉的具体问题（宽泛类型） | 乱收费、票务纠纷、司机绕路、黑车、排队过长 |
| EVENT | **投诉中具体发生了什么事**，必须从内容中提取 | 大巴票收130元、计价器乱跳多收5元、门票只给一张没给发票 |
| VEHICLE | 投诉中涉及到的交通工具 | 景区大巴、出租车、观光车、618路公交车、黑车 |
| TYPE | 三级投诉分类 | 景区秩序、景区服务、收费管理 |
| AREA | 投诉所属的区域/线路 | 流清河区域、仰口区域 |

## 关系类型（共12种）

| 关系 | 格式 | 说明 | 必/选 |
|------|------|------|------|
| 反映人 | (PERSON) --[反映人]--> (COMPLAINT) | 谁反映了这个投诉 | 必 |
| 处理 | (COMPLAINT) --[处理]--> (ORGANIZATION) | 由哪个部门处理此投诉 | 必 |
| 涉及部门 | (COMPLAINT) --[涉及部门]--> (ORGANIZATION) | 投诉还涉及哪些部门 | 选 |
| 涉及人 | (COMPLAINT) --[涉及人]--> (PERSON) | 投诉中提到的具体人员（司机、导游等） | 选 |
| 发生地 | (COMPLAINT) --[发生地]--> (LOCATION) | 投诉发生在哪里 | 必 |
| 区域 | (COMPLAINT) --[区域]--> (AREA) | 投诉属于哪个景区区域 | 选 |
| 乘坐 | (COMPLAINT) --[乘坐]--> (VEHICLE) | 投诉人乘坐了什么交通工具 | 选 |
| 具体事件 | (COMPLAINT) --[具体事件]--> (EVENT) | **投诉中具体发生了什么（从内容原句提取）** | 必 |
| 涉及问题 | (COMPLAINT) --[涉及问题]--> (ISSUE) | 投诉的问题类型 | 必 |
| 属于 | (COMPLAINT) --[属于]--> (TYPE) | 投诉属于哪个分类 | 必 |
| 金额 | (EVENT) --[涉及金额]--> (ISSUE) | 具体事件中涉及的金额 | 选 |
| 对方 | (EVENT) --[对方]--> (PERSON) | 涉及的具体人员（司机/导游姓名或身份） | 选 |

## 核心规则

1. **EVENT 是最重要的实体**：必须从投诉内容中提取**具体发生了什么事**的原句或核心描述
   - ❌ 错误："乱收费"（太抽象）
   - ✅ 正确："大巴票收130元"、"计价器乱跳多收5元"、"门票只给一张没给发票"
2. **每条COMPLAINT必须建立反映人、处理、发生地、具体事件、涉及问题5条关系**
3. PERSON实体的name格式为"姓名@时间"，**相同姓名但不同时间的人是不同实体**
4. EVENT实体从内容的核心句中提取，保留关键数字和细节
5. 禁止：自己连自己的反射关系、类型错配

## 示例

输入：
"工单160714142327592834：2016年7月14日刘先生反映在流清河停车场乘坐景区大巴时，被司机收取了130元/张的高于正常标准的车费。离开时计价器仍在跳表计价，多收5元。要求景区核实处理，多收的钱退给乘客。该投诉由崂山旅游稽查大队负责处理。"

正确输出：
```json
{
  "entities": [
    {"name": "工单160714142327592834", "type": "COMPLAINT"},
    {"name": "刘先生@2016年7月14日", "type": "PERSON"},
    {"name": "流清河停车场", "type": "LOCATION"},
    {"name": "景区大巴", "type": "VEHICLE"},
    {"name": "大巴票收130元", "type": "EVENT"},
    {"name": "计价器乱跳多收5元", "type": "EVENT"},
    {"name": "乱收费", "type": "ISSUE"},
    {"name": "崂山旅游稽查大队", "type": "ORGANIZATION"},
    {"name": "收费管理", "type": "TYPE"}
  ],
  "relations": [
    {"head": "刘先生@2016年7月14日", "relation": "反映人", "tail": "工单160714142327592834"},
    {"head": "工单160714142327592834", "relation": "发生地", "tail": "流清河停车场"},
    {"head": "工单160714142327592834", "relation": "乘坐", "tail": "景区大巴"},
    {"head": "工单160714142327592834", "relation": "处理", "tail": "崂山旅游稽查大队"},
    {"head": "工单160714142327592834", "relation": "具体事件", "tail": "大巴票收130元"},
    {"head": "工单160714142327592834", "relation": "具体事件", "tail": "计价器乱跳多收5元"},
    {"head": "工单160714142327592834", "relation": "涉及问题", "tail": "乱收费"},
    {"head": "工单160714142327592834", "relation": "属于", "tail": "收费管理"}
  ]
}
```

## 待处理文本
"""

        self.ner_prompt = self._complaint_ner_re_prompt
        self.ner_re_prompt = self._complaint_ner_re_prompt
        self.re_prompt = self._complaint_ner_re_prompt
        self._entity_aliases = {
            # 部门别名 -> 标准名
            "景区管委会": "崂山风景区管理局",
            "管委会": "崂山风景区管理局",
            "管理委员会": "崂山风景区管理局",
            "游客中心": "游客服务中心",
            "停车场收费": "停车收费",
            "景区管理局": "崂山风景区管理局",
            "执法局": "崂山旅游稽查大队",
            "执法大队": "崂山旅游稽查大队",
            # 地点别名 -> 标准名
            "崂山": "崂山景区",
            "流清": "流清河",
            "流清河景区": "流清河",
            "仰口": "仰口",
            "大河东": "大河东客服中心",
            "客服中心": "大河东客服中心",
            # 交通工具别名
            "大巴": "景区大巴",
            "观光大巴": "景区大巴",
            "大巴车": "景区大巴",
            "出租车": "出租车",
            "黑车": "黑车",
            "黑大巴": "黑车",
            "公交车": "公交车",
            "618路": "618路公交车",
            "618": "618路公交车",
        }

        # === 通用文本实体关系抽取 Prompt ===
        # 兼容任何领域的文本，自动识别实体和关系类型
        self.universal_ner_re_prompt = """## 任务

从下面的文本中提取实体和关系，**只输出JSON**，不要输出任何解释。

## 核心原则

1. **开放域识别**：文本可能来自任何领域（书籍介绍、新闻报道、技术文档、评论文章、投诉记录等），实体类型由文本内容决定。
2. **关系从文本中来**：关系类型必须从文本的语义中自然提取，不要臆造。
3. **只输出真实存在的内容**：只提取文本中明确提到或可以合理推断出的实体和关系。
4. **实体粒度适中**：人名、地名、组织名等保持完整，不要过度拆分。
5. **关系方向正确**：head → tail 方向符合自然语言描述（"A 做了 B" 则 head=A, tail=B）。
6. **能提尽提**：在保证质量的前提下，尽量多提取实体和关系，不要遗漏有价值的信息。

## 实体类型（共20种）

| 类型 | 说明 | 典型示例 |
|------|------|----------|
| PERSON | 人物、角色、作者、历史人物 | 张三、贾宝玉、曹雪芹 |
| ORGANIZATION | 组织、机构、公司、团体、政府部门 | 清华大学、谷歌、崂山风景区管理局 |
| LOCATION | 地名、地点、场所、地址、建筑 | 北京、香港、故宫、西湖 |
| EVENT | 事件、活动、会议、事故、战争 | 鸦片战争、巴黎和会、产品发布会 |
| CONCEPT | 概念、理论、观点、思想、学科 | 人工智能、量子力学、民主 |
| OBJECT | 物体、作品、产品、书籍、电影、软件 | 《红楼梦》、iPhone、Windows |
| TIME | 时间点、时间段、日期、时代、朝代 | 2020年、唐朝、三国时期 |
| ABSTRACT | 抽象概念、情感、品质、政策、制度 | 自由、诚信、法律 |
| WORK | 作品、创作、文学/影视/艺术作品 | 《红楼梦》、《活着》、《泰坦尼克号》 |
| PRODUCT | 产品、商品、软件、硬件 | iPhone、ChatGPT、Python |
| DISEASE | 疾病、症状 | 肺癌、流感、糖尿病 |
| DRUG | 药物、药品 | 阿司匹林、青霉素 |
| TECHNOLOGY | 技术、方法、工艺 | 深度学习、区块链、5G |
| AWARD | 奖项、荣誉 | 诺贝尔奖、金鸡奖、奥斯卡 |
| LAW | 法律、法规、条约 | 民法典、劳动合同法 |
| MONEY | 金额、货币、价格 | 100万元、5美元 |
| NATIONALITY | 国籍、民族、种族 | 中华民族、汉族 |
| MOVEMENT | 运动、流派、风格 | 文艺复兴、印象派 |
| OTHER | 无法归入以上类型的实体 | 特殊实体 |

## 关系类型（共40种，含方向说明）

| 关系 | 说明 | 方向 |
|------|------|------|
| 创作 | 作者/导演/艺术家创作了作品 | 创作者 → 作品 |
| 主演 | 演员参演电影/戏剧 | 演员 → 作品 |
| 导演 | 导演执导电影/戏剧 | 导演 → 作品 |
| 编剧 | 编剧编写剧本 | 编剧 → 作品 |
| 属于 | 归属、分类、成员关系 | 整体 → 部分/成员 |
| 类型 | 属于某种类型/类别 | 实体 → 类型 |
| 发生于 | 事件发生的时间或地点 | 事件 → 时间/地点 |
| 导致 | 因果关系中的原因 | 原因 → 结果 |
| 引起 | 导致某事件发生 | 原因 → 事件 |
| 参与 | 人或组织参与某事件 | 人/组织 → 事件 |
| 位于 | 实体所在的地理位置 | 实体 → 地点 |
| 任职于 | 人物在某个组织工作 | 人物 → 组织 |
| 创始人 | 创建了某组织/产品 | 人物 → 组织/产品 |
| 获奖 | 作品/人获得奖项 | 作品/人 → 奖项 |
| 使用 | 使用某种工具/技术/药物 | 使用者 → 工具/技术 |
| 治疗 | 药物治疗疾病 | 药物 → 疾病 |
| 应用于 | 技术/方法被应用于某领域 | 技术 → 领域 |
| 毕业于 | 人物毕业于某学校 | 人物 → 学校 |
| 研究 | 人物研究某领域/问题 | 人物 → 领域 |
| 推动 | 推动某项政策/改革 | 人物/组织 → 政策 |
| 反对 | 对某事物持反对态度 | 人物/组织 → 事物 |
| 促进 | 促进某事物发展 | 原因 → 结果 |
| 表达 | 表达某种情感/观点 | 人物 → 情感/观点 |
| 描写 | 作品描写某主题/人物 | 作品 → 主题/人物 |
| 讲述 | 作品讲述某故事/事件 | 作品 → 故事/事件 |
| 影响 | 对某人/事产生影响了 | 影响者 → 被影响者 |
| 对立 | 两个事物对立/冲突 | 事物A → 事物B |
| 合作 | 两人/组织合作 | 一方 → 另一方 |
| 竞争 | 两人/组织竞争 | 竞争者A → 竞争者B |
| 出生于 | 人物出生于某地/时间 | 人物 → 地点/时间 |
| 死于 | 人物死亡的时间/原因 | 人物 → 时间/原因 |
| 相似 | 两个事物相似 | 事物A → 事物B |
| 包含 | 整体包含部分 | 整体 → 部分 |
| 朝代 | 属于某朝代/时代 | 作品/人物 → 朝代 |
| 发行 | 发行/出版了某作品/产品 | 发行方 → 作品/产品 |
| 翻译 | 翻译了某作品 | 译者 → 作品 |
| 评论 | 评论某作品/事件 | 评论者 → 作品/事件 |
| 来源 | 引用/参考了某来源 | 作品 → 来源 |
| 改进 | 对某技术/产品进行改进 | 改进者 → 技术/产品 |
| 代表 | 代表某种风格/流派 | 作品/人物 → 风格/流派 |
| [其他] | 文本中自然出现的其他关系 | 根据语义判断 |

## 提取规则

1. **实体去重**：相同名称、相同类型的实体只出现一次
2. **关系去重**：相同 head、relation、tail 的关系只出现一条
3. **关系有效性**：head 和 tail 必须是已提取的实体
4. **禁止自环**：head 和 tail 不能是同一个实体
5. **适度提取**：每个文本块提取 8-25 个实体、10-30 条关系，宁多勿漏
6. **跨类型关系**：鼓励跨类型建立关系（如 PERSON→AWARD, WORK→MOVEMENT）
7. **中文规范**：实体名用中文，人名/地名保持原文

## 示例一：文学/小说类文本

输入：
"《红楼梦》是曹雪芹创作的长篇小说，中国古典四大名著之一。全书以贾宝玉和林黛玉的爱情悲剧为主线，描写了贾府的兴衰历程。曹雪芹（约1715年-1763年）是清代著名文学家，《红楼梦》未完是其一生最大的遗憾。脂砚斋批语揭示了曹雪芹创作背后的更多细节。"

正确输出：
```json
{
  "entities": [
    {"name": "红楼梦", "type": "WORK"},
    {"name": "曹雪芹", "type": "PERSON"},
    {"name": "贾宝玉", "type": "PERSON"},
    {"name": "林黛玉", "type": "PERSON"},
    {"name": "贾府", "type": "ORGANIZATION"},
    {"name": "清代", "type": "TIME"},
    {"name": "中国古典四大名著", "type": "CONCEPT"},
    {"name": "长篇小说", "type": "CONCEPT"},
    {"name": "爱情悲剧", "type": "EVENT"},
    {"name": "脂砚斋", "type": "PERSON"},
    {"name": "脂砚斋批语", "type": "OBJECT"}
  ],
  "relations": [
    {"head": "红楼梦", "relation": "创作", "tail": "曹雪芹"},
    {"head": "红楼梦", "relation": "类型", "tail": "长篇小说"},
    {"head": "红楼梦", "relation": "属于", "tail": "中国古典四大名著"},
    {"head": "红楼梦", "relation": "描写", "tail": "贾府"},
    {"head": "红楼梦", "relation": "主线", "tail": "爱情悲剧"},
    {"head": "曹雪芹", "relation": "朝代", "tail": "清代"},
    {"head": "曹雪芹", "relation": "创作", "tail": "红楼梦"},
    {"head": "爱情悲剧", "relation": "参与", "tail": "贾宝玉"},
    {"head": "爱情悲剧", "relation": "参与", "tail": "林黛玉"},
    {"head": "贾府", "relation": "位于", "tail": "清代"},
    {"head": "脂砚斋批语", "relation": "评论", "tail": "红楼梦"},
    {"head": "脂砚斋", "relation": "创作", "tail": "脂砚斋批语"}
  ]
}
```

## 示例二：电影/影视类文本

输入：
"《大话西游之大圣娶亲》是周星驰主演的古装爱情电影，1995年上映。刘镇伟执导，周星驰、吴孟达共同出演。影片由彩星电影公司出品，讲述至尊宝为救白晶晶穿越回五百年前，遇到紫霞仙子发生的一系列故事。"

正确输出：
```json
{
  "entities": [
    {"name": "大话西游之大圣娶亲", "type": "WORK"},
    {"name": "周星驰", "type": "PERSON"},
    {"name": "刘镇伟", "type": "PERSON"},
    {"name": "吴孟达", "type": "PERSON"},
    {"name": "彩星电影公司", "type": "ORGANIZATION"},
    {"name": "紫霞仙子", "type": "PERSON"},
    {"name": "至尊宝", "type": "PERSON"},
    {"name": "白晶晶", "type": "PERSON"},
    {"name": "古装爱情", "type": "CONCEPT"},
    {"name": "1995年", "type": "TIME"}
  ],
  "relations": [
    {"head": "大话西游之大圣娶亲", "relation": "主演", "tail": "周星驰"},
    {"head": "大话西游之大圣娶亲", "relation": "导演", "tail": "刘镇伟"},
    {"head": "大话西游之大圣娶亲", "relation": "类型", "tail": "古装爱情"},
    {"head": "大话西游之大圣娶亲", "relation": "发生于", "tail": "1995年"},
    {"head": "大话西游之大圣娶亲", "relation": "发行", "tail": "彩星电影公司"},
    {"head": "大话西游之大圣娶亲", "relation": "主演", "tail": "吴孟达"},
    {"head": "周星驰", "relation": "合作", "tail": "刘镇伟"},
    {"head": "周星驰", "relation": "合作", "tail": "吴孟达"},
    {"head": "大话西游之大圣娶亲", "relation": "讲述", "tail": "至尊宝"},
    {"head": "大话西游之大圣娶亲", "relation": "讲述", "tail": "紫霞仙子"},
    {"head": "刘镇伟", "relation": "导演", "tail": "大话西游之大圣娶亲"}
  ]
}
```

## 示例三：科技/技术类文本

输入：
"Python是由Guido van Rossum于1991年创建的高级编程语言。它以简洁易读的语法著称，广泛用于Web开发、数据科学、人工智能等领域。Python支持多种编程范式，包括面向对象编程和函数式编程。"

正确输出：
```json
{
  "entities": [
    {"name": "Python", "type": "PRODUCT"},
    {"name": "Guido van Rossum", "type": "PERSON"},
    {"name": "1991年", "type": "TIME"},
    {"name": "编程语言", "type": "CONCEPT"},
    {"name": "Web开发", "type": "CONCEPT"},
    {"name": "数据科学", "type": "CONCEPT"},
    {"name": "人工智能", "type": "CONCEPT"},
    {"name": "面向对象编程", "type": "TECHNOLOGY"},
    {"name": "函数式编程", "type": "TECHNOLOGY"},
    {"name": "高级编程语言", "type": "CONCEPT"}
  ],
  "relations": [
    {"head": "Python", "relation": "创建", "tail": "Guido van Rossum"},
    {"head": "Python", "relation": "发生于", "tail": "1991年"},
    {"head": "Python", "relation": "类型", "tail": "高级编程语言"},
    {"head": "Python", "relation": "应用于", "tail": "Web开发"},
    {"head": "Python", "relation": "应用于", "tail": "数据科学"},
    {"head": "Python", "relation": "应用于", "tail": "人工智能"},
    {"head": "Python", "relation": "使用", "tail": "面向对象编程"},
    {"head": "Python", "relation": "使用", "tail": "函数式编程"}
  ]
}
```

## 示例四：历史/事件类文本

输入：
"鸦片战争于1840年爆发，是英国为打开中国市场而发动的侵略战争。战争导致中国签定了《南京条约》，割让香港岛给英国，赔款2100万银元。这场战争深刻影响了中国的近代史走向。"

正确输出：
```json
{
  "entities": [
    {"name": "鸦片战争", "type": "EVENT"},
    {"name": "1840年", "type": "TIME"},
    {"name": "英国", "type": "ORGANIZATION"},
    {"name": "中国", "type": "LOCATION"},
    {"name": "南京条约", "type": "LAW"},
    {"name": "香港岛", "type": "LOCATION"},
    {"name": "2100万银元", "type": "MONEY"},
    {"name": "近代史", "type": "TIME"},
    {"name": "侵略战争", "type": "CONCEPT"}
  ],
  "relations": [
    {"head": "鸦片战争", "relation": "发生于", "tail": "1840年"},
    {"head": "鸦片战争", "relation": "参与", "tail": "英国"},
    {"head": "鸦片战争", "relation": "导致", "tail": "中国"},
    {"head": "鸦片战争", "relation": "类型", "tail": "侵略战争"},
    {"head": "鸦片战争", "relation": "引起", "tail": "南京条约"},
    {"head": "南京条约", "relation": "影响", "tail": "中国"},
    {"head": "南京条约", "relation": "导致", "tail": "香港岛"},
    {"head": "南京条约", "relation": "导致", "tail": "2100万银元"},
    {"head": "鸦片战争", "relation": "影响", "tail": "近代史"}
  ]
}
```

## 待处理文本
"""

        self._current_prompt_mode = "complaint"
        self._prompt_mode = "complaint"
        self._universal_entity_aliases = {}

    def set_prompt_mode(self, mode: str):
        """
        切换 Prompt 模式。

        Parameters
        ----------
        mode : str
            - "complaint" : 使用投诉数据专用 Prompt（崂山景区投诉工单）
            - "universal" : 使用通用 Prompt（适用于任何领域的文本，如书籍介绍、技术文档等）
        """
        if mode == "universal":
            self.ner_re_prompt = self.universal_ner_re_prompt
            self.ner_prompt = self.universal_ner_re_prompt
            self.re_prompt = self.universal_ner_re_prompt
            self._entity_aliases = self._universal_entity_aliases
            self._current_prompt_mode = "universal"
            self._prompt_mode = "universal"
            safe_print("[INFO] Prompt 模式切换为: universal（通用模式）")
        else:
            self.ner_re_prompt = self._complaint_ner_re_prompt
            self.ner_prompt = self._complaint_ner_re_prompt
            self.re_prompt = self._complaint_ner_re_prompt
            self._current_prompt_mode = "complaint"
            self._prompt_mode = "complaint"
            safe_print("[INFO] Prompt 模式切换为: complaint（投诉模式）")

    def get_prompt_mode(self) -> str:
        """返回当前 Prompt 模式。"""
        return self._current_prompt_mode

    def _normalize_entity_name(self, name: str) -> str:
        """实体名称归一化，减少同义词导致的图谱碎片。"""
        normalized = (name or "").strip()
        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.replace("（", "(").replace("）", ")")
        normalized = normalized.replace("“", "").replace("”", "").replace('"', "")
        return self._entity_aliases.get(normalized, normalized)

    def connect(self) -> bool:
        """连接Neo4j数据库"""
        try:
            from neo4j import GraphDatabase
            
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            self.is_connected = True
            safe_print(f"[OK] 成功连接到Neo4j: {self.uri}")
            return True
            
        except Exception as e:
            safe_print(f"[WARN] Neo4j连接失败: {e}")
            safe_print("[INFO] 将使用本地内存模式（演示用）")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开Neo4j连接"""
        if self.driver:
            self.driver.close()
            self.is_connected = False
            safe_print("[OK] 已断开Neo4j连接")

    def set_vector_store(self, vector_store):
        """设置向量库引用"""
        self.vector_store = vector_store

    def extract_entities_and_relations_with_llm(self, text: str, chunk_id: str = None) -> Tuple[List[Entity], List[Relation]]:
        """使用LLM从文本中提取实体和关系 - 无长度限制"""
        if not self.llm_service or not self.llm_service.is_initialized:
            raise Exception("LLM服务未初始化！知识图谱构建必须使用LLM进行实体关系提取。")

        def _log(msg):
            """安全的日志输出，兼容 Windows 控制台 Unicode"""
            try:
                safe_print(msg)
            except OSError:
                try:
                    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
                except Exception:
                    pass

        try:
            _log(f"\n  [LLM调用] 文本长度: {len(text)} 字符")

            # 构建完整prompt
            full_prompt = f"{self.ner_re_prompt}\n{text}"

            response = self.llm_service.generate_answer(
                full_prompt,
                system_prompt="你是一个JSON输出专家。只输出JSON，不要输出任何解释、说明或其他文字。"
            )

            raw = response if isinstance(response, str) else str(response)
            try:
                _log(f"\n  [LLM原始输出] ({len(raw)}字):\n{raw[:2000]}")
                if len(raw) > 2000:
                    _log(f"  [LLM原始输出] (...共{len(raw)}字，省略中间部分...)\n{repr(raw[-500:])}")
            except OSError:
                _log(f"  [LLM原始输出] 长度: {len(raw)} 字 (输出包含敏感字符，已隐藏)")
                _log(f"  原始内容 repr: {repr(raw[-500:])}")

            entities, relations = self._parse_combined_response(response, chunk_id)

            _log(f"  [DEBUG] 解析后: {len(entities)}实体, {len(relations)}关系")

            entities = self._validate_entities(entities)
            relations = self._validate_relations(relations, entities)

            _log(f"[OK] LLM提取完成: {len(entities)}个实体, {len(relations)}个关系")
            return entities, relations

        except Exception as e:
            _log(f"[ERROR] LLM提取失败: {e}")
            try:
                import traceback
                traceback.print_exc()
            except OSError:
                try:
                    import traceback, io
                    sio = io.StringIO()
                    traceback.print_exc(file=sio)
                    safe = sio.getvalue().encode("utf-8", errors="replace").decode("utf-8", errors="replace")
                    sys.stdout.buffer.write(safe.encode("utf-8", errors="replace"))
                except Exception:
                    pass
            raise

    def _parse_combined_response(self, response_text: str, chunk_id: str = None) -> Tuple[List[Entity], List[Relation]]:
        """解析合并的NER+RE响应 - 与诊断脚本一致"""
        entities = []
        relations = []
        
        try:
            text = response_text.strip()
            
            # 去除markdown代码块标记 - 正确处理方式
            if text.startswith('```'):
                lines = text.split('\n')
                start_idx = 0
                end_idx = len(lines) - 1
                for i, line in enumerate(lines):
                    if line.strip().startswith('```'):
                        start_idx = i + 1
                        break
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == '```':
                        end_idx = i
                        break
                text = '\n'.join(lines[start_idx:end_idx])
                text = text.strip()
                safe_print(f"  [INFO] 去除markdown后文本长度: {len(text)}")
            
            # 方法1: 直接解析整个响应为JSON
            try:
                data = json.loads(text)
                safe_print(f"  [OK] 直接解析成功，类型: {type(data)}")
                
                if isinstance(data, dict):
                    if 'entities' in data:
                        safe_print(f"  [OK] 找到entities字段: {len(data['entities'])} 个")
                    if 'relations' in data:
                        safe_print(f"  [OK] 找到relations字段: {len(data['relations'])} 个")
                    self._extract_from_dict(data, entities, relations, chunk_id)
                    if entities or relations:
                        safe_print(f"  [OK] 从dict提取完成: {len(entities)}实体, {len(relations)}关系")
                        return entities, relations
                        
                elif isinstance(data, list):
                    safe_print(f"  [INFO] 响应是数组，包含 {len(data)} 个元素")
                    for idx, item in enumerate(data):
                        if isinstance(item, dict):
                            if 'entities' in item:
                                for e in item['entities']:
                                    if isinstance(e, dict) and 'name' in e and 'type' in e:
                                        entity = Entity(
                                            name=str(e['name']).strip(),
                                            type=str(e['type']).strip().upper(),
                                            properties={k: v for k, v in e.items() if k not in ['name', 'type']}
                                        )
                                        entity.vector_id = chunk_id
                                        entities.append(entity)
                            if 'relations' in item:
                                for r in item['relations']:
                                    if isinstance(r, dict) and 'head' in r and 'relation' in r and 'tail' in r:
                                        relation = Relation(
                                            head=str(r['head']).strip(),
                                            relation=str(r['relation']).strip().upper(),
                                            tail=str(r['tail']).strip(),
                                            confidence=float(r.get('confidence', 0.8))
                                        )
                                        relation.source_chunk_id = chunk_id
                                        relations.append(relation)
                    if entities or relations:
                        safe_print(f"  [OK] 从数组提取完成: {len(entities)}实体, {len(relations)}关系")
                        return entities, relations
                        
            except json.JSONDecodeError as e:
                safe_print(f"  [WARN] 直接解析失败: {e}")
            
            # 方法2: 找到第一个完整的JSON对象
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = text[first_brace:last_brace+1]
                safe_print(f"  [INFO] 尝试提取JSON对象: 位置 {first_brace}-{last_brace}")
                try:
                    data = json.loads(json_str)
                    safe_print(f"  [OK] 提取JSON成功")
                    if isinstance(data, dict):
                        self._extract_from_dict(data, entities, relations, chunk_id)
                        if entities or relations:
                            safe_print(f"  [OK] 从提取的JSON完成: {len(entities)}实体, {len(relations)}关系")
                            return entities, relations
                except Exception as e:
                    safe_print(f"  [WARN] JSON对象解析失败: {e}")
            
            # 方法3: 找到JSON数组
            first_bracket = text.find('[')
            last_bracket = text.rfind(']')
            
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                json_str = text[first_bracket:last_bracket+1]
                safe_print(f"  [INFO] 尝试提取JSON数组: 位置 {first_bracket}-{last_bracket}")
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        safe_print(f"  [OK] 数组解析成功，包含 {len(data)} 个元素")
                        for item in data:
                            if isinstance(item, dict):
                                if 'entities' in item:
                                    for e in item['entities']:
                                        if isinstance(e, dict) and 'name' in e and 'type' in e:
                                            entity = Entity(
                                                name=str(e['name']).strip(),
                                                type=str(e['type']).strip().upper(),
                                                properties={}
                                            )
                                            entity.vector_id = chunk_id
                                            entities.append(entity)
                                if 'relations' in item:
                                    for r in item['relations']:
                                        if isinstance(r, dict) and 'head' in r and 'relation' in r and 'tail' in r:
                                            relation = Relation(
                                                head=str(r['head']).strip(),
                                                relation=str(r['relation']).strip().upper(),
                                                tail=str(r['tail']).strip(),
                                                confidence=0.8
                                            )
                                            relation.source_chunk_id = chunk_id
                                            relations.append(relation)
                        if entities or relations:
                            safe_print(f"  [OK] 从数组提取完成: {len(entities)}实体, {len(relations)}关系")
                            return entities, relations
                except Exception as e:
                    safe_print(f"  [WARN] 数组解析失败: {e}")
            
            # 方法4: 正则提取entities和relations
            entities_match = re.search(r'"entities"\s*:\s*(\[.*?\])\s*,?\s*"relations"', text, re.DOTALL)
            if entities_match:
                try:
                    ent_data = json.loads(entities_match.group(1))
                    safe_print(f"  [OK] 正则提取entities: {len(ent_data)} 个")
                    for item in ent_data:
                        if isinstance(item, dict) and 'name' in item and 'type' in item:
                            entity = Entity(
                                name=str(item['name']).strip(),
                                type=str(item['type']).strip().upper(),
                                properties={}
                            )
                            entity.vector_id = chunk_id
                            entities.append(entity)
                except Exception as e:
                    safe_print(f"  [WARN] 正则提取entities失败: {e}")
            
            relations_match = re.search(r'"relations"\s*:\s*(\[.*?\])\s*}', text, re.DOTALL)
            if relations_match:
                try:
                    rel_data = json.loads(relations_match.group(1))
                    safe_print(f"  [OK] 正则提取relations: {len(rel_data)} 个")
                    for item in rel_data:
                        if isinstance(item, dict) and 'head' in item and 'relation' in item and 'tail' in item:
                            relation = Relation(
                                head=str(item['head']).strip(),
                                relation=str(item['relation']).strip().upper(),
                                tail=str(item['tail']).strip(),
                                confidence=0.8
                            )
                            relation.source_chunk_id = chunk_id
                            relations.append(relation)
                except Exception as e:
                    safe_print(f"  [WARN] 正则提取relations失败: {e}")
            
            if entities or relations:
                safe_print(f"  [OK] 最终解析结果: {len(entities)}实体, {len(relations)}关系")

        except Exception as e:
            safe_print(f"[ERROR] 解析响应异常: {e}")
            import traceback
            traceback.print_exc()

        return entities, relations
    
    def _extract_from_dict(self, data: Dict, entities: List[Entity], relations: List[Relation], chunk_id: str = None):
        """从字典中提取实体和关系"""
        if 'entities' in data and isinstance(data['entities'], list):
            for item in data['entities']:
                if isinstance(item, dict) and 'name' in item and 'type' in item:
                    entity = Entity(
                        name=str(item['name']).strip(),
                        type=str(item['type']).strip().upper(),
                        properties={k: v for k, v in item.items() if k not in ['name', 'type']}
                    )
                    entity.vector_id = chunk_id
                    entities.append(entity)

        if 'relations' in data and isinstance(data['relations'], list):
            for item in data['relations']:
                if isinstance(item, dict) and 'head' in item and 'relation' in item and 'tail' in item:
                    relation = Relation(
                        head=str(item['head']).strip(),
                        relation=str(item['relation']).strip().upper(),
                        tail=str(item['tail']).strip(),
                        confidence=float(item.get('confidence', 0.8)),
                        properties={k: v for k, v in item.items() if k not in ['head', 'relation', 'tail', 'confidence']}
                    )
                    relation.source_chunk_id = chunk_id
                    relations.append(relation)

    def _validate_entities(self, entities: List[Entity]) -> List[Entity]:
        """验证实体 - 只保留最终出现在关系中的有效实体"""
        valid = []
        seen = {}

        for entity in entities:
            name = self._normalize_entity_name(entity.name)

            if not name or not name.strip():
                continue
            if len(name) < 2:
                continue
            if name.strip().isdigit():
                continue
            if not any(c.isalnum() for c in name):
                continue
            if len(name) > 30:
                continue
            if name.endswith(('。', '！', '？')) and len(name) > 15:
                continue

            entity.name = name
            key = name
            if key not in seen:
                seen[key] = entity
                valid.append(entity)

        return valid

    def _validate_relations(self, relations: List[Relation], entities: List[Entity]) -> List[Relation]:
        """验证关系 + 过滤无效关系"""
        valid = []

        # 关系类型归一化映射
        relation_type_map = {
            '处理': '处理', 'HANDLED_BY': '处理', 'CHULI': '处理',
            '涉及部门': '涉及部门', '涉及': '涉及',
            'ABOUT': '涉及', '涉及对象': '涉及', '涉及方': '涉及',
            '发生地': '发生地', 'HAPPENED_AT': '发生地', '发生地点': '发生地',
            '关联': '关联', 'RELATED_TO': '关联', '关联到': '关联',
            '反映人': '反映人', 'REPORTER': '反映人',
            '时间': '时间', 'TIME': '时间', 'AT_TIME': '时间',
            '属于': '属于', 'BELONGS_TO': '属于',
            '咨询': '涉及', '投诉': '涉及', '收费': '涉及',
            '游览': '涉及', 'VISITED': '涉及',
            '同投诉人': '同投诉人', '同部门': '同部门', '同地点': '同地点',
            '涉及人': '涉及人', 'INVOLVES_PERSON': '涉及人',
            '乘坐': '乘坐', 'RIDES': '乘坐',
            '区域': '区域', 'AREA': '区域',
            '主题': '主题', 'THEME': '主题',
            '具体事件': '具体事件', 'EVENT': '具体事件',
            '涉及问题': '涉及问题', '涉及金额': '涉及金额',
            '对方': '对方',
        }

        # 构建实体名->类型的快速查找表
        entity_name_to_type = {}
        for e in entities:
            entity_name_to_type[e.name] = e.type

        for rel in relations:
            original_relation = rel.relation
            rel.relation = relation_type_map.get(rel.relation, '关联')
            rel.head = self._normalize_entity_name(rel.head)
            rel.tail = self._normalize_entity_name(rel.tail)

            # 基础过滤
            if not rel.head or not rel.head.strip() or not rel.tail or not rel.tail.strip():
                continue
            if len(rel.head) < 2 or len(rel.tail) < 2:
                continue
            if len(rel.head) > 30 or len(rel.tail) > 30:
                continue
            if rel.head.strip().isdigit() or rel.tail.strip().isdigit():
                continue
            if rel.head == rel.tail:
                continue

            # 类型约束：head/tail 必须是已知实体
            head_type = entity_name_to_type.get(rel.head)
            tail_type = entity_name_to_type.get(rel.tail)

            # 禁止的关系类型组合（无意义的反射/类型错配）
            bad_patterns = [
                # 反射关系：head和tail是同一类型的同一实体
                (rel.head == rel.tail, "反射关系"),
                # PERSON不能是时间或地点
                (head_type == 'PERSON' and tail_type in ('TIME',), "PERSON不能是时间"),
                # TIME/ORGANIZATION不能是发生地
                (rel.relation == '发生地' and head_type in ('TIME', 'ORGANIZATION', 'TYPE'), "发生地head类型错误"),
                # 处理关系的tail必须是ORGANIZATION
                (rel.relation == '处理' and tail_type not in ('ORGANIZATION', None), "处理关系的tail必须是ORGANIZATION"),
                # 反映人的head必须是PERSON
                (rel.relation == '反映人' and head_type not in ('PERSON', None), "反映人的head必须是PERSON"),
                # 属于关系的tail必须是TYPE
                (rel.relation == '属于' and tail_type not in ('TYPE', None), "属于的tail必须是TYPE"),
            ]
            skip = False
            for condition, reason in bad_patterns:
                if condition:
                    skip = True
                    break

            if skip:
                continue

            valid.append(rel)

        # 过滤孤立实体：只保留出现在关系中的实体
        connected_names = set()
        for rel in valid:
            connected_names.add(rel.head)
            connected_names.add(rel.tail)

        connected_entities = [e for e in entities if e.name in connected_names]
        self._temp_connected_entities = connected_entities

        safe_print(f"  [DEBUG] 关系验证后: {len(valid)}条关系, 孤立过滤后: {len(connected_entities)}个实体")
        return valid

    def _get_connected_entities(self) -> List[Entity]:
        """返回当前批次中有连接的实体"""
        return getattr(self, '_temp_connected_entities', [])

    def _build_cross_cluster_relations(
        self,
        entities: List[Entity],
        relations: List[Relation],
        chunks: List[Dict]
    ) -> List[Relation]:
        """构建簇间关系：同一投诉人/同一部门/同一地点的多条工单建立关联"""
        cross_rels = []
        seen_rels = set()

        # 构建工单->人的映射 (通过反映人关系找)
        complaint_to_person = {}
        complaint_to_org = {}
        complaint_to_loc = {}
        for rel in relations:
            if rel.relation == '反映人':
                complaint_to_person[rel.tail] = rel.head  # tail=工单, head=人
            elif rel.relation == '处理':
                complaint_to_org[rel.tail] = rel.head   # tail=工单, head=部门
            elif rel.relation == '发生地':
                if rel.tail not in complaint_to_loc:
                    complaint_to_loc[rel.tail] = []
                complaint_to_loc[rel.tail].append(rel.head)

        # 1. 同投诉人关系：同一人的多条工单之间
        person_to_complaints = {}
        for comp, person in complaint_to_person.items():
            if person not in person_to_complaints:
                person_to_complaints[person] = []
            person_to_complaints[person].append(comp)

        for person, complaints in person_to_complaints.items():
            if len(complaints) < 2:
                continue
            # 对同一人的工单两两建立关系（最多每组5条，避免爆炸）
            for i in range(min(len(complaints) - 1, 4)):
                for j in range(i + 1, len(complaints)):
                    key = (complaints[i], '同投诉人', complaints[j])
                    if key not in seen_rels:
                        seen_rels.add(key)
                        cross_rels.append(Relation(
                            head=complaints[i],
                            relation='同投诉人',
                            tail=complaints[j],
                            confidence=0.9
                        ))

        # 2. 同部门关系：同一部门处理的多条工单之间
        org_to_complaints = {}
        for comp, org in complaint_to_org.items():
            if org not in org_to_complaints:
                org_to_complaints[org] = []
            org_to_complaints[org].append(comp)

        for org, complaints in org_to_complaints.items():
            if len(complaints) < 2:
                continue
            for i in range(min(len(complaints) - 1, 4)):
                for j in range(i + 1, len(complaints)):
                    key = (complaints[i], '同部门', complaints[j])
                    if key not in seen_rels:
                        seen_rels.add(key)
                        cross_rels.append(Relation(
                            head=complaints[i],
                            relation='同部门',
                            tail=complaints[j],
                            confidence=0.7
                        ))

        # 3. 同地点关系：同一地点发生的多条工单之间
        loc_to_complaints = {}
        for loc, complaints in complaint_to_loc.items():
            for comp in complaints:
                if loc not in loc_to_complaints:
                    loc_to_complaints[loc] = []
                loc_to_complaints[loc].append(comp)

        for loc, complaints in loc_to_complaints.items():
            if len(complaints) < 2:
                continue
            for i in range(min(len(complaints) - 1, 4)):
                for j in range(i + 1, len(complaints)):
                    key = (complaints[i], '同地点', complaints[j])
                    if key not in seen_rels:
                        seen_rels.add(key)
                        cross_rels.append(Relation(
                            head=complaints[i],
                            relation='同地点',
                            tail=complaints[j],
                            confidence=0.7
                        ))

        safe_print(f"  [簇间关系] 同投诉人:{sum(1 for r in cross_rels if r.relation=='同投诉人')} "
              f"同部门:{sum(1 for r in cross_rels if r.relation=='同部门')} "
              f"同地点:{sum(1 for r in cross_rels if r.relation=='同地点')}")
        return cross_rels

    def _parse_llm_json_response(self, response_text: str, expected_type: str) -> List:
        """解析LLM返回的JSON"""
        try:
            text = response_text.strip()
            
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return self._convert_to_objects(data, expected_type)
            except:
                pass
            
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                data = json.loads(json_match.group())
                return self._convert_to_objects(data, expected_type)
            
            safe_print(f"[WARN] 未能从响应中提取有效JSON")
            return []
            
        except json.JSONDecodeError as e:
            safe_print(f"[WARN] JSON解析错误: {e}")
            return []
        except Exception as e:
            safe_print(f"[WARN] 解析异常: {e}")
            return []
    
    def _convert_to_objects(self, data: list, expected_type: str) -> List:
        """将字典列表转换为Entity或Relation对象"""
        result = []
        
        for item in data:
            if not isinstance(item, dict):
                continue
                
            try:
                if expected_type == "entities":
                    if 'name' in item and 'type' in item:
                        entity = Entity(
                            name=str(item['name']).strip(),
                            type=str(item['type']).strip().upper(),
                            properties={k: v for k, v in item.items() if k not in ['name', 'type']}
                        )
                        result.append(entity)
                        
                elif expected_type == "relations":
                    if 'head' in item and 'relation' in item and 'tail' in item:
                        relation = Relation(
                            head=str(item['head']).strip(),
                            relation=str(item['relation']).strip().upper(),
                            tail=str(item['tail']).strip(),
                            confidence=float(item.get('confidence', 0.8)),
                            properties={k: v for k, v in item.items() if k not in ['head', 'relation', 'tail', 'confidence']}
                        )
                        result.append(relation)
                        
            except Exception as e:
                safe_print(f"[WARN] 转换对象失败: {e}")
                continue
        
        return result

    def add_to_neo4j(self, entities: List[Entity], relations: List[Relation]) -> int:
        """将实体和关系添加到Neo4j"""
        self.local_entities.extend(entities)
        self.local_relations.extend(relations)
        
        if self.is_connected and self.driver:
            try:
                with self.driver.session() as session:
                    for entity in entities:
                        session.run("""
                            MERGE (e:Entity {name: $name})
                            SET e.type = $type,
                                e.properties = $properties,
                                e.vector_id = $vector_id,
                                e.updated_at = datetime()
                            RETURN e
                        """, 
                            name=entity.name,
                            type=entity.type,
                            properties=json.dumps(entity.properties, ensure_ascii=False),
                            vector_id=entity.vector_id or ""
                        )
                    
                    count = 0
                    for rel in relations:
                        safe_relation = re.sub(r'[^A-Z0-9_]', '_', rel.relation.upper())
                        if not safe_relation:
                            safe_relation = "RELATED_TO"

                        cypher = f"""
                            MATCH (h:Entity {{name: $head}})
                            MATCH (t:Entity {{name: $tail}})
                            MERGE (h)-[r:`{safe_relation}`]->(t)
                            SET r.confidence = $confidence,
                                r.properties = $properties,
                                r.source_chunk_id = $source_chunk_id,
                                r.updated_at = datetime()
                            RETURN r
                        """
                        session.run(cypher,
                            head=rel.head,
                            tail=rel.tail,
                            confidence=rel.confidence,
                            properties=json.dumps(rel.properties, ensure_ascii=False),
                            source_chunk_id=rel.source_chunk_id or ""
                        )
                        count += 1
                    
                    safe_print(f"[OK] 同步到Neo4j完成: {len(entities)}个节点, {count}条边")
                    return count
                    
            except Exception as e:
                safe_print(f"[ERROR] Neo4j写入失败: {e}")
                safe_print("[INFO] 数据已保存在本地缓存中")
        
        return len(relations)

    def build_knowledge_graph_from_documents(self, chunks: List[Dict], progress_callback=None) -> Dict:
        """从文档块批量构建知识图谱 - 分批处理，支持大量数据"""
        if not self.llm_service or not self.llm_service.is_initialized:
            safe_print("[WARN] LLM服务未初始化或未连接，将跳过图谱构建。向量库已成功构建，可直接进行向量检索。")
            return {
                'unique_entities': 0,
                'unique_relations': 0,
                'total_chunks_input': len(chunks),
                'llm_calls_made': 0,
                'elapsed_seconds': 0.0,
                'local_entities': 0,
                'local_relations': 0,
                'note': 'LLM未初始化，图谱构建已跳过'
            }

        import time
        start_time = time.time()
        
        total_chunks = len(chunks)
        safe_print(f"\n{'='*60}")
        safe_print(f"开始构建知识图谱 (分批处理模式)")
        safe_print(f"文档块总数: {total_chunks}")
        safe_print(f"{'='*60}\n")

        # 分批处理参数
        BATCH_SIZE = 5  # 每批5条，确保LLM每次处理内容适量，质量更好
        total_entities = 0
        total_relations = 0
        all_entities = []
        all_relations = []
        llm_calls = 0

        # 计算批次数
        num_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
        safe_print(f"[INFO] 将分 {num_batches} 批处理，每批 {BATCH_SIZE} 条投诉")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, total_chunks)
            batch_chunks = chunks[start_idx:end_idx]

            safe_print(f"\n{'='*60}")
            safe_print(f"[批次 {batch_idx + 1}/{num_batches}] 处理第 {start_idx + 1}-{end_idx} 条投诉")
            safe_print(f"{'='*60}")

            if progress_callback:
                progress_callback(batch_idx + 1, num_batches + 1, f"正在处理批次 {batch_idx + 1}/{num_batches}...")

            # 合并当前批次的文本
            batch_texts = []
            for chunk in batch_chunks:
                text = chunk.get('text', '')
                if len(text.strip()) >= 10:
                    batch_texts.append(text)

            if not batch_texts:
                safe_print(f"  [SKIP] 批次文本为空，跳过")
                continue

            combined_text = "\n\n".join(batch_texts)
            safe_print(f"  [批次文本] 共{len(batch_texts)}条, {len(combined_text)}字")

            try:
                entities, relations = self.extract_entities_and_relations_with_llm(
                    combined_text,
                    chunk_id=f"batch_{batch_idx + 1}"
                )
                llm_calls += 1

                safe_print(f"\n  [本批次结果] 提取: {len(entities)}实体, {len(relations)}关系")
                if relations:
                    all_relations.extend(relations)
                    connected = self._get_connected_entities()
                    for e in connected:
                        key = (e.name, e.type)
                        if key not in {(_e.name, _e.type) for _e in all_entities}:
                            all_entities.append(e)

            except Exception as e:
                safe_print(f"  [WARN] 批次 {batch_idx + 1} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 去重
        safe_print(f"\n[INFO] 合并并去重...")
        unique_entities = {}
        for e in all_entities:
            key = (e.name, e.type)
            if key not in unique_entities:
                unique_entities[key] = e

        unique_relations = []
        seen_relations = set()
        for r in all_relations:
            key = (r.head, r.relation, r.tail)
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(r)

        # 二次过滤：只保留出现在关系中的实体
        connected_names = set()
        for rel in unique_relations:
            connected_names.add(rel.head)
            connected_names.add(rel.tail)

        final_entities = [e for e in unique_entities.values() if e.name in connected_names]
        safe_print(f"[OK] 去重后: {len(final_entities)}实体(全部有连接), {len(unique_relations)}关系")

        # ── 簇间关系构建 ──
        # 规则：同一PERSON（LIGHT）的多条工单之间建立"同投诉人"关系
        # 同一ORGANIZATION处理的多条工单之间建立"同部门"关系
        cross_relations = self._build_cross_cluster_relations(
            final_entities, unique_relations, chunks
        )
        unique_relations.extend(cross_relations)
        safe_print(f"[INFO] 簇间关系: +{len(cross_relations)}条")

        # 存储到知识图谱
        if final_entities or unique_relations:
            self.add_to_neo4j(final_entities, unique_relations)
            total_entities = len(final_entities)
            total_relations = len(unique_relations)
        
        if progress_callback:
            progress_callback(num_batches + 1, num_batches + 1, "执行社团划分...")

        communities = self.detect_communities()
        
        elapsed_time = time.time() - start_time

        stats = {
            'total_chunks_input': total_chunks,
            'total_entities': total_entities,
            'total_relations': total_relations,
            'unique_entities': len(set(e.name for e in self.local_entities)),
            'communities': len(communities) if communities else 0,
            'storage_mode': 'Neo4j' if self.is_connected else 'Local Memory',
            'build_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'extraction_method': 'LLM (分批处理-严格验证)',
            'llm_calls_made': llm_calls,
            'batches_processed': num_batches,
            'elapsed_seconds': round(elapsed_time, 2),
            'optimization_mode': True,
            'prompt_mode': self._prompt_mode,
        }

        safe_print(f"\n{'='*60}")
        safe_print(f"知识图谱构建完成！(分批处理模式)")
        safe_print(f"{'='*60}")
        safe_print(f"输入文档块: {stats['total_chunks_input']}")
        safe_print(f"处理批次数: {stats['batches_processed']}")
        safe_print(f"LLM调用次数: {stats['llm_calls_made']} 次")
        safe_print(f"提取实体总数: {stats['total_entities']}")
        safe_print(f"提取关系总数: {stats['total_relations']}")
        safe_print(f"唯一实体数: {stats['unique_entities']}")
        safe_print(f"检测社团数: {stats['communities']}")
        safe_print(f"实际耗时: {stats['elapsed_seconds']}秒 ({stats['elapsed_seconds']/60:.1f}分钟)")
        safe_print(f"存储模式: {stats['storage_mode']}")
        safe_print(f"{'='*60}")

        return stats

    def detect_communities(self) -> Dict[str, List[str]]:
        """使用NetworkX进行社团划分"""
        import networkx as nx
        from collections import defaultdict
        
        G = nx.DiGraph()
        
        # 过滤：只保留出现在关系中的节点（孤立节点不加入社团图）
        connected_names = set()
        for rel in self.local_relations:
            connected_names.add(rel.head)
            connected_names.add(rel.tail)

        for entity in self.local_entities:
            if entity.name in connected_names:
                G.add_node(entity.name, type=entity.type, vector_id=entity.vector_id)
        
        for rel in self.local_relations:
            if G.has_node(rel.head) and G.has_node(rel.tail):
                G.add_edge(rel.head, rel.tail, 
                           relation=rel.relation, 
                           weight=rel.confidence)
        
        if G.number_of_nodes() == 0:
            safe_print("[WARN] 图谱为空，无法进行社团划分")
            return {}
        
        G_undirected = G.to_undirected()
        
        try:
            from networkx.algorithms.community import louvain_communities
            communities_list = louvain_communities(G_undirected, weight='weight')
            
            communities = {}
            for idx, community in enumerate(communities_list):
                community_id = f"Community_{idx+1}"
                communities[community_id] = list(community)
                
                abstract_keywords = self._generate_abstract_keywords(list(community))
                
                for node in community:
                    if G.has_node(node):
                        G.nodes[node]['community'] = community_id
                        G.nodes[node]['abstract_keywords'] = abstract_keywords
            
            safe_print(f"[OK] 社团划分完成: 检测到{len(communities)}个社团")
            
            self.communities = communities
            self.nx_graph = G
            
            return communities
            
        except Exception as e:
            safe_print(f"[WARN] 社团划分失败: {e}")
            return {}

    def _generate_abstract_keywords(self, entity_names: List[str]) -> List[str]:
        """为实体生成抽象关键词"""
        type_counts = defaultdict(int)
        for name in entity_names:
            for entity in self.local_entities:
                if entity.name == name:
                    type_counts[entity.type] += 1
                    break
        
        type_to_keyword = {
            'PERSON': ['人物'],
            'LOCATION': ['地点'],
            'ORGANIZATION': ['组织'],
            'OBJECT': ['事物'],
            'EVENT': ['事件'],
            'CONCEPT': ['概念'],
            'ISSUE': ['问题']
        }
        
        keywords = []
        for entity_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            if entity_type in type_to_keyword:
                keywords.extend(type_to_keyword[entity_type][:1])
        
        return keywords[:3] if keywords else ['通用']

    def hybrid_query(self, query: str, top_k_graph: int = 3, top_k_vector: int = 3) -> Dict:
        """综合查询方法"""
        result = {
            'query': query,
            'graph_results': [],
            'vector_results': [],
            'fused_results': [],
            'community_context': {},
            'explanation': ''
        }
        vector_results = []
        
        graph_paths = self.multi_hop_search(query, max_hops=2, top_k=top_k_graph)
        result['graph_results'] = graph_paths
        
        if self.vector_store and hasattr(self.vector_store, 'similarity_search'):
            vector_results = self.vector_store.similarity_search(query, top_k=top_k_vector)
            result['vector_results'] = vector_results
        
        fused = self._fuse_results(graph_paths, vector_results)
        result['fused_results'] = fused
        
        if hasattr(self, 'communities') and self.communities:
            community_ctx = self._get_community_context(query)
            result['community_context'] = community_ctx
        
        explanation = self._generate_query_explanation(result)
        result['explanation'] = explanation

        return result

    def search_relations(self, head: str = None, tail: str = None,
                         relation: str = None) -> List[Dict]:
        """
        查找满足条件的三元组（用于多跳推理）

        Args:
            head: 头实体名（模糊匹配）
            tail: 尾实体名（模糊匹配）
            relation: 关系类型

        Returns:
            [{'head', 'relation', 'tail'}, ...]
        """
        results = []

        # 先查Neo4j
        if self.is_connected and self.driver:
            try:
                with self.driver.session() as session:
                    if head and relation:
                        cypher = """
                            MATCH (a:Entity {name: $head})-[r]->(b:Entity)
                            WHERE r.type = $relation
                            RETURN a.name as head, r.type as relation, b.name as tail
                            LIMIT 50
                        """
                        recs = session.run(cypher, head=head, relation=relation)
                        for r in recs:
                            results.append({'head': r['head'], 'relation': r['relation'], 'tail': r['tail']})
                    elif tail and relation:
                        cypher = """
                            MATCH (a:Entity)-[r]->(b:Entity {name: $tail})
                            WHERE r.type = $relation
                            RETURN a.name as head, r.type as relation, b.name as tail
                            LIMIT 50
                        """
                        recs = session.run(cypher, tail=tail, relation=relation)
                        for r in recs:
                            results.append({'head': r['head'], 'relation': r['relation'], 'tail': r['tail']})
                    elif head:
                        cypher = """
                            MATCH (a:Entity {name: $head})-[r]->(b:Entity)
                            RETURN a.name as head, r.type as relation, b.name as tail
                            LIMIT 50
                        """
                        recs = session.run(cypher, head=head)
                        for r in recs:
                            results.append({'head': r['head'], 'relation': r['relation'], 'tail': r['tail']})
            except Exception as e:
                safe_print(f"[WARN] Neo4j search_relations失败: {e}")

        # 回退到本地内存
        if not results:
            for rel in self.local_relations:
                match = True
                if head and head not in rel.head:
                    match = False
                if tail and tail not in rel.tail:
                    match = False
                if relation and relation != rel.relation:
                    match = False
                if match:
                    results.append({'head': rel.head, 'relation': rel.relation, 'tail': rel.tail})

        return results

    # ============================================================
    # 三层实体链接
    # ============================================================

    def link_query_to_entities(self, query: str) -> List[Dict]:
        """
        三层实体链接：
        Layer 1: 同义词扩展（规则）
        Layer 2: 向量相似度匹配（嵌入向量）
        Layer 3: 图谱关系验证（检查实体是否真正在图中）
        """
        linked = []  # [{'entity': str, 'type': str, 'layer': int, 'confidence': float}]

        # ── Layer 1: 同义词扩展 ──
        SYNONYMS = {
            "景区大巴": ["大巴", "观光大巴", "旅游大巴", "景区车"],
            "大巴": ["景区大巴", "观光大巴"],
            "乱收费": ["收费过高", "不合理收费", "重复收费", "欺诈收费"],
            "景区": ["崂山景区", "崂山", "风景区", "旅游景区"],
            "刘镇伟": ["刘镇伟"],
            "周星驰": ["星爷", "周星驰"],
            "周星驰": ["周星驰", "星爷"],
        }

        # 从查询中提取关键词（人名/电影名/普通词）
        # 人名模式：2-4字中文 + 先生/女士/导演/演员
        person_pattern = re.findall(r'[\u4e00-\u9fa5]{2,4}(?:先生|女士|导演|演员)?', query)
        # 电影名模式
        movie_pattern = re.findall(r'《([^》]+)》', query)
        # 普通词
        words = re.findall(r'[\u4e00-\u9fa5]{2,6}', query)

        candidates = list(dict.fromkeys(person_pattern + movie_pattern + words))

        for cand in candidates:
            if len(cand) < 2:
                continue
            # 检查是否是已知实体的子串
            for known in [e.name for e in self.local_entities]:
                if cand in known or known in cand:
                    if cand in SYNONYMS:
                        linked.append({
                            'entity': cand,
                            'canonical': known,
                            'layer': 1,
                            'confidence': 0.9
                        })
                        break
                    else:
                        linked.append({
                            'entity': cand,
                            'canonical': known,
                            'layer': 1,
                            'confidence': 0.8
                        })
                        break

        # ── Layer 2: 向量相似度（如果向量库存在） ──
        if self.vector_store and self.vector_store.is_initialized:
            try:
                entity_names = [e.name for e in self.local_entities]
                if entity_names:
                    query_vec = self.vector_store.model.encode([query])
                    entity_vecs = self.vector_store.model.encode(entity_names)
                    from sklearn.metrics.pairwise import cosine_similarity
                    scores = cosine_similarity(query_vec, entity_vecs)[0]
                    for i, score in enumerate(scores):
                        if score > 0.6:
                            name = entity_names[i]
                            if not any(l['canonical'] == name for l in linked):
                                linked.append({
                                    'entity': name,
                                    'canonical': name,
                                    'layer': 2,
                                    'confidence': float(score)
                                })
            except Exception as e:
                safe_print(f"[WARN] 向量实体链接失败: {e}")

        # ── Layer 3: 图谱关系验证 ──
        # 确保所有linked实体的canonical版本在图中确实存在
        verified = []
        known_names = set(e.name for e in self.local_entities)
        for item in linked:
            canonical = item['canonical']
            if canonical in known_names:
                # 找到该实体的类型
                for e in self.local_entities:
                    if e.name == canonical:
                        item['type'] = e.type
                        break
                verified.append(item)

        # 按 confidence 排序
        verified.sort(key=lambda x: x['confidence'], reverse=True)
        return verified

    def expand_query_entities(self, query: str) -> List[str]:
        """将查询中的实体扩展为所有可能的同义词/候选"""
        linked = self.link_query_to_entities(query)
        expanded = set()
        for item in linked:
            expanded.add(item['canonical'])
        # 添加原始关键词
        for word in re.findall(r'[\u4e00-\u9fa5]{2,6}', query):
            expanded.add(word)
        return list(expanded)

    def entity_alignment(self) -> Dict:
        """
        自动化实体对齐（实体消解）

        策略：
        1. 字符串相似度（编辑距离 + 相同子串）
        2. 相同 @ 时间戳 的人物实体保留带时间的版本
        3. 设置等价关系写入 Neo4j

        Returns:
            {'merged': int, 'alignments': [{'canonical': str, 'merged': [str]}]}
        """
        import itertools

        alignments = {}  # canonical_name -> [alias1, alias2, ...]
        processed = set()

        entities = list(self.local_entities)
        for e1, e2 in itertools.combinations(entities, 2):
            if id(e1) in processed or id(e2) in processed:
                continue
            n1, n2 = e1.name, e2.name
            if not n1 or not n2:
                continue

            # 策略1: 相同子串（一方完全包含另一方）
            if n1 in n2 or n2 in n1:
                canonical = n1 if len(n1) >= len(n2) else n2
                alias = n2 if canonical == n1 else n1
                # 保留带时间戳的版本作为规范名
                if '@' in n1 and '@' not in n2:
                    canonical = n1
                    alias = n2
                elif '@' in n2 and '@' not in n1:
                    canonical = n2
                    alias = n1
                if canonical not in alignments:
                    alignments[canonical] = []
                if alias not in alignments[canonical]:
                    alignments[canonical].append(alias)
                processed.add(id(e1))
                processed.add(id(e2))
                continue

            # 策略2: 编辑距离 < 2 且长度 > 3
            def levenshtein(s1, s2):
                if len(s1) < len(s2):
                    s1, s2 = s2, s1
                if len(s2) == 0:
                    return len(s1)
                prev = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    curr = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = prev[j + 1] + 1
                        deletions = curr[j] + 1
                        substitutions = prev[j] + (c1 != c2)
                        curr.append(min(insertions, deletions, substitutions))
                    prev = curr
                return prev[-1]

            dist = levenshtein(n1, n2)
            if dist <= 2 and len(n1) > 3 and len(n2) > 3:
                canonical = n1 if len(n1) >= len(n2) else n2
                alias = n2 if canonical == n1 else n1
                if canonical not in alignments:
                    alignments[canonical] = []
                if alias not in alignments[canonical]:
                    alignments[canonical].append(alias)
                processed.add(id(e1))
                processed.add(id(e2))

        # 写入 Neo4j：设置等价关系
        if self.is_connected and self.driver and alignments:
            try:
                with self.driver.session() as session:
                    for canonical, aliases in alignments.items():
                        for alias in aliases:
                            # 创建 SAME_AS 等价关系
                            cypher = """
                                MERGE (a:Entity {name: $alias})
                                MERGE (b:Entity {name: $canonical})
                                MERGE (a)-[:SAME_AS]->(b)
                            """
                            session.run(cypher, alias=alias, canonical=canonical)
            except Exception as e:
                safe_print(f"[WARN] 实体对齐写入Neo4j失败: {e}")

        result = {
            'merged_count': sum(len(v) for v in alignments.values()),
            'alignment_groups': [{'canonical': k, 'merged': v} for k, v in alignments.items()]
        }
        safe_print(f"[ENTITY ALIGNMENT] 合并了 {result['merged_count']} 个实体, {len(alignments)} 组")
        return result

    def multi_hop_search(self, query: str, max_hops: int = 2, top_k: int = 3) -> List[Dict]:
        """
        在知识图谱中执行多跳查询。

        Args:
            query: 用户问题（用于提取查询实体）
            max_hops: 最大跳数
            top_k: 返回路径数量

        Returns:
            [{'path', 'nodes', 'relations', 'length', 'source_entity'}, ...]
        """
        paths: List[Dict] = []
        query_entities = self._extract_query_entities(query)

        if not query_entities:
            # 没有提取到实体，直接返回空
            return paths

        # ── 方式1: Neo4j 图查询 ──
        if self.is_connected and self.driver:
            try:
                with self.driver.session() as session:
                    for entity in query_entities[:3]:
                        cypher = f"""
                            MATCH path = (start:Entity {{name: $entity}})-[*1..{max_hops}]-(end:Entity)
                            RETURN path
                            LIMIT {top_k}
                        """
                        results = session.run(cypher, entity=entity)

                        for record in results:
                            path = record['path']
                            nodes = [node['name'] for node in path.nodes]
                            relationships = [rel.type for rel in path.relationships]

                            paths.append({
                                'path': ' -> '.join(nodes),
                                'nodes': nodes,
                                'relations': relationships,
                                'length': len(relationships),
                                'source_entity': entity
                            })
            except Exception as e:
                safe_print(f"[WARN] Neo4j查询失败: {e}")

        # ── 方式2: NetworkX 内存图查询（回退） ──
        if not paths and hasattr(self, 'nx_graph') and self.nx_graph:
            G = self.nx_graph

            for entity in query_entities[:3]:
                if entity not in G:
                    continue
                try:
                    from networkx.algorithms.traversal import bfs_edges

                    edges = list(bfs_edges(G, entity, depth_limit=max_hops))

                    for u, v in edges:
                        if u == entity:
                            nodes_on_path = [entity, v]
                            paths.append({
                                'path': ' -> '.join(nodes_on_path),
                                'nodes': nodes_on_path,
                                'relations': [G[u][v].get('relation', 'RELATED')],
                                'length': 1,
                                'source_entity': entity
                            })
                        if len(paths) >= top_k:
                            break
                except Exception as e:
                    safe_print(f"[WARN] NetworkX查询失败: {e}")

        # ── 去重并返回 ──
        unique_paths = []
        seen = set()
        for p in paths:
            path_key = tuple(p['nodes'])
            if path_key not in seen:
                seen.add(path_key)
                unique_paths.append(p)

        return unique_paths[:top_k]

    def _extract_query_entities(self, query: str) -> List[str]:
        """从查询中提取可能的实体名称"""
        entities = []
        
        known_entities = set(e.name for e in self.local_entities)
        
        for entity_name in known_entities:
            if entity_name in query:
                entities.append(entity_name)
        
        if not entities:
            patterns = [
                r'《([^》]+)》',
                r'[\u4e00-\u9fa5]{2,4}(?:是谁|是什么|在哪里|怎么样)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query)
                if match:
                    extracted = match.group(1) if match.lastindex else match.group(0)[:-3]
                    if len(extracted) >= 2:
                        entities.append(extracted)
        
        return entities

    def _fuse_results(self, graph_results: List[Dict], vector_results: List[Dict]) -> List[Dict]:
        """融合图谱和向量检索结果"""
        fused = []
        
        for idx, g_result in enumerate(graph_results):
            fused.append({
                **g_result,
                'type': 'graph',
                'score': 0.8 - (idx * 0.1),
                'weight': 0.4
            })
        
        for idx, v_result in enumerate(vector_results):
            fused.append({
                **v_result,
                'type': 'vector',
                'weight': 0.6
            })
        
        if hasattr(self, 'communities') and self.communities:
            for item in fused:
                if item.get('type') == 'graph' and item.get('nodes'):
                    for other in fused:
                        if other != item and other.get('type') == 'graph':
                            common_community = self._find_common_community(
                                item.get('nodes', []), 
                                other.get('nodes', [])
                            )
                            if common_community:
                                item['score'] = item.get('score', 0) + 0.1
                                break
        
        fused.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return fused

    def _find_common_community(self, nodes1: List[str], nodes2: List[str]) -> Optional[str]:
        """查找两组节点的共同社团"""
        if not hasattr(self, 'communities') or not self.communities:
            return None
        
        for comm_id, members in self.communities.items():
            if any(n in members for n in nodes1) and any(n in members for n in nodes2):
                return comm_id
        
        return None

    def _get_community_context(self, query: str) -> Dict:
        """获取查询相关的社团上下文"""
        context = {}
        
        query_entities = self._extract_query_entities(query)
        
        for comm_id, members in self.communities.items():
            if any(e in members for e in query_entities):
                context[comm_id] = {
                    'members': members,
                    'size': len(members),
                    'related_to_query': True
                }
        
        return context

    def _generate_query_explanation(self, result: Dict) -> str:
        """生成查询过程的自然语言解释"""
        parts = []
        
        parts.append(f"[查询] '{result['query']}'\n")
        
        if result['graph_results']:
            parts.append(f"[图谱] 找到{len(result['graph_results'])}条路径")
            for i, path in enumerate(result['graph_results'][:2]):
                parts.append(f"   {i+1}. {path.get('path', '')}")
        
        if result['vector_results']:
            parts.append(f"\n[向量] 找到{len(result['vector_results'])}条语义相关片段")
        
        if result['community_context']:
            parts.append(f"\n[社团] 发现{len(result['community_context'])}个相关社群")
        
        return "\n".join(parts)

    def get_statistics(self) -> Dict:
        """获取知识图谱统计信息"""
        stats = {
            'storage_mode': 'Neo4j' if self.is_connected else 'Local Memory',
            'is_connected': self.is_connected,
            'uri': self.uri if self.is_connected else 'N/A',
            'total_entities': len(self.local_entities),
            'total_relations': len(self.local_relations),
            'unique_entities': len(set(e.name for e in self.local_entities)),
            'entity_types': {},
            'relation_types': {},
            'communities': len(getattr(self, 'communities', {})),
            'has_vector_association': self.vector_store is not None
        }
        
        for entity in self.local_entities:
            etype = entity.type
            stats['entity_types'][etype] = stats['entity_types'].get(etype, 0) + 1
        
        for rel in self.local_relations:
            rtype = rel.relation
            stats['relation_types'][rtype] = stats['relation_types'].get(rtype, 0) + 1
        
        if self.is_connected and self.driver:
            try:
                with self.driver.session() as session:
                    node_count = session.run("MATCH (n) RETURN count(n) AS count").single()['count']
                    edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()['count']
                    stats['neo4j_nodes'] = node_count
                    stats['neo4j_edges'] = edge_count
            except:
                pass
        
        return stats

    def export_to_networkx(self):
        """导出为NetworkX图对象"""
        import networkx as nx
        
        if hasattr(self, 'nx_graph') and self.nx_graph:
            return self.nx_graph
        
        G = nx.DiGraph()
        
        for entity in self.local_entities:
            G.add_node(entity.name, 
                      type=entity.type, 
                      vector_id=entity.vector_id,
                      community=getattr(entity, 'community', None))
        
        for rel in self.local_relations:
            if G.has_node(rel.head) and G.has_node(rel.tail):
                G.add_edge(rel.head, rel.tail,
                           relation=rel.relation,
                           confidence=rel.confidence)
        
        self.nx_graph = G
        return G

    def clear(self):
        """清空知识图谱"""
        self.local_entities.clear()
        self.local_relations.clear()
        self.communities = {}
        self.nx_graph = None
        
        if self.is_connected and self.driver:
            try:
                with self.driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
                safe_print("[OK] Neo4j已清空")
            except Exception as e:
                safe_print(f"[WARN] Neo4j清空失败: {e}")
        
        safe_print("[OK] 本地缓存已清空")


def create_neo4j_knowledge_graph(uri="bolt://localhost:7687", 
                                 user="neo4j", 
                                 password="password",
                                 llm_service=None) -> Neo4jKnowledgeGraph:
    """创建Neo4j知识图谱实例并自动连接"""
    kg = Neo4jKnowledgeGraph(uri=uri, user=user, password=password, llm_service=llm_service)
    kg.connect()
    return kg
