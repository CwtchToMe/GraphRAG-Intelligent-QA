# GraphRAG 智能问答系统

基于**大模型+知识图谱**的增强型RAG问答系统，完整实现从文档上传、知识抽取、混合检索到问答生成的全流程。

## 系统特性

- **知识库构建**: 支持PDF/TXT/DOCX/XLSX文档上传和处理
- **智能问答**: 集成大模型API（DeepSeek/OpenAI）+ 本地Ollama双模式
- **知识图谱可视化**: Plotly交互式图表展示
- **Prompt模板展示**: 完整展示系统、NER、RE Prompt
- **测试案例对比**: 3个典型案例（简单查询/多跳推理/边界案例）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 3. 启动系统

```bash
streamlit run app.py
```

访问: http://localhost:8501

### 4. 使用流程

1. **构建知识库** - 上传文档 或 点击"加载示例数据"
2. **配置API（可选）** - 输入API Key，测试连接
3. **开始提问** - 输入问题，查看答案和溯源

### 5. 命令行脚本

```bash
# 图谱构建（数据处理）
python scripts/build_graph.py

# 功能测试
python tests/test_all.py
python tests/test_movie_pipeline.py
python tests/test_visualization.py
```

## 项目结构

```
GraphRAG/
├── app.py                  # Streamlit 主程序（UI层）
├── config.py               # 全局配置
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
│
├── core/                   # 核心业务逻辑层
│   ├── __init__.py
│   ├── document.py         # 文档处理（PDF/TXT/DOCX/XLSX）
│   ├── knowledge_graph.py  # NetworkX 基础知识图谱
│   ├── neo4j_kg.py         # Neo4j 增强知识图谱（LLM提取+社团划分）
│   ├── llm_service.py      # 大模型服务（API + 本地Ollama双模式）
│   ├── retriever.py        # 检索引擎（向量+图谱+多跳推理）
│   ├── vector_store.py     # 向量数据库（ChromaDB + BGE Embedding）
│   └── movie_data.py       # 电影数据处理模块
│
├── scripts/                # 数据处理脚本
│   └── build_graph.py          # 图谱构建脚本
│
├── tests/                  # 测试脚本
│   ├── test_all.py             # 完整功能测试
│   ├── test_visualization.py   # 可视化测试
│   └── test_movie_pipeline.py  # 电影数据管线测试
│
└── data/                   # 数据目录
    ├── raw_data/               # 原始数据（movie_data.json等）
    ├── uploads/                # 上传文件
    ├── graph_db/               # 图谱数据
    └── vector_db/              # 向量数据
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit + Plotly |
| 后端 | LangChain (OpenAI兼容) |
| 图谱 | NetworkX + Neo4j |
| 向量库 | ChromaDB + BGE Embedding |
| 文档处理 | pypdf, python-docx, pandas |
| 大模型 | DeepSeek / OpenAI / Ollama本地模型 |

## 核心功能模块

### Tab 1: 知识库构建
- 文档上传（支持多格式）
- 文本分块和清洗
- 实体和关系提取（LLM驱动）
- 知识图谱构建

### Tab 2: 智能问答
- 大模型API配置和测试
- 知识图谱检索 / 向量检索 / 综合查询
- 混合检索融合
- 多跳推理引擎
- 对话历史管理

### Tab 3: Prompt模板
- 系统Prompt（问答生成）
- NER Prompt（实体识别）
- RE Prompt（关系抽取）

### Tab 4: 知识图谱可视化
- 交互式Plotly图表
- 三元组数据表格
- 图谱统计信息

### Tab 5: 测试案例
- 案例1: 简单查询
- 案例2: 多跳推理
- 案例3: 边界案例
- GraphRAG vs 纯向量RAG对比

## API配置说明

支持多种大模型API：

### DeepSeek（推荐）
- API Base: `https://api.deepseek.com/v1`
- Model: `deepseek-chat`

### OpenAI
- API Base: `https://api.openai.com/v1`
- Model: `gpt-3.5-turbo` / `gpt-4`

### 本地Ollama（免费）
- 安装: https://ollama.com/download
- 模型: `qwen2.5:7b-instruct`
- 系统自动检测，优先使用本地模型

## GraphRAG vs 纯向量RAG

| 能力维度 | 纯向量RAG | GraphRAG |
|---------|-----------|----------|
| 简单查询 | 中 | 优 |
| 多跳推理 | 差 | 优 |
| 避免幻觉 | 差 | 优 |
| 知识溯源 | 困难 | 清晰 |
| 可解释性 | 低 | 高 |

## 许可证

MIT License
