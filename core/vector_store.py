"""
向量数据库模块
负责：文档向量化、向量存储、相似度检索

技术栈：
- LangChain TextSplitter: 智能文本切分
- Sentence-Transformers: 预训练中文嵌入模型
- ChromaDB: 向量数据库存储和检索
"""
import os
import re
import tempfile
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    """文档块数据结构"""
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: Dict = None


class VectorStore:
    """
    向量数据库类
    
    功能：
    - 文档加载与智能切分
    - 文本向量化（使用预训练Embedding模型）
    
    - 向量存储（ChromaDB）
    - 相似度检索
    """
    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-small-zh-v1.5",
                 persist_directory: str = "./chroma_db",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        初始化向量数据库
        
        Args:
            embedding_model: 嵌入模型名称（支持HuggingFace模型）
            persist_directory: ChromaDB持久化目录
            chunk_size: 文本块大小（字符数）
            chunk_overlap: 重叠大小（字符数）
        """
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.chroma_collection = None
        self.embeddings = None
        self.text_splitter = None
        self.documents = []
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        初始化向量数据库组件
        
        Returns:
            success: 是否成功初始化
        """
        try:
            # 1. 初始化文本切分器（LangChain）
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", ";", "；", "，", " ", ""]
            )
            
            # 2. 初始化ChromaDB
            import chromadb
            
            # 创建持久化客户端 (保存为实例变量，避免丢失引用)
            os.makedirs(self.persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            
            # 使用ChromaDB内置的嵌入函数（避免复杂依赖）
            # 方案1: 优先尝试使用sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np

                print(f"[INFO] 正在加载嵌入模型: {self.embedding_model_name}...")

                # 加载预训练模型
                self.model = SentenceTransformer(self.embedding_model_name)

                # 创建自定义嵌入函数
                class CustomEmbeddingFunction(chromadb.EmbeddingFunction):
                    def __init__(self, model):
                        self.model = model

                    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
                        embeddings = self.model.encode(input, normalize_embeddings=True)
                        return embeddings.tolist()

                self.embeddings_fn = CustomEmbeddingFunction(self.model)
                self.embedding_type = "sentence_transformers"
                print("[OK] 使用Sentence-Transformers嵌入模型")

            except Exception as e:
                print(f"[WARN] Sentence-Transformers加载失败: {e}")
                print("[INFO] 回退到ChromaDB默认嵌入函数...")

                # 方案2: 回退到ChromaDB默认函数
                self.embeddings_fn = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
                self.embedding_type = "default"
                print("[OK] 使用ChromaDB默认嵌入函数")

            # 获取或创建集合
            # 注意：如果集合已存在且嵌入函数不同，需要删除重建
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name="graphrag_documents"
                )
                # 集合已存在，检查是否需要删除重建
                print("[WARN] 检测到已有向量库，正在重建以应用新嵌入函数...")
                self.chroma_client.delete_collection(name="graphrag_documents")
            except Exception:
                pass  # 集合不存在，正常创建

            self.chroma_collection = self.chroma_client.create_collection(
                name="graphrag_documents",
                embedding_function=self.embeddings_fn,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )

            self.is_initialized = True
            return True

        except Exception as e:
            print(f"[ERROR] 向量数据库初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_document(self, file_path: str = None, uploaded_file=None) -> List[Dict]:
        """
        加载并处理单个文档

        Args:
            file_path: 文件路径（可选）
            uploaded_file: Streamlit上传的文件对象（可选）

        Returns:
            chunks: 切分后的文档块列表
        """
        if not self.is_initialized:
            if not self.initialize():
                raise Exception("向量数据库未初始化")

        # 读取文件内容
        if uploaded_file:
            content = uploaded_file.read()
            file_name = uploaded_file.name
            file_type = uploaded_file.type
        elif file_path:
            with open(file_path, 'rb') as f:
                content = f.read()
            file_name = os.path.basename(file_path)
            file_type = self._guess_type(file_path)
        else:
            raise ValueError("必须提供file_path或uploaded_file")

        # 根据文件类型选择处理策略
        if "sheet" in str(file_type) or ".xlsx" in file_name.lower() or ".xls" in file_name.lower():
            # 结构化数据处理：每行生成一个独立的向量
            chunks = self._process_structured_data(content, file_name)
        else:
            # 非结构化文本处理：使用LangChain智能切分
            chunks = self._process_unstructured_text(content, file_type, file_name)

        print(f"[OK] 文档 {file_name} 处理完成: 共{len(chunks)}个文本块")
        return chunks

    def _process_structured_data(self, content: bytes, file_name: str) -> List[Dict]:
        """
        处理结构化数据（Excel/CSV）

        核心改进：每行数据生成一个独立的向量，保持事件完整性
        例如：12345投诉数据，每行一个投诉事件 → 一个向量
        """
        import pandas as pd
        import io

        try:
            df = pd.read_excel(io.BytesIO(content))

            if df.empty:
                print(f"警告: Excel文件 {file_name} 为空")
                return []

            chunks = []
            for idx, row in df.iterrows():
                # 将每行数据转换为自然语言描述
                row_text = self._format_row_to_text(row, df.columns)

                if len(row_text.strip()) < 10:
                    continue

                chunk = {
                    'id': f"{file_name}_row_{idx}",
                    'text': row_text,
                    'source': file_name,
                    'chunk_index': idx,
                    'metadata': {
                        'type': 'structured_row',
                        'row_index': idx,
                        'source_file': file_name,
                        **{col: str(row[col]) for col in df.columns if pd.notna(row[col])}
                    }
                }
                chunks.append(chunk)

            print(f"   [DATA] Excel处理: {len(df)}行数据 -> {len(chunks)}个独立向量")
            return chunks

        except Exception as e:
            print(f"[ERROR] Excel文件 {file_name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _format_row_to_text(self, row, columns) -> str:
        """将表格行转换为自然语言描述，每行作为一条完整的投诉工单"""
        import pandas as pd

        # 工单编号（唯一标识，核心！）
        gdbh = None
        for col in ['gdbh', 'id', '编号']:
            if col in columns and pd.notna(row.get(col)):
                gdbh = str(row.get(col)).strip()
                break

        # 收集各字段
        person = None
        for col in ['ldr', 'sfd', 'tsr', 'btsxr']:
            if col in columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                person = str(row.get(col)).strip()
                break

        time_val = None
        for col in ['lrsj', 'fssj', 'clsj']:
            if col in columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                time_val = str(row.get(col)).strip()
                break

        location = None
        for col in ['location', 'dd', 'bz']:
            if col in columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                location = str(row.get(col)).strip()
                break

        dept = None
        for col in ['blbm']:
            if col in columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                dept = str(row.get(col)).strip()
                break

        category = None
        for col in ['yjfl', 'ejfl', 'sjfl']:
            if col in columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                category = str(row.get(col)).strip()
                break

        title = None
        for col in ['gdbt', 'bt']:
            if col in columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                title = str(row.get(col)).strip()
                break

        content = None
        for col in ['zynr', 'nr', 'content']:
            if col in columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                content = str(row.get(col)).strip()
                break

        event_type = None
        for col in ['sjlx']:
            if col in columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                event_type = str(row.get(col)).strip()
                break

        # 组装自然语言：以工单编号开头，叙事顺序拼接
        parts = []

        # 开头：工单编号 + 时间
        if gdbh:
            parts.append(f"工单{gdbh}")
        if time_val:
            time_str = time_val[:19] if len(time_val) > 19 else time_val
            parts.append(f"：{time_str}")
        else:
            parts.append("：")

        # 人物
        if person:
            parts.append(f"{person}反映")
        else:
            parts.append("有人反映")

        # 地点（如果有）
        if location:
            parts.append(f"在{location}")

        # 主要内容（最关键）
        if content:
            content = content.strip()
            if not content.endswith(('。', '！', '？', '.')):
                content += '。'
            parts.append(content)
        elif title:
            parts.append(f"反映问题：{title}。")

        # 处理部门
        if dept:
            parts.append(f"该投诉由{dept}负责处理。")

        combined = "".join(parts)
        combined = re.sub(r'[，。]+', lambda m: m.group()[-1] if m.group() else '', combined)
        combined = re.sub(r'\.{2,}', '。', combined)
        combined = re.sub(r'。{2,}', '。', combined)

        return combined if combined.strip() else None

    def _process_unstructured_text(self, content: bytes, file_type: str, file_name: str) -> List[Dict]:
        """处理非结构化文本（PDF/TXT/DOCX）"""
        text = self._extract_text(content, file_type, file_name)

        if not text or len(text.strip()) < 50:
            print(f"警告: 文件 {file_name} 内容过短或为空")
            return []

        # 使用LangChain TextSplitter进行智能切分
        chunks_data = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[{"source": file_name}]
        )

        # 转换为统一格式
        chunks = []
        for idx, chunk in enumerate(chunks_data):
            chunks.append({
                'id': f"{file_name}_chunk_{idx}",
                'text': chunk.page_content,
                'source': file_name,
                'chunk_index': idx,
                'metadata': chunk.metadata
            })

        print(f"   [TEXT] 文本切分: 1个文档 -> {len(chunks)}个文本块 (LangChain)")
        return chunks
    
    def _extract_text(self, content: bytes, file_type: str, file_name: str) -> str:
        """根据文件类型提取文本内容"""
        
        try:
            if "text/plain" in file_type:
                text = content.decode('utf-8')
                
            elif "pdf" in file_type:
                import pypdf
                pdf_file = tempfile.SpooledTemporaryFile()
                pdf_file.write(content)
                pdf_file.seek(0)
                pdf_reader = pypdf.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    
            elif "sheet" in file_type or "xlsx" in file_name.lower():
                import pandas as pd
                import io
                df = pd.read_excel(io.BytesIO(content))
                text = df.to_string(index=False)
                
            elif "word" in file_type or "docx" in file_name.lower():
                import docx
                docx_file = tempfile.SpooledTemporaryFile()
                docx_file.write(content)
                docx_file.seek(0)
                doc = docx.Document(docx_file)
                text = "\n".join([para.text for para in doc.paragraphs])
                
            else:
                # 尝试UTF-8解码
                text = content.decode('utf-8')
                
        except Exception as e:
            print(f"文件 {file_name} 提取文本失败: {e}")
            text = f"[无法解析的内容] {file_name}"
        
        return text
    
    @staticmethod
    def _guess_type(file_path: str) -> str:
        """根据文件扩展名猜测MIME类型"""
        ext = os.path.splitext(file_path)[1].lower()
        type_map = {
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        return type_map.get(ext, 'application/octet-stream')
    
    def add_documents(self, chunks: List[Dict]) -> int:
        """
        将文档块添加到向量数据库
        
        流程：文本 → Embedding模型 → 向量 → 存入ChromaDB
        
        Args:
            chunks: 文档块列表
        
        Returns:
            count: 成功添加的数量
        """
        if not self.is_initialized:
            raise Exception("向量数据库未初始化")
        
        if not chunks:
            return 0
        
        try:
            # 准备数据
            ids = [chunk['id'] for chunk in chunks]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = [{
                'source': chunk.get('source', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                **chunk.get('metadata', {})
            } for chunk in chunks]
            
            # 添加到ChromaDB（自动进行向量化）
            self.chroma_collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            # 同时保存到本地文档列表
            self.documents.extend(chunks)
            
            print(f"[OK] 成功添加{len(chunks)}个文档块到向量数据库")
            return len(chunks)

        except Exception as e:
            print(f"[ERROR] 添加文档到向量数据库失败: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        向量相似度检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            results: 检索结果列表
        """
        if not self.is_initialized:
            return []
        
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # 格式化结果
            formatted_results = []
            if results and results.get('documents'):
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0,
                        'id': results['ids'][0][i],
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"[ERROR] 向量检索失败: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """获取向量数据库统计信息"""
        if not self.is_initialized:
            return {'total_documents': 0}
        
        try:
            count = self.chroma_collection.count()
            return {
                'total_documents': count,
                'embedding_type': self.embedding_type,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            print(f"[ERROR] 获取统计信息失败: {e}")
            return {'total_documents': 0}
    
    def clear(self):
        """清空向量数据库"""
        try:
            self.chroma_client.delete_collection(name="graphrag_documents")
            self.chroma_collection = self.chroma_client.create_collection(
                name="graphrag_documents",
                embedding_function=self.embeddings_fn,
                metadata={"hnsw:space": "cosine"}
            )
            self.documents = []
            print("[OK] 向量数据库已清空")
        except Exception as e:
            print(f"[ERROR] 清空向量数据库失败: {e}")


def build_knowledge_base(uploaded_files, chunk_size=500, chunk_overlap=50,
                         embedding_model="BAAI/bge-small-zh-v1.5",
                         persist_directory="./chroma_db") -> Tuple[VectorStore, int]:
    """
    批量构建知识库
    
    Args:
        uploaded_files: Streamlit上传的文件列表
        chunk_size: 文本块大小
        chunk_overlap: 重叠大小
        embedding_model: 嵌入模型
        persist_directory: 持久化目录
    
    Returns:
        vs: VectorStore实例
        total_chunks: 总文本块数
    """
    vs = VectorStore(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        persist_directory=persist_directory
    )
    
    if not vs.initialize():
        raise Exception("向量数据库初始化失败")
    
    total_chunks = 0
    for file in uploaded_files:
        chunks = vs.load_document(uploaded_file=file)
        count = vs.add_documents(chunks)
        total_chunks += count
    
    return vs, total_chunks


def create_vector_store(chunk_size=500, chunk_overlap=50,
                         embedding_model="BAAI/bge-small-zh-v1.5",
                         persist_directory="./chroma_db") -> VectorStore:
    """工厂函数：创建并初始化 VectorStore"""
    vs = VectorStore(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        persist_directory=persist_directory
    )
    vs.initialize()
    return vs


def process_and_vectorize(uploaded_file, chunk_size=500, chunk_overlap=50,
                          embedding_model="BAAI/bge-small-zh-v1.5",
                          persist_directory="./chroma_db") -> Tuple[VectorStore, int]:
    """
    一站式处理：初始化 + 加载文档 + 向量化存储

    Args:
        uploaded_file: Streamlit 上传文件对象
        chunk_size: 文本块大小
        chunk_overlap: 重叠大小
        embedding_model: 嵌入模型
        persist_directory: 持久化目录

    Returns:
        (vs, chunk_count): VectorStore实例和处理的块数
    """
    vs = create_vector_store(chunk_size, chunk_overlap, embedding_model, persist_directory)
    chunks = vs.load_document(uploaded_file=uploaded_file)
    count = vs.add_documents(chunks)
    return vs, count
