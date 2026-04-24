"""
文档处理模块
负责：文档读取、文本分块、知识提取
"""
import re
import tempfile
from typing import List, Dict, Tuple


def process_document(uploaded_file, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """
    处理单个文档文件
    
    Args:
        uploaded_file: Streamlit上传的文件对象
        chunk_size: 文本块大小
        chunk_overlap: 重叠大小
    
    Returns:
        chunks: 文本块列表 [{'id', 'text', 'source', 'chunk_index'}]
    """
    content = uploaded_file.read()
    
    # 根据文件类型处理
    if uploaded_file.type == "text/plain":
        text = content.decode('utf-8')
    elif uploaded_file.type == "application/pdf":
        try:
            import pypdf
            pdf_reader = pypdf.PdfReader(tempfile.SpooledTemporaryFile())
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except:
            text = f"[PDF内容] {uploaded_file.name}"
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        try:
            import pandas as pd
            import io
            df = pd.read_excel(io.BytesIO(content))
            text = df.to_string(index=False)
            text = f"[XLSX内容] {uploaded_file.name}\n\n{text}"
        except Exception as e:
            text = f"[XLSX内容] {uploaded_file.name} (读取失败)"
    else:
        try:
            import docx
            doc = docx.Document(tempfile.SpooledTemporaryFile())
            text = "\n".join([para.text for para in doc.paragraphs])
        except:
            text = f"[DOCX内容] {uploaded_file.name}"
    
    # 文本分块
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 20:
            chunks.append({
                'id': f"{uploaded_file.name}_chunk_{i}",
                'text': chunk,
                'source': uploaded_file.name,
                'chunk_index': i
            })
    
    return chunks


def extract_knowledge_from_text(text: str) -> Tuple[List[str], List[Dict]]:
    """
    从文本中提取实体和关系。
    注意：此函数已废弃，请使用 core.neo4j_kg 的 LLM 提取功能。
    此处返回空列表，避免误导性的正则匹配结果。
    """
    return [], []


def batch_process_documents(uploaded_files, chunk_size=500, chunk_overlap=50) -> Dict:
    """
    批量处理多个文档
    
    Returns:
        {
            'chunks': 所有文本块,
            'triples': 所有三元组,
            'stats': 处理统计
        }
    """
    all_chunks = []
    all_triples = []
    
    for file in uploaded_files:
        chunks = process_document(file, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        
        full_text = " ".join([c['text'] for c in chunks])
        _, triples = extract_knowledge_from_text(full_text)
        all_triples.extend(triples)
    
    return {
        'chunks': all_chunks,
        'triples': all_triples,
        'stats': {
            'total_files': len(uploaded_files),
            'total_chunks': len(all_chunks),
            'total_triples': len(all_triples)
        }
    }
