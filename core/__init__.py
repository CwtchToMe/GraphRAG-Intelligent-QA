"""
Core模块初始化
"""
from .document import process_document, extract_knowledge_from_text, batch_process_documents
from .knowledge_graph import KnowledgeGraph, create_sample_graph
from .llm_service import LLMService
from .retriever import Retriever, MultiHopReasoner
from .vector_store import VectorStore, create_vector_store, process_and_vectorize
from .neo4j_kg import Neo4jKnowledgeGraph, Entity, Relation, create_neo4j_knowledge_graph
from .movie_data import (
    load_movie_data,
    movie_to_chunks,
    process_movie_data,
    get_available_datasets,
    MOVIE_NER_RE_PROMPT,
)

__all__ = [
    'process_document',
    'extract_knowledge_from_text',
    'batch_process_documents',
    'KnowledgeGraph',
    'create_sample_graph',
    'LLMService',
    'Retriever',
    'VectorStore',
    'create_vector_store',
    'process_and_vectorize',
    'Neo4jKnowledgeGraph',
    'Entity',
    'Relation',
    'create_neo4j_knowledge_graph',
    'load_movie_data',
    'movie_to_chunks',
    'process_movie_data',
    'get_available_datasets',
    'MOVIE_NER_RE_PROMPT',
]
