"""
知识图谱模块
负责：图谱构建、存储、查询、可视化
"""
import networkx as nx
from typing import List, Dict, Optional


class KnowledgeGraph:
    """
    知识图谱管理类
    
    功能：
    - 图谱构建和管理
    - 实体和关系查询
    - 多跳查询
    - 图谱统计
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.triples = []
    
    def add_triples(self, triples: List[Dict]):
        """
        添加三元组到图谱
        
        Args:
            triples: 三元组列表 [{'head', 'relation', 'tail', ...}]
        """
        for triple in triples:
            self.graph.add_node(triple['head'], type='Entity')
            self.graph.add_node(triple['tail'], type='Entity')
            self.graph.add_edge(
                triple['head'],
                triple['tail'],
                relation=triple.get('relation', 'RELATED_TO'),
                confidence=triple.get('confidence', 0.7)
            )
            self.triples.append(triple)
    
    def clear(self):
        """清空图谱"""
        self.graph.clear()
        self.triples = []
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        在知识图谱中搜索相关实体和关系

        Args:
            query: 查询文本
            top_k: 返回结果数量，None 表示返回全部

        Returns:
            results: 检索结果列表
        """
        import re
        entities = re.findall(r'[\u4e00-\u9fa5]{2,10}|《[^》]+》', query)
        entities = list(set(entities))
        
        results = []
        for entity in entities:
            if entity in self.graph:
                # 查找出边（实体作为头）
                for neighbor in self.graph.successors(entity):
                    edge_data = self.graph.get_edge_data(entity, neighbor)
                    results.append({
                        'type': 'direct_relation',
                        'head': entity,
                        'relation': edge_data.get('relation', 'RELATED_TO'),
                        'tail': neighbor,
                        'score': 0.9
                    })
                
                # 查找入边（实体作为尾）
                for predecessor in self.graph.predecessors(entity):
                    edge_data = self.graph.get_edge_data(predecessor, entity)
                    results.append({
                        'type': 'direct_relation',
                        'head': predecessor,
                        'relation': edge_data.get('relation', 'RELATED_TO'),
                        'tail': entity,
                        'score': 0.9
                    })
        
        return results[:top_k]
    
    def multi_hop_query(self, start_entity: str, max_hops: int = 2) -> List[Dict]:
        """
        多跳查询
        
        Args:
            start_entity: 起始实体
            max_hops: 最大跳数
        
        Returns:
            paths: 路径列表
        """
        if start_entity not in self.graph:
            return []
        
        paths = []
        
        for target in self.graph.nodes():
            if target == start_entity:
                continue
            
            try:
                path = nx.shortest_path(
                    self.graph, 
                    source=start_entity, 
                    target=target
                )
                
                if path and len(path) <= max_hops + 1:
                    paths.append({
                        'path': path,
                        'hops': len(path) - 1,
                        'target': target
                    })
            except:
                continue
        
        return paths[:10]
    
    def get_subgraph(self, entities: List[str], max_hops: int = 2) -> nx.DiGraph:
        """
        获取子图
        
        Args:
            entities: 实体列表
            max_hops: 最大跳数
        
        Returns:
            subgraph: 子图对象
        """
        subgraph = nx.DiGraph()
        
        for entity in entities:
            if entity in self.graph:
                # 添加该节点及其邻居
                subgraph.add_node(entity, **self.graph.nodes[entity])
                
                for neighbor in list(self.graph.successors(entity))[:max_hops]:
                    subgraph.add_node(neighbor, **self.graph.nodes[neighbor])
                    edge_data = self.graph.get_edge_data(entity, neighbor)
                    subgraph.add_edge(entity, neighbor, **edge_data)
                
                for predecessor in list(self.graph.predecessors(entity))[:max_hops]:
                    subgraph.add_node(predecessor, **self.graph.nodes[predecessor])
                    edge_data = self.graph.get_edge_data(predecessor, entity)
                    subgraph.add_edge(predecessor, entity, **edge_data)
        
        return subgraph
    
    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'triples': len(self.triples),
            'density': f"{nx.density(self.graph):.3f}"
        }
    
    def export_to_json(self) -> str:
        """导出为JSON格式"""
        import json
        return json.dumps(self.triples, ensure_ascii=False, indent=2)
    
    def export_to_csv(self) -> str:
        """导出为CSV格式"""
        import pandas as pd
        df = pd.DataFrame(self.triples)
        return df.to_csv(index=False)
    
    def export_to_gexf(self, filepath: str):
        """导出为GEXF格式（可用Gephi打开）"""
        nx.write_gexf(self.graph, filepath)


def create_sample_graph() -> KnowledgeGraph:
    """创建示例知识图谱"""
    kg = KnowledgeGraph()
    
    sample_triples = [
        {"head": "大话西游", "relation": "DIRECTED_BY", "tail": "刘镇伟"},
        {"head": "大话西游", "relation": "STARRING", "tail": "周星驰"},
        {"head": "大话西游", "relation": "STARRING", "tail": "朱茵"},
        {"head": "大话西游", "relation": "RELEASED_IN", "tail": "1995年"},
        {"head": "唐伯虎点秋香", "relation": "DIRECTED_BY", "tail": "李力持"},
        {"head": "唐伯虎点秋香", "relation": "STARRING", "tail": "周星驰"},
        {"head": "功夫", "relation": "DIRECTED_BY", "tail": "周星驰"},
        {"head": "功夫", "relation": "STARRING", "tail": "周星驰"},
        {"head": "少林足球", "relation": "DIRECTED_BY", "tail": "周星驰"},
        {"head": "喜剧之王", "relation": "DIRECTED_BY", "tail": "李力持"},
        {"head": "喜剧之王", "relation": "STARRING", "tail": "周星驰"},
        {"head": "喜剧之王", "relation": "STARRING", "tail": "张柏芝"},
        {"head": "刘镇伟", "relation": "DIRECTED", "tail": "大话西游"},
        {"head": "周星驰", "relation": "ACTED_IN", "tail": "大话西游"},
        {"head": "周星驰", "relation": "ACTED_IN", "tail": "功夫"},
        {"head": "周星驰", "relation": "DIRECTED", "tail": "功夫"},
    ]
    
    kg.add_triples(sample_triples)
    return kg
