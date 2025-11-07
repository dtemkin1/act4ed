from typing import Union
import networkx as nx


def get_edge_weight(graph: nx.MultiDiGraph, u: int, v: int, edge_type: str = 'distance') -> Union[float, None]:
    # Iterate through all edges between u and v
    for key, edge_data in graph.get_edge_data(u, v, default={}).items():
        if edge_data.get('type') == edge_type:
            return edge_data.get('weight', None)
    # Return None if no matching edge is found
    return None
