import networkx as nx

# Create a 2D grid graph with integer node indices
def create_street_grid(m, n):
    g = nx.Graph()
    node_mapping = {}  # Map (row, col) to a unique integer
    counter = 0

    for i in range(m):
        for j in range(n):
            node_mapping[(i, j)] = counter
            counter += 1

    for i in range(m):
        for j in range(n):
            if i > 0:
                g.add_edge(node_mapping[(i, j)], node_mapping[(i - 1, j)], weight=1)
            if j > 0:
                g.add_edge(node_mapping[(i, j)], node_mapping[(i, j - 1)], weight=1)

    # Add length attribute to edges
    for u, v in g.edges():
        g.edges[u, v]['length'] = 1

    # Generate positions for nodes (2D grid layout)
    pos = {node_mapping[(x, y)]: (y, -x) for x, y in node_mapping.keys()}

    return g, pos
