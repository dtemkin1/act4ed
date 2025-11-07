from matplotlib import pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch


def plot_problem_graph(G, pos, routes, od_flows, node_size=300, edge_width=1):
    """
    Plot a graph with routes and OD flows.

    Args:
        G (networkx.Graph): The graph to be plotted.
        pos (dict): Node positions for plotting.
        routes (list): List of routes (list of node sequences).
        od_flows (dict): Dictionary with OD pairs as keys and flow values as values.
        node_size (int, optional): Size of the graph nodes. Default is 300.
        edge_width (int, optional): Width of the graph edges. Default is 1.
    """
    fig, axes = plt.subplots(len(routes), 1, figsize=(8, len(routes) * 4))

    if len(routes) == 1:
        axes = [axes]

    for ax, route in zip(axes, routes):
        # Draw the graph with bold node labels
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=node_size, font_weight='bold', font_size=18, ax=ax)

        # Highlight the route with a visually enhanced red
        edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='#d62728',
                               width=edge_width, ax=ax)

        for (start, end), flow in od_flows.items():
            # Calculate the position for text at the first quarter of the arrow
            text_pos = (0.25 * pos[start][0] + 0.75 * pos[end][0]-0.2,
                        0.25 * pos[start][1] + 0.75 * pos[end][1])

            # Draw arrow from start to end
            ax.annotate('',
                        xy=pos[end],
                        xytext=pos[start],
                        arrowprops=dict(facecolor='teal', shrink=0.05, alpha=0.5))

            # Add a white rounded box background for text
            bbox = FancyBboxPatch(
                (text_pos[0] - 0.13, text_pos[1] - 0.02),  # Lower-left corner
                0.27, 0.08,  # Width and height of the box
                boxstyle="round,pad=0.05",
                facecolor="white",
                edgecolor="black",
                lw=1
            )
            ax.add_patch(bbox)

            # Add annotation text on top of the white box
            ax.text(text_pos[0], text_pos[1],
                    f'Flow: {flow}',
                    fontsize=14,
                    fontweight='bold',
                    color='black',
                    ha='center',
                    va='center')

    plt.tight_layout()
    return fig, axes
