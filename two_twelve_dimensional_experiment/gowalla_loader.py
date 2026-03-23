import networkx as nx
import random

def load_gowalla_graph(path):

    print("Loading Gowalla...")

    G = nx.read_edgelist(
        path,
        nodetype=int
    )

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    return G

def extract_lcc(G):

    lcc = max(nx.connected_components(G), key=len)

    G_lcc = G.subgraph(lcc).copy()

    print("LCC Nodes:", G_lcc.number_of_nodes())
    print("LCC Edges:", G_lcc.number_of_edges())

    return G_lcc


def sample_connected_subgraph(G, target_size=4096):

    start_node = random.choice(list(G.nodes()))

    visited = set()
    queue = [start_node]

    while queue and len(visited) < target_size:

        node = queue.pop(0)

        if node not in visited:
            visited.add(node)

            neighbors = list(G.neighbors(node))
            random.shuffle(neighbors)

            queue.extend(neighbors)

    subgraph = G.subgraph(visited).copy()

    print("Sampled connected nodes:", subgraph.number_of_nodes())
    print("Sampled edges:", subgraph.number_of_edges())

    return subgraph
