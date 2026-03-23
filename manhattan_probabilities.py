import networkx as nx

n = 5
r_exp = 2

def manhattan(u, v):
    return abs(u[0] - v[0]) + abs(u[1] - v[1])

def create_nodes(n):
    return [(i, j) for i in range(n) for j in range(n)]

def build_lattice_graph(n):
    G = nx.Graph()
    nodes = create_nodes(n)
    G.add_nodes_from(nodes)

    for (x, y) in nodes:
        for dx, dy, in [(-1,0), (1,0), (0,-1), (0, 1)]:
            nx_, ny_ = x + dx, y + dy
            if (nx_, ny_) in nodes:
                G.add_edge((x, y), (nx_, ny_))
    
    return G

def compute_probabilities(u, nodes, r_exp):
    weights = {}

    for v in nodes:
        if v == u:
            continue
        d = manhattan(u, v)
        weights[v] = 1 / (d ** r_exp)

    total = sum(weights.values())
    probs = {v: w / total for v, w in weights.items()}
        
    return probs

G = build_lattice_graph(n)
nodes = list(G.nodes())

print("Nodes:", len(nodes))
print("Edges:", G.number_of_edges())
print()
u = (2, 2)
probs = compute_probabilities(u, nodes, r_exp)

print(f"Manhattan-based probabilities from node {u}:\n")

for v, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
    print(f"{v}  d={manhattan(u, v)}  p={p:.5f}")

print("\nSum of probabilities:", sum(probs.values()))