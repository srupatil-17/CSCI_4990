import networkx as nx

# -----------------------------
# DISTANCE FUNCTIONS
# -----------------------------

def manhattan_distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def graph_distance_heuristic(G, target, cache):

    if target not in cache:
        cache[target] = nx.single_source_shortest_path_length(G, target)

    return cache[target]


# -----------------------------
# GREEDY ROUTING
# -----------------------------

def greedy_route(
    G,
    start,
    target,
    distance_func,
    max_steps=1000,
    cache=None
):
    """
    Generic greedy routing function.

    distance_func must be:
        f(current_node, neighbor, target, cache) -> distance
    """

    if cache is None:
        cache = {}

    current = start
    path = [current]
    visited = set([current])

    steps = 0

    while current != target and steps < max_steps:

        neighbors = list(G.neighbors(current))

        if not neighbors:
            return path, False

        best_neighbor = None
        best_distance = float("inf")

        for n in neighbors:

            d = distance_func(current, n, target, cache)

            if d < best_distance:
                best_distance = d
                best_neighbor = n

        if best_neighbor is None:
            return path, False

        if best_neighbor in visited:
            return path, False

        current = best_neighbor
        path.append(current)
        visited.add(current)

        steps += 1

    return path, current == target


# -----------------------------
# DISTANCE ADAPTERS
# -----------------------------

def lattice_distance(_, neighbor, target, cache):
    return manhattan_distance(neighbor, target)


def graph_distance(_, neighbor, target, cache):
    dist_map = graph_distance_heuristic(cache["G"], target, cache)

    return dist_map.get(neighbor, float("inf"))


# -----------------------------
# EXPERIMENT RUNNER
# -----------------------------

def verify_greedy_property(G, path, target):
    for i in range(len(path) - 1):
        current = path[i]
        chosen = path[i+1]

        neighbors = list(G.neighbors(current))

        best = min(neighbors, key=lambda n: manhattan_distance(n, target))

        if chosen != best:
            return False

    return True


def run_greedy_trials(
    G,
    trials,
    distance_type="kleinberg"
):

    import random

    nodes = list(G.nodes())

    path_lengths = []
    successes = 0

    cache = {}

    if distance_type == "graph":
        cache["G"] = G

    for _ in range(trials):

        start = random.choice(nodes)
        target = random.choice(nodes)

        if start == target:
            continue

        if distance_type == "kleinberg":
            dist_func = lattice_distance
        else:
            dist_func = graph_distance

        path, success = greedy_route(
            G,
            start,
            target,
            dist_func,
            cache=cache
        )

        if success:
            path_lengths.append(len(path) - 1)
            successes += 1

    success_rate = successes / trials if trials > 0 else 0

    return path_lengths, success_rate

