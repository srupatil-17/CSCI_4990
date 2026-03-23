import random
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os

from task_one import generate_graph, manhattan

# before running dont forget about Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# and then .\venv\Scripts\Activate.ps1

# PARAMETERS
n = 10
r = 2
seed = 42

# MEETING NOTES:
# change start and target into inputs

# GREEDY ROUTING FUNCTION

def greedy_routing(G, start, target):

    current = start
    path = [current]
    visited = set([current])

    while current != target:

        neighbors = list(G.neighbors(current))

        best_neighbor = None
        best_distance = manhattan(current, target)

        for neighbor in neighbors:

            d = manhattan(neighbor, target)

            if d < best_distance:
                best_distance = d
                best_neighbor = neighbor

        if best_neighbor is None:
            return path, False

        if best_neighbor in visited:
            return path, False

        current = best_neighbor
        path.append(current)
        visited.add(current)

    return path, True



# VISUALIZE ROUTING ONLY

def visualize_routing(G, shortcuts, path):

    pos = {node: node for node in G.nodes()}

    shortcut_edges = set(shortcuts)

    local_edges = []
    for u, v in G.edges():
        if (u, v) not in shortcut_edges and manhattan(u, v) == 1:
            local_edges.append((u, v))

    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]

    plt.figure(figsize=(8, 8))

    nx.draw_networkx_nodes(
        G, pos,
        node_size=20,
        node_color="black"
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=local_edges,
        edge_color="lightgray",
        width=0.5
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=shortcut_edges,
        edge_color="red",
        width=0.5,
        arrows=True
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=path_edges,
        edge_color="blue",
        width=3,
        arrows=True
    )

    plt.title("Greedy Routing Path")
    plt.axis("off")
    plt.show()


def get_node_input(prompt, n):
    while True:
        try:
            text = input(prompt + " (format: x,y): ")
            x_str, y_str = text.split(",")
            x, y = int(x_str), int(y_str)

            if 0 <= x < n and 0 <= y < n:
                return (x, y)
            else:
                print(f"Coordinates must be between 0 and {n-1}")

        except:
            print("Invalid format. Example: 0,4")



# PRINT STATS

def print_stats(start, target, path, success):

    print("\nGREEDY ROUTING RESULTS")
    print("----------------------")

    print("Start:", start)
    print("Target:", target)

    print("Path length:", len(path) - 1)

    print("Success:", success)

    print("\nPath:")
    for node in path:
        print(node)


def run_master_experiments(G, shortcuts, num_experiments, experiment_id, n, use_random=True):

    nodes = list(G.nodes())

    results = []

    for i in range(num_experiments):

        if use_random:
            start = random.choice(nodes)
            target = random.choice(nodes)

            while target == start:
                target = random.choice(nodes)

        else:
            start = get_node_input("Enter START node", n)
            target = get_node_input("Enter TARGET node", n)

            while target == start:
                print("Target cannot equal start.")
                target = get_node_input("Enter TARGET node", n)

        path, success = greedy_routing(G, start, target)

        results.append({
            "experiment_id": experiment_id,
            "start": start,
            "target": target,
            "path": path,
            "hops": len(path) - 1 if success else None,
            "success": success
        })

    return results


# Master experiment exporter
def export_experiments(results, filename="experiments.csv"):

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "experiment_id",
                "start",
                "target",
                "path",
                "hops",
                "success"
            ])

        for r in results:
            writer.writerow([
                r["experiment_id"],
                r["start"],
                r["target"],
                r["path"],
                r["hops"],
                r["success"]
            ])


# Master experiment summary exporter
def export_summary(results):

    total_experiments = len(results)
    successes = sum(1 for r in results if r["success"])
    hops_list = [r["hops"] for r in results if r["success"]]

    success_rate = successes / total_experiments if total_experiments else 0
    mean_hops = sum(hops_list) / len(hops_list) if hops_list else 0

    with open("master_summary.csv", "w", newline="") as f:

        import csv
        writer = csv.writer(f)

        writer.writerow([
            "total_experiments",
            "successes",
            "success_rate",
            "mean_hops"
        ])

        writer.writerow([
            total_experiments,
            successes,
            success_rate,
            mean_hops
        ])



# MAIN

if __name__ == "__main__":


    experiment_id = 1
    num_experiments = 10

    # Generate graph
    G, shortcuts = generate_graph(n, r, seed)

    nodes = list(G.nodes())

    mode = input("Choose mode:\n1 = Single Test\n2 = Batch Experiments\nEnter: ")



    # SINGLE TEST MODE


    if mode == "1":

        choice = input("Use random nodes? (y/n): ")

        if choice.lower() == "y":

            start = random.choice(nodes)
            target = random.choice(nodes)

            while target == start:
                target = random.choice(nodes)

        else:

            start = get_node_input("Enter START node", n)

            target = get_node_input("Enter TARGET node", n)

            while target == start:
                print("Target cannot equal start.")
                target = get_node_input("Enter TARGET node", n)

        path, success = greedy_routing(G, start, target)

        print_stats(start, target, path, success)

        visualize_routing(G, shortcuts, path)



    # BATCH EXPERIMENT MODE
   

    elif mode == "2":

        results = run_master_experiments(
            G,
            shortcuts,
            num_experiments,
            experiment_id,
            n,
            use_random=True
        )

        export_experiments(results)

        export_summary(results)

        print("\nExperiments complete.")
        print("Exported to:")
        print("experiments.csv")
        print("summary.csv")
