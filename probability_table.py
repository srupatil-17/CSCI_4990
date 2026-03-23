import networkx as nx

n = 5

def manhattan(u, v):
    return abs(u[0] - v[0]) + abs(u[1] - v[1])

nodes = [(i, j) for i in range(n) for j in range(n)]

probabilities = {}

for u in nodes:
    rows = []
    total_weight = 0.0

    for v in nodes:
        if u == v:
            continue
        d = manhattan(u, v)
        w = 1 / (d ** 2)
        rows.append([v, d, w])
        total_weight += w

    cumulative = 0.0
    table = []

    for v, d, w in sorted(rows, key=lambda x: x[1]):
        p = w / total_weight
        cumulative += p
        table.append((v, d, w, p, cumulative))

    probabilities[u] = (table, total_weight)

# ---- PRINT TABLES ----

for u in nodes:
    table, total_weight = probabilities[u]

    print("\n" + "=" * 70)
    print(f"Source node u = {u}")
    print(f"Total unnormalized weight Σ(1/d²) = {total_weight:.6f}")
    print("=" * 70)
    print(f"{'Target v':>10} | {'d(u,v)':>6} | {'1/d^2':>10} | {'p(u,v)':>10} | {'CDF':>10}")
    print("-" * 70)

    for v, d, w, p, cdf in table:
        print(f"{str(v):>10} | {d:6d} | {w:10.6f} | {p:10.6f} | {cdf:10.6f}")

    total_p = sum(row[3] for row in table)
    print("-" * 70)
    print(f"{'Σ p(u,v)':>32} | {total_p:10.6f}")

