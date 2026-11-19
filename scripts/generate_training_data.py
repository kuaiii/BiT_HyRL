"""Generate the synthetic training corpus.

Usage::

    python scripts/generate_training_data.py --min_nodes 50 --max_nodes 150 --count 100
    python scripts/generate_training_data.py --count_per_type 200 --types ba ws er
"""
import os
import sys
import argparse
import random
import numpy as np
import networkx as nx
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, 'dataset', 'all', 'syn')


def generate_ba_graph(n, m=3, seed=None):
    """Build a BA scale-free graph."""
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G


def generate_er_graph(n, p=None, seed=None):
    """Build an ER random graph."""
    if p is None:
        p = 2 * np.log(n) / n
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            G.add_edge(u, v)
    
    return G


def generate_ws_graph(n, k=4, p=0.3, seed=None):
    """Build a WS small-world graph."""
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return G


def generate_powerlaw_cluster_graph(n, m=3, p=0.5, seed=None):
    """Build a power-law clustered graph."""
    try:
        G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)
        return G
    except:
        return generate_ba_graph(n, m, seed)


def ensure_connected(G):
    """Keep the largest connected component."""
    if nx.is_connected(G):
        return G
    
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        u = random.choice(list(components[i]))
        v = random.choice(list(components[i + 1]))
        G.add_edge(u, v)
    
    return G


def generate_graphs(min_nodes, max_nodes, count=None, graph_types=['ba'], output_dir=OUTPUT_DIR, count_per_type=None):
    """Generate graphs within the requested size range."""
    os.makedirs(output_dir, exist_ok=True)
    if count_per_type is not None:
        total = count_per_type * len(graph_types)
        print(f"{count_per_type} graphs per type, {total} in total (nodes: {min_nodes}-{max_nodes})")
    else:
        total = count or 100
        print(f"{total} graphs (nodes: {min_nodes}-{max_nodes})")
    print(f"graph types: {graph_types}")
    print(f"output directory: {output_dir}")

    existing_counts = {}
    for t in graph_types:
        prefix = f"{t.upper()}_gen_"
        existing = [f for f in os.listdir(output_dir) if f.startswith(prefix)]
        existing_counts[t] = len(existing)

    generated = 0
    failed = 0
    pbar = tqdm(total=total, desc="Generating")

    def try_generate_one(graph_type, n, seed):
        if graph_type == 'ba':
            m = random.randint(2, min(4, n // 2))
            G = generate_ba_graph(n, m, seed)
        elif graph_type == 'er':
            G = generate_er_graph(n, seed=seed)
        elif graph_type == 'ws':
            k = random.randint(4, min(8, n // 2))
            p = random.uniform(0.1, 0.5)
            G = generate_ws_graph(n, k, p, seed)
        elif graph_type == 'plc':
            m = random.randint(2, min(4, n // 2))
            p = random.uniform(0.3, 0.7)
            G = generate_powerlaw_cluster_graph(n, m, p, seed)
        else:
            return None
        G = ensure_connected(G)
        return G if nx.is_connected(G) else None

    if count_per_type is not None:
        for graph_type in graph_types:
            type_generated = 0
            while type_generated < count_per_type:
                n = random.randint(min_nodes, max_nodes)
                seed = random.randint(0, 1000000)
                try:
                    G = try_generate_one(graph_type, n, seed)
                    if G is None:
                        failed += 1
                        continue
                    idx = existing_counts[graph_type] + type_generated
                    filename = f"{graph_type.upper()}_gen_{n}n_{idx}.gml"
                    filepath = os.path.join(output_dir, filename)
                    nx.write_gml(G, filepath)
                    type_generated += 1
                    generated += 1
                    pbar.update(1)
                except Exception:
                    failed += 1
    else:
        type_generated = {t: 0 for t in graph_types}
        while generated < total:
            n = random.randint(min_nodes, max_nodes)
            graph_type = random.choice(graph_types)
            seed = random.randint(0, 1000000)
            try:
                G = try_generate_one(graph_type, n, seed)
                if G is None:
                    failed += 1
                    continue
                type_generated[graph_type] += 1
                idx = existing_counts[graph_type] + type_generated[graph_type]
                filename = f"{graph_type.upper()}_gen_{n}n_{idx}.gml"
                filepath = os.path.join(output_dir, filename)
                nx.write_gml(G, filepath)
                generated += 1
                pbar.update(1)
            except Exception:
                failed += 1

    pbar.close()
    print(f"\ndone: {generated} graphs")
    if failed > 0:
        print(f"failed: {failed} attempts")
    return generated


def analyze_distribution(data_dir):
    """Report the node-count distribution of the corpus."""
    sizes = []
    
    for f in os.listdir(data_dir):
        if f.endswith('.gml'):
            try:
                G = nx.read_gml(os.path.join(data_dir, f))
                sizes.append(G.number_of_nodes())
            except:
                pass
    
    if not sizes:
        print("no training data found")
        return
    
    print("\ntraining corpus: size distribution")
    print(f"{'='*50}")
    print(f"graphs: {len(sizes)}")
    print(f"node range: {min(sizes)} - {max(sizes)}")
    print(f"mean node count: {sum(sizes)/len(sizes):.1f}")
    print()
    
    ranges = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, 1000)]
    print("node counts:")
    for lo, hi in ranges:
        count = sum(1 for s in sizes if lo <= s < hi)
        pct = count / len(sizes) * 100
        bar = '█' * int(pct / 2)
        print(f"  {lo:4d}-{hi:4d}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='generate the synthetic training corpus')
    parser.add_argument('--min_nodes', type=int, default=50, help='minimum node count')
    parser.add_argument('--max_nodes', type=int, default=150, help='maximum node count')
    parser.add_argument('--count', type=int, default=None, help='total graphs (alternative to --count_per_type)')
    parser.add_argument('--count_per_type', type=int, default=None,
                        help='graphs per type, e.g. 200 means 200 each of BA, WS and ER')
    parser.add_argument('--types', nargs='+', default=['ba'],
                        choices=['ba', 'er', 'ws', 'plc'],
                        help='graph types: ba, er, ws, plc')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='output directory')
    parser.add_argument('--analyze', action='store_true', help='only report the existing distribution')

    args = parser.parse_args()

    if args.analyze:
        analyze_distribution(args.output_dir)
        return

    count = args.count
    if args.count_per_type is not None:
        count = None
    elif count is None:
        count = 100

    generate_graphs(
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        count=count,
        graph_types=args.types,
        output_dir=args.output_dir,
        count_per_type=args.count_per_type,
    )
    
    analyze_distribution(args.output_dir)


if __name__ == '__main__':
    main()
