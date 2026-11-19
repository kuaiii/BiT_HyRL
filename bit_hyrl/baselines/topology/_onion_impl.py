import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

class OnionNetworkOptimizer:
    def __init__(self, G):
        """Initialise the onion optimiser."""
        self.G = G.copy()
        self.N = len(G.nodes())
        self.original_degree_sequence = [G.degree(node) for node in G.nodes()]
        
    def robustness_measure(self, G, attack_type="HDA"):
        """Robustness R of a graph under the given attack."""
        if len(G.nodes()) == 0:
            return 0.0
        
        G_copy = G.copy()
        N = len(G_copy.nodes())
        
        s_values = []
        
        largest_cc = max(nx.connected_components(G_copy), key=len)
        s_values.append(len(largest_cc) / N)
        
        if attack_type == "HDA":
            nodes_removed = 0
            while nodes_removed < N and len(G_copy.nodes()) > 0:
                degrees = dict(G_copy.degree())
                max_degree = max(degrees.values())
                max_degree_nodes = [n for n, d in degrees.items() if d == max_degree]
                
                target_node = random.choice(max_degree_nodes)
                
                G_copy.remove_node(target_node)
                nodes_removed += 1
                
                if len(G_copy.nodes()) > 0:
                    connected_components = list(nx.connected_components(G_copy))
                    if connected_components:
                        largest_cc = max(connected_components, key=len)
                        s_values.append(len(largest_cc) / N)
                    else:
                        s_values.append(0.0)
                else:
                    s_values.append(0.0)
        else:
            nodes_list = list(G_copy.nodes())
            random.shuffle(nodes_list)
            
            for target_node in nodes_list:
                if target_node in G_copy.nodes():
                    G_copy.remove_node(target_node)
                
                if len(G_copy.nodes()) > 0:
                    connected_components = list(nx.connected_components(G_copy))
                    if connected_components:
                        largest_cc = max(connected_components, key=len)
                        s_values.append(len(largest_cc) / N)
                    else:
                        s_values.append(0.0)
                else:
                    s_values.append(0.0)
        
        while len(s_values) < N + 1:
            s_values.append(0.0)
        
        s_values = s_values[:N + 1]
        
        R = sum(s_values) / (N + 1)
        return R
    
    def optimize_network(self, max_iterations=10000, threshold=0.0, tolerance=0.01, verbose=True):
        """Optimise robustness with degree-preserving edge swaps."""
        G_optimized = self.G.copy()
        R_old = self.robustness_measure(G_optimized, attack_type="HDA")
        
        if verbose:
            print(f"initial robustness: {R_old:.4f}")
            print("optimising...")
            
        recent_improvements = []
        last_check_point = 0
        
        for i in tqdm(range(max_iterations), disable=not verbose):
            edges = list(G_optimized.edges())
            if len(edges) < 2:
                break
                
            edge1, edge2 = random.sample(edges, 2)
            i, j = edge1
            k, l = edge2
            
            if len({i, j, k, l}) < 4:
                continue
                
            if G_optimized.has_edge(i, k) or G_optimized.has_edge(j, l):
                continue
            
            G_optimized.remove_edge(i, j)
            G_optimized.remove_edge(k, l)
            
            G_optimized.add_edge(i, k)
            G_optimized.add_edge(j, l)
            
            R_new = self.robustness_measure(G_optimized, attack_type="HDA")
            
            if R_new > R_old + threshold:
                R_old = R_new
                recent_improvements.append(R_new - R_old)
            else:
                G_optimized.remove_edge(i, k)
                G_optimized.remove_edge(j, l)
                G_optimized.add_edge(i, j)
                G_optimized.add_edge(k, l)
            
            if i - last_check_point >= 1000:
                last_check_point = i
                recent_improvement = sum(recent_improvements[-1000:] if recent_improvements else [0])
                
                if recent_improvement < tolerance * R_old:
                    if verbose:
                        print(f"\nconverged at iteration {i}: total improvement over the last 1000 iterations = {recent_improvement:.6f}")
                    break
                    
                if verbose and i % 10000 == 0:
                    print(f"iteration {i}: robustness = {R_old:.4f}")
        
        if verbose:
            print(f"final robustness: {R_old:.4f}")
            
        return G_optimized
    
    def analyze_onion_structure(self, G):
        """Describe the onion structure of the graph."""
        k_nn = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                k_nn[node] = sum(G.degree(neighbor) for neighbor in neighbors) / len(neighbors)
            else:
                k_nn[node] = 0
        
        clustering = nx.clustering(G)
        
        k_nn_k = {}
        for node, degree in G.degree():
            if degree not in k_nn_k:
                k_nn_k[degree] = []
            k_nn_k[degree].append(k_nn[node])
        
        for k in k_nn_k:
            k_nn_k[k] = sum(k_nn_k[k]) / len(k_nn_k[k])
        
        clustering_k = {}
        for node, degree in G.degree():
            if degree not in clustering_k:
                clustering_k[degree] = []
            clustering_k[degree].append(clustering[node])
        
        for k in clustering_k:
            if clustering_k[k]:
                clustering_k[k] = sum(clustering_k[k]) / len(clustering_k[k])
            else:
                clustering_k[k] = 0
        
        onion_property = {}
        degrees = dict(G.degree())
        
        for k in set(degrees.values()):
            nodes_with_degree_k = [node for node, deg in degrees.items() if deg == k]
            
            if len(nodes_with_degree_k) <= 1:
                onion_property[k] = 0
                continue
            
            low_degree_nodes = [node for node, deg in degrees.items() if deg <= k]
            subgraph = G.subgraph(low_degree_nodes)
            
            connected_components = list(nx.connected_components(subgraph))
            
            max_component_size = 0
            for component in connected_components:
                k_nodes_in_component = [node for node in component if degrees[node] == k]
                if len(k_nodes_in_component) > max_component_size:
                    max_component_size = len(k_nodes_in_component)
            
            onion_property[k] = max_component_size / len(nodes_with_degree_k)
        
        return {
            'k_nn_k': k_nn_k,
            'clustering_k': clustering_k,
            'onion_property': onion_property
        }


        
def compare_degree_distributions(initial_graph, optimized_graph, 
                                 initial_name="Initial Network", 
                                 optimized_name="Optimized Network",
                                 show_plot=True):
    """Compare the degree distributions of two graphs."""
    initial_degrees = [d for _, d in initial_graph.degree()]
    optimized_degrees = [d for _, d in optimized_graph.degree()]
    
    stats = {
        "initial": {
            "nodes": initial_graph.number_of_nodes(),
            "edges": initial_graph.number_of_edges(),
            "mean_degree": sum(initial_degrees) / len(initial_degrees),
            "max_degree": max(initial_degrees)
        },
        "optimized": {
            "nodes": optimized_graph.number_of_nodes(),
            "edges": optimized_graph.number_of_edges(),
            "mean_degree": sum(optimized_degrees) / len(optimized_degrees),
            "max_degree": max(optimized_degrees)
        }
    }
    
    node_change = ((stats["optimized"]["nodes"] - stats["initial"]["nodes"]) / 
                   stats["initial"]["nodes"]) * 100
    edge_change = ((stats["optimized"]["edges"] - stats["initial"]["edges"]) / 
                   stats["initial"]["edges"]) * 100
    mean_degree_change = ((stats["optimized"]["mean_degree"] - stats["initial"]["mean_degree"]) / 
                          stats["initial"]["mean_degree"]) * 100
    max_degree_change = ((stats["optimized"]["max_degree"] - stats["initial"]["max_degree"]) / 
                         stats["initial"]["max_degree"]) * 100
    
    from scipy import stats as scipy_stats
    ks_stat, ks_pvalue = scipy_stats.ks_2samp(initial_degrees, optimized_degrees)
    
    print("Degree Distribution Analysis:")
    print(f"1. Nodes: {stats['initial']['nodes']} -> {stats['optimized']['nodes']} (Change: {node_change:.2f}%)")
    print(f"2. Edges: {stats['initial']['edges']} -> {stats['optimized']['edges']} (Change: {edge_change:.2f}%)")
    print(f"3. Mean degree: {stats['initial']['mean_degree']:.2f} -> {stats['optimized']['mean_degree']:.2f} (Change: {mean_degree_change:.2f}%)")
    print(f"4. Max degree: {stats['initial']['max_degree']} -> {stats['optimized']['max_degree']} (Change: {max_degree_change:.2f}%)")
    print(f"5. KS test: p={ks_pvalue:.4f} {'(Similar distributions)' if ks_pvalue > 0.05 else '(Different distributions)'}")
    
    if show_plot:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        max_degree = max(stats["initial"]["max_degree"], stats["optimized"]["max_degree"])
        bins = range(0, max_degree + 2)
        
        plt.hist(initial_degrees, bins=bins, alpha=0.7, density=True, 
                 label=f"{initial_name} (Mean={stats['initial']['mean_degree']:.2f})")
        plt.hist(optimized_degrees, bins=bins, alpha=0.7, density=True, 
                 label=f"{optimized_name} (Mean={stats['optimized']['mean_degree']:.2f})")
        
        plt.title("Degree Distribution Comparison")
        plt.xlabel("Degree")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    
    return {
        "initial": {
            "nodes": stats["initial"]["nodes"],
            "edges": stats["initial"]["edges"],
            "mean_degree": stats["initial"]["mean_degree"],
            "max_degree": stats["initial"]["max_degree"]
        },
        "optimized": {
            "nodes": stats["optimized"]["nodes"],
            "edges": stats["optimized"]["edges"],
            "mean_degree": stats["optimized"]["mean_degree"],
            "max_degree": stats["optimized"]["max_degree"]
        },
        "change": {
            "nodes": node_change,
            "edges": edge_change,
            "mean_degree": mean_degree_change,
            "max_degree": max_degree_change
        },
        "ks_test": {
            "statistic": ks_stat,
            "pvalue": ks_pvalue
        }
    }
  


 
 

   
   


    
