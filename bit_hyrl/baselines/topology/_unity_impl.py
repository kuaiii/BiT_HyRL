import numpy as np
import networkx as nx
import random
import math
from typing import Tuple, List, Dict, get_origin
import time
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class UNITY:
    def __init__(self, area_diameter: float = 500, comm_range: float = 200, epsilon: float = 1e-9):
        """Initialise UNITY."""
        self.area_diameter = area_diameter
        self.comm_range = comm_range
        self.epsilon = epsilon
        
    def generate_robust_topology(self, N: int, M: int) -> nx.Graph:
        """Generate a robust topology; the core construction."""
        G = self._initialize_network(N)
        
        for m in range(M):
            for i in range(N):
                neighbor_list, degree_list = self._get_potential_neighbors(G, i)
                
                if not neighbor_list:
                    continue
                    
                probability_list = self._mapping_operator(degree_list)
                
                selected_idx = self._roulette_selection(probability_list)
                if selected_idx is not None:
                    k = neighbor_list[selected_idx]
                    if not G.has_edge(i, k):
                        G.add_edge(i, k)
        
        return G
    
    def optimize_topology_structure(self, G: nx.Graph) -> nx.Graph:
        """Rebuild a robust topology from the structure of an existing graph."""
        N = G.number_of_nodes()
        M = max(1, int(2 * G.number_of_edges() / N))
        
        print(f"parameters: N={N}, M={M}, input edges={G.number_of_edges()}")
        
        optim_G = self.generate_robust_topology(N, M)
        
        return optim_G
    
    def _initialize_network(self, N: int) -> nx.Graph:
        """Initialise the empty topology."""
        G = nx.Graph()
        
        for i in range(N):
            r = self.area_diameter/2 * math.sqrt(random.random())
            theta = random.random() * 2 * math.pi
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            
            G.add_node(i, pos=(x, y))
            
        return G
    
    def _get_potential_neighbors(self, G: nx.Graph, node_idx: int) -> Tuple[List[int], List[int]]:
        """Potential neighbours of every node, with their degrees."""
        neighbor_list = []
        degree_list = []
        
        node_pos = G.nodes[node_idx]['pos']
        
        for j in G.nodes():
            if j == node_idx or G.has_edge(node_idx, j):
                continue
                
            j_pos = G.nodes[j]['pos']
            
            distance = math.sqrt((node_pos[0] - j_pos[0])**2 + (node_pos[1] - j_pos[1])**2)
            
            if distance <= self.comm_range:
                neighbor_list.append(j)
                degree_list.append(G.degree(j))
                
        return neighbor_list, degree_list
    
    def _mapping_operator(self, degree_list: List[int]) -> List[float]:
        """Map node degrees to connection probabilities.
        """
        if not degree_list:
            return []
            
        adjusted_degrees = [d + self.epsilon for d in degree_list]
        
        inverse_degrees = [1.0 / d for d in adjusted_degrees]
        sum_inverse = sum(inverse_degrees)
        
        probability_list = [inv_d / sum_inverse for inv_d in inverse_degrees]
        
        return probability_list
    
    def _roulette_selection(self, probability_list: List[float]) -> int:
        """Roulette-wheel selection."""
        if not probability_list:
            return None
            
        cum_prob = np.cumsum(probability_list)
        
        r = random.random()
        
        for i, prob in enumerate(cum_prob):
            if r <= prob:
                return i
                
        return len(probability_list) - 1
    
    def generate_matched_topology(self, input_graph: nx.Graph) -> nx.Graph:
        """Rebuild a topology with exactly the input node and edge counts."""
        N = input_graph.number_of_nodes()
        target_edges = input_graph.number_of_edges()
        
        matched_graph = nx.Graph()
        for node in input_graph.nodes():
            if 'pos' in input_graph.nodes[node]:
                matched_graph.add_node(node, pos=input_graph.nodes[node]['pos'])
            else:
                r = self.area_diameter/2 * math.sqrt(random.random())
                theta = random.random() * 2 * math.pi
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                matched_graph.add_node(node, pos=(x, y))
        
        edges_added = 0
        max_iter = target_edges * 10
        iter_count = 0
        while edges_added < target_edges and iter_count < max_iter:
            iter_count += 1
            made_progress = False
            for i in range(N):
                if edges_added >= target_edges:
                    break
                neighbor_list, degree_list = self._get_potential_neighbors(matched_graph, i)
                if not neighbor_list:
                    continue
                probability_list = self._mapping_operator(degree_list)
                selected_idx = self._roulette_selection(probability_list)
                if selected_idx is not None:
                    k = neighbor_list[selected_idx]
                    if not matched_graph.has_edge(i, k):
                        matched_graph.add_edge(i, k)
                        edges_added += 1
                        made_progress = True
            if not made_progress:
                break
        return matched_graph


 
 

    
    
    
