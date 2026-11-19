"""Quantum-inspired learning model for IoT topology reconstruction."""

import numpy as np
import networkx as nx
import random
import math
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


class QuantumEnvironment:
    """Graph and communication constraints for the reconstruction."""
    def __init__(self, G, comm_range=0.3, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.r = comm_range
        
        self.n = G.number_of_nodes()
        self.positions = np.zeros((self.n, 2))
        
        if sorted(G.nodes()) != list(range(self.n)):
            raise ValueError("Graph nodes must be labeled 0 to n-1 consecutively")
        
        for i in range(self.n):
            if 'pos' not in G.nodes[i]:
                raise ValueError(f"Node {i} missing 'pos' attribute")
            self.positions[i] = np.array(G.nodes[i]['pos'])
        
        self.dist_matrix = np.zeros((self.n, self.n))
        self.possible_edges = []
        
        for u in range(self.n):
            for v in range(u + 1, self.n):
                dist = np.linalg.norm(self.positions[u] - self.positions[v])
                self.dist_matrix[u][v] = self.dist_matrix[v][u] = dist
                
                if dist <= self.r:
                    self.possible_edges.append((u, v))
        
        self.m = len(self.possible_edges)
        self.target_edges = G.number_of_edges()
    
    def vector_to_adj(self, binary_vector):
        """Turn a binary vector into an adjacency matrix with the same edge count."""
        adj = np.zeros((self.n, self.n), dtype=int)
        selected_edges = []
        for idx, val in enumerate(binary_vector):
            if val == 1 and idx < len(self.possible_edges):
                selected_edges.append(self.possible_edges[idx])
        
        if len(selected_edges) > self.target_edges:
            np.random.shuffle(selected_edges)
            selected_edges = selected_edges[:self.target_edges]
        elif len(selected_edges) < self.target_edges:
            available = [e for e in self.possible_edges if e not in selected_edges]
            needed = self.target_edges - len(selected_edges)
            if len(available) >= needed:
                additional = np.random.choice(len(available), needed, replace=False)
                for idx in additional:
                    selected_edges.append(available[idx])
        
        for u, v in selected_edges:
            adj[u][v] = adj[v][u] = 1
        return adj


class QuantumChromosome:
    """Quantum chromosome: one qubit per bit, with amplitudes alpha and beta."""
    def __init__(self, length):
        self.length = length
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        self.alpha = np.full(length, sqrt2_inv, dtype=float)
        self.beta = np.full(length, sqrt2_inv, dtype=float)
    
    def observe(self):
        """Measure the amplitudes into a classical bit string."""
        binary = []
        for i in range(self.length):
            prob_1 = self.beta[i] ** 2
            if random.random() < prob_1:
                binary.append(1)
            else:
                binary.append(0)
        return np.array(binary)
    
    def update_quantum_gate(self, index, rotation_angle, direction):
        """Apply a quantum rotation gate to one qubit.
        """
        if index >= self.length:
            return
        
        cos_theta = math.cos(rotation_angle * direction)
        sin_theta = math.sin(rotation_angle * direction)
        
        alpha_new = self.alpha[index] * cos_theta - self.beta[index] * sin_theta
        beta_new = self.alpha[index] * sin_theta + self.beta[index] * cos_theta
        
        norm = math.sqrt(alpha_new**2 + beta_new**2)
        if norm > 1e-10:
            self.alpha[index] = alpha_new / norm
            self.beta[index] = beta_new / norm
        else:
            sqrt2_inv = 1.0 / np.sqrt(2.0)
            self.alpha[index] = sqrt2_inv
            self.beta[index] = sqrt2_inv


class RobustnessEvaluator:
    """Robustness evaluator."""
    @staticmethod
    def calculate_robustness(adj_matrix, attack_type='malicious', sample_ratio=0.3):
        """Robustness R of an adjacency matrix under the given attack.
        """
        G = nx.from_numpy_array(adj_matrix)
        n = G.number_of_nodes()
        if n == 0:
            return 0.0
        
        max_steps = max(5, int(n * sample_ratio))
        
        nodes = list(G.nodes())
        if attack_type == 'malicious':
            nodes_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
            removal_queue = [node for node, degree in nodes_sorted]
        else:
            removal_queue = list(G.nodes())
            random.shuffle(removal_queue)
        
        sum_tolerance = 0.0
        G_temp = G.copy()
        
        for theta in range(min(max_steps, n)):
            if G_temp.number_of_nodes() > 0:
                try:
                    components = list(nx.connected_components(G_temp))
                    if components:
                        largest_cc = max(components, key=len)
                        phi_theta = len(largest_cc)
                    else:
                        phi_theta = 0
                except:
                    phi_theta = 0
            else:
                phi_theta = 0
            
            remaining_count = n - theta
            t_theta = phi_theta / remaining_count if remaining_count > 0 else 0
            sum_tolerance += t_theta
            
            if theta < len(removal_queue):
                node_to_remove = removal_queue[theta]
                if G_temp.has_node(node_to_remove):
                    G_temp.remove_node(node_to_remove)
            else:
                break
        
        return (sum_tolerance / max_steps) * (max_steps / n) if max_steps > 0 else 0.0


class QuantumLearningModel:
    """Quantum genetic search with adaptive learning."""
    def __init__(self, env, pop_size=30, max_iterations=50):
        self.env = env
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.evaluator = RobustnessEvaluator()
        
        self.quantum_population = [QuantumChromosome(env.m) for _ in range(pop_size)]
        
        self.classical_population = []
        self.fitness_values = []
        
        self.best_fitness = -1.0
        self.best_solution = None
        self.best_adj = None
        
        self.rotation_angle_base = 0.1 * math.pi
        self.learning_rate = 0.1
        
    def initialize_population(self):
        """Initialise the classical population."""
        self.classical_population = []
        self.fitness_values = []
        
        for q_chrom in self.quantum_population:
            binary_vec = q_chrom.observe()
            adj = self.env.vector_to_adj(binary_vec)
            fitness = self.evaluator.calculate_robustness(adj)
            
            self.classical_population.append(binary_vec)
            self.fitness_values.append(fitness)
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = binary_vec.copy()
                self.best_adj = adj.copy()
    
    def quantum_rotation_gate_update(self, q_chrom, classical_solution, best_solution, index):
        """Quantum rotation-gate update.
        """
        if index >= len(classical_solution) or index >= len(best_solution):
            return
        
        current_bit = classical_solution[index]
        best_bit = best_solution[index]
        
        if current_bit == 0 and best_bit == 1:
            direction = 1
        elif current_bit == 1 and best_bit == 0:
            direction = -1
        else:
            direction = 0
        
        if direction != 0:
            rotation_angle = self.rotation_angle_base * (1 + self.learning_rate)
            q_chrom.update_quantum_gate(index, rotation_angle, direction)
    
    def crossover(self, parent1, parent2):
        """Single-point crossover."""
        if len(parent1) != len(parent2):
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def mutation(self, solution, mutation_rate=0.01):
        """Mutation."""
        mutated = solution.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def selection(self):
        """Tournament selection."""
        tournament_size = 3
        selected_indices = []
        
        for _ in range(self.pop_size):
            candidates = random.sample(range(self.pop_size), min(tournament_size, self.pop_size))
            winner = max(candidates, key=lambda i: self.fitness_values[i])
            selected_indices.append(winner)
        
        return selected_indices
    
    def run(self, verbose=True):
        """Run the quantum search."""
        import time
        start_time = time.time()
        
        if verbose:
            print(f"  [QDLM] initialising the population (size {self.pop_size})...")
        
        self.initialize_population()
        if verbose:
            elapsed = time.time() - start_time
            print(f"  [QDLM] initialised | best R = {self.best_fitness:.4f} ({elapsed:.2f}s)")
        
        if verbose:
            print(f"  [QDLM] optimising ({self.max_iterations} generations)...")
        
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            
            selected_indices = self.selection()
            
            for i, selected_idx in enumerate(selected_indices):
                q_chrom = self.quantum_population[i]
                classical_sol = self.classical_population[selected_idx]
                
                for j in range(min(len(classical_sol), q_chrom.length)):
                    self.quantum_rotation_gate_update(q_chrom, classical_sol, 
                                                      self.best_solution, j)
            
            new_classical = []
            new_fitness = []
            
            for q_chrom in self.quantum_population:
                binary_vec = q_chrom.observe()
                adj = self.env.vector_to_adj(binary_vec)
                fitness = self.evaluator.calculate_robustness(adj, sample_ratio=0.3)
                
                new_classical.append(binary_vec)
                new_fitness.append(fitness)
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = binary_vec.copy()
                    self.best_adj = adj.copy()
                    if verbose:
                        elapsed = time.time() - start_time
                        print(f"  [QDLM] gen {iteration + 1}/{self.max_iterations}: new best R = {fitness:.4f} ({elapsed:.2f}s)")
            
            if iteration % 5 == 0:
                for i in range(0, self.pop_size - 1, 2):
                    if random.random() < 0.7:
                        child1, child2 = self.crossover(new_classical[i], new_classical[i+1])
                        new_classical[i] = self.mutation(child1)
                        new_classical[i+1] = self.mutation(child2)
                        
                        adj1 = self.env.vector_to_adj(new_classical[i])
                        adj2 = self.env.vector_to_adj(new_classical[i+1])
                        new_fitness[i] = self.evaluator.calculate_robustness(adj1, sample_ratio=0.3)
                        new_fitness[i+1] = self.evaluator.calculate_robustness(adj2, sample_ratio=0.3)
            
            self.classical_population = new_classical
            self.fitness_values = new_fitness
            
            self.rotation_angle_base *= 0.99
            
            if verbose and (iteration % 5 == 0 or iteration == self.max_iterations - 1):
                elapsed = time.time() - start_time
                print(f"  [QDLM] gen {iteration + 1}/{self.max_iterations}: best R = {self.best_fitness:.4f} ({elapsed:.2f}s)")
        
        if verbose:
            total_time = time.time() - start_time
            edge_count = np.sum(self.best_adj) // 2
            print(f"  [QDLM] done | best R: {self.best_fitness:.4f} | edges: {edge_count} | {total_time:.2f}s")
        
        return self.best_adj
