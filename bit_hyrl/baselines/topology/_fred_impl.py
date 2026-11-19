import numpy as np
import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
import random
import warnings

warnings.filterwarnings("ignore")


class IIoTEnv:
    """IIoT environment built on a predefined graph and a communication radius."""
    def __init__(self, G, comm_range=0.3, seed=42):
        np.random.seed(seed)
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


class ACC_Codec:
    """Auxiliary continuous coding: keeps nearby topologies close in the continuous space."""
    def __init__(self, dim, p1=0.625):
        self.dim = dim
        self.p1 = p1
        self.p0 = 1.0 - p1

    def decode(self, value):
        """Map a continuous value in [0, 1) to a binary vector."""
        value = np.clip(value, 0.0, 0.9999999999)
        vector = []
        low = 0.0
        high = 1.0
        z = 0

        for _ in range(self.dim):
            span = high - low
            if z == 0:
                mid = low + span * self.p0
                if value < mid:
                    bit = 0
                    high = mid
                else:
                    bit = 1
                    low = mid
            else:
                mid = low + span * self.p1
                if value < mid:
                    bit = 1
                    high = mid
                else:
                    bit = 0
                    low = mid

            vector.append(bit)
            z = bit ^ z

        return np.array(vector)


class RobustnessEvaluator:
    """Robustness evaluator for R = (1/N) * sum(T_theta)."""
    @staticmethod
    def calculate_R(adj_matrix, attack_type='malicious', sample_ratio=0.3):
        """Robustness R of an adjacency matrix."""
        G = nx.from_numpy_array(adj_matrix)
        n = G.number_of_nodes()
        if n == 0: 
            return 0

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

        return (sum_tolerance / max_steps) * (max_steps / n) if max_steps > 0 else 0


class FRED_ABL:
    """Bayesian optimisation over the coded space."""
    def __init__(self, env):
        self.env = env
        self.codec = ACC_Codec(dim=env.m)
        self.evaluator = RobustnessEvaluator()

        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e2))
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)

        self.X_train = []
        self.y_train = []
        self.best_R = -1.0
        self.best_topology = None

    def ucb_acquisition(self, num_samples=300, kappa=2.576):
        """Upper confidence bound acquisition.
        """
        candidates = np.random.uniform(0, 1, (num_samples, 1))
        
        if len(self.X_train) == 0:
            return candidates[0, 0]

        X = np.array(self.X_train).reshape(-1, 1)
        self.gpr.fit(X, np.array(self.y_train))
        
        mus, stds = self.gpr.predict(candidates, return_std=True)
        ucb_values = mus + kappa * stds
        best_idx = np.argmax(ucb_values)
        return float(candidates[best_idx, 0])

    def run(self, iterations=30, initial_samples=5, verbose=True):
        import time
        
        start_time = time.time()
        if verbose:
            print(f"  [FRED-ABL] initial sampling ({initial_samples} points)...")

        for i in range(initial_samples):
            x_sample = np.random.uniform(0, 1)
            binary_vec = self.codec.decode(x_sample)
            adj = self.env.vector_to_adj(binary_vec)
            r_val = self.evaluator.calculate_R(adj, sample_ratio=0.3)

            self.X_train.append([float(x_sample)])
            self.y_train.append(r_val)

            if r_val > self.best_R:
                self.best_R = r_val
                self.best_topology = adj.copy()

            if verbose and (i == 0 or i == initial_samples - 1 or r_val > self.best_R * 0.95):
                elapsed = time.time() - start_time
                print(f"  [FRED-ABL] init {i + 1}/{initial_samples}: R = {r_val:.4f} ({elapsed:.2f}s)")

        if verbose:
            print(f"  [FRED-ABL] Bayesian optimisation ({iterations} iterations)...")

        for i in range(iterations):
            x_next = self.ucb_acquisition()
            binary_vec_next = self.codec.decode(x_next)
            adj_next = self.env.vector_to_adj(binary_vec_next)
            r_next = self.evaluator.calculate_R(adj_next, sample_ratio=0.3)

            self.X_train.append([float(x_next)])
            self.y_train.append(r_next)

            if r_next > self.best_R:
                self.best_R = r_next
                self.best_topology = adj_next.copy()
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  [FRED-ABL] iter {i + 1}/{iterations}: new best R = {r_next:.4f} ({elapsed:.2f}s)")
            elif verbose and (i % 5 == 0 or i == iterations - 1):
                elapsed = time.time() - start_time
                print(f"  [FRED-ABL] iter {i + 1}/{iterations}: R = {r_next:.4f} ({elapsed:.2f}s)")

        if verbose:
            total_time = time.time() - start_time
            edge_count = np.sum(self.best_topology) // 2
            print(f"  [FRED-ABL] done | best R: {self.best_R:.4f} | edges: {edge_count} | {total_time:.2f}s")
        return self.best_topology
