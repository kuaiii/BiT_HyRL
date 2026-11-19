import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import networkx as nx
import random
import copy
from collections import deque
import itertools
import os
import time
from tqdm import tqdm
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

_FAST = os.environ.get("ROMEN_LEGACY", "0") != "1"


def _edge_arrays(graph) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints of every edge as a ``(lo, hi)`` pair of int arrays with ``lo < hi``."""
    m = graph.number_of_edges()
    if m == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    flat = np.fromiter(itertools.chain.from_iterable(graph.edges()),
                       dtype=np.int64, count=2 * m)
    pair = flat.reshape(-1, 2)
    return np.minimum(pair[:, 0], pair[:, 1]), np.maximum(pair[:, 0], pair[:, 1])


def _triu_position(n: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Index of ``(lo, hi)`` inside ``np.triu_indices(n, 1)``, in C order."""
    return lo * n - lo * (lo + 1) // 2 + (hi - lo - 1)


class NetworkTopology:
    """Network topology used by the ROMEN construction."""
    
    def __init__(self, num_nodes=None, density=None, area_size=None, comm_distance=None, 
                 graph=None, positions=None):
        """Initialise the topology."""
        if graph is not None:
            self.graph = graph.copy()
            self.num_nodes = graph.number_of_nodes()
            self.positions = positions if positions else self._generate_positions_from_graph()
            self.density = max(1, int(np.mean([graph.degree(n) for n in graph.nodes()])))
            self.area_size = area_size if area_size else 1000
            self.comm_distance = comm_distance if comm_distance else self._estimate_comm_distance()
        else:
            self.num_nodes = num_nodes
            self.density = density
            self.area_size = area_size
            self.comm_distance = comm_distance
            self.positions = {}
            self.graph = None
            
        self._valid_connections_cache = None
        self._distance_matrix = None

    def __deepcopy__(self, memo):
        """OPT(port): cheap structural clone instead of generic recursive deepcopy."""
        new = copy.copy(self)
        new.graph = self.graph.copy() if self.graph is not None else None
        new.positions = dict(self.positions)
        new._distance_matrix = (
            self._distance_matrix.copy() if self._distance_matrix is not None else None
        )
        new._valid_connections_cache = (
            list(self._valid_connections_cache)
            if self._valid_connections_cache is not None else None
        )
        return new

    def _generate_positions_from_graph(self):
        """Derive node positions from the graph structure."""
        positions = {}
        pos = nx.spring_layout(self.graph, scale=500)
        for node, (x, y) in pos.items():
            positions[node] = (x + 500, y + 500)
        return positions
    
    def _estimate_comm_distance(self):
        """Estimate the communication range."""
        if not self.positions or len(self.positions) < 2:
            return 200
        
        distances = []
        for edge in self.graph.edges():
            node1, node2 = edge
            if node1 in self.positions and node2 in self.positions:
                pos1 = self.positions[node1]
                pos2 = self.positions[node2]
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                distances.append(dist)
        
        if distances:
            return max(distances) * 1.2
        return 200
        
    def generate_scale_free_topology(self):
        """Generate a scale-free topology."""
        logger.info(f"generating a scale-free graph on {self.num_nodes} nodes")
        
        for i in range(self.num_nodes):
            self.positions[i] = (
                random.uniform(0, self.area_size),
                random.uniform(0, self.area_size)
            )
        
        self._compute_distance_matrix()
        
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))
        
        initial_nodes = min(self.density, self.num_nodes)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                if self._is_within_distance(i, j):
                    self.graph.add_edge(i, j)
        
        for node in tqdm(range(initial_nodes, self.num_nodes), desc="building scale-free graph"):
            existing_nodes = list(range(node))
            if not existing_nodes:
                continue
                
            degrees = [self.graph.degree(n) + 1 for n in existing_nodes]
            probabilities = np.array(degrees, dtype=np.float64)
            probabilities = probabilities / probabilities.sum()
            
            connections_made = 0
            attempts = 0
            max_attempts = min(50, len(existing_nodes) * 2)
            
            while connections_made < self.density and attempts < max_attempts:
                target = np.random.choice(existing_nodes, p=probabilities)
                if not self.graph.has_edge(node, target) and self._is_within_distance(node, target):
                    self.graph.add_edge(node, target)
                    connections_made += 1
                attempts += 1
        
        self._update_valid_connections_cache()
        logger.info(f"graph built, {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _compute_distance_matrix(self):
        """Precompute the distance matrix."""
        self._distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                pos1 = self.positions[i]
                pos2 = self.positions[j]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                self._distance_matrix[i, j] = distance
                self._distance_matrix[j, i] = distance
    
    def _is_within_distance(self, node1, node2):
        """Whether two nodes lie within the communication range."""
        return self._distance_matrix[node1, node2] <= self.comm_distance
    
    def _update_valid_connections_cache(self):
        """Rebuild the cache of valid connections."""
        self._valid_connections_cache = []
        edges = list(self.graph.edges())
        max_connections = 5000
         
        for _ in range(min(max_connections, len(edges)**2)):
            edge1, edge2 = random.sample(edges, 2)
            if len(set(edge1 + edge2)) == 4:
                self._valid_connections_cache.append((edge1, edge2))
        
        logger.info(f"valid connections: {len(self._valid_connections_cache)}")
    
    def get_valid_connections(self):
        """Valid connections, read from the cache."""
        if self._valid_connections_cache is None:
            self._update_valid_connections_cache()
        return self._valid_connections_cache
    
    def get_adjacency_matrix(self):
        """Adjacency matrix."""
        return nx.adjacency_matrix(self.graph).todense()
    
    def get_state_vector(self):
        """State vector of the topology."""
        n = self.num_nodes
        if _FAST:
            lo, hi = _edge_arrays(self.graph)
            state = 1.0 / self._distance_matrix[np.triu_indices(n, 1)]
            if lo.size:
                state[_triu_position(n, lo, hi)] = 1.0
            return state
        state = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) in self.graph.edges():
                    state.append(1.0)
                else:
                    state.append(1.0 / self._distance_matrix[i, j])
        return np.array(state)
    
    def _robustness_fast(self) -> float:
        """Degree-ordered removal curve in near-linear time."""
        n = self.num_nodes
        lo, hi = _edge_arrays(self.graph)

        alive = np.ones(n, dtype=bool)
        keep = np.ones(lo.size, dtype=bool)
        degree = np.bincount(np.concatenate([lo, hi]), minlength=n).astype(np.int64)
        order = np.empty(n, dtype=np.int64)
        edges_left = np.empty(n, dtype=np.int64)
        for step in range(n):
            edges_left[step] = keep.sum()
            score = np.where(alive, degree, -1)
            victim = int(np.argmax(score))
            order[step] = victim
            alive[victim] = False
            incident = keep & ((lo == victim) | (hi == victim))
            if incident.any():
                degree[lo[incident]] -= 1
                degree[hi[incident]] -= 1
                keep &= ~incident

        neighbours = [[] for _ in range(n)]
        for u, v in zip(lo.tolist(), hi.tolist()):
            neighbours[u].append(v)
            neighbours[v].append(u)

        parent = list(range(n))
        size = [1] * n
        present = [False] * n
        giant = 0
        largest = np.zeros(n, dtype=np.int64)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for step in range(n - 1, -1, -1):
            node = int(order[step])
            present[node] = True
            if size[node] > giant:
                giant = size[node]
            for other in neighbours[node]:
                if not present[other]:
                    continue
                a, b = find(node), find(other)
                if a != b:
                    if size[a] < size[b]:
                        a, b = b, a
                    parent[b] = a
                    size[a] += size[b]
                    if size[a] > giant:
                        giant = size[a]
            survivors = n - step
            if edges_left[step] == 0 and survivors > 1:
                largest[step] = 0
            else:
                largest[step] = giant

        total = 0.0
        for step in range(n):
            total += largest[step] / n
        return total / (n + 1)

    def _labels_ok(self) -> bool:
        """True when nodes are ``0..N-1`` in insertion order."""
        cached = getattr(self, "_labels_ok_cache", None)
        if cached is None:
            cached = list(self.graph.nodes()) == list(range(self.num_nodes))
            self._labels_ok_cache = cached
        return cached

    def _rollback_graph(self, edges_backup):
        """Restore the graph from its backup."""
        self.graph.clear_edges()
        self.graph.add_edges_from(edges_backup)
    
    def edge_swap(self, edge_pair1, edge_pair2):
        """Swap edges while keeping the edge count fixed."""
        i, j = edge_pair1
        m, n = edge_pair2
        
        original_edge_count = self.graph.number_of_edges()
        original_node_count = self.graph.number_of_nodes()
        
        if not (self.graph.has_edge(i, j) and self.graph.has_edge(m, n)):
            return False
            
        if len(set([i, j, m, n])) != 4:
            return False
        
        swap_option1 = [(i, m), (j, n)]
        swap_option2 = [(i, n), (j, m)]
        
        def is_valid_swap_option(new_edges):
            """Whether a swap is legal."""
            for edge in new_edges:
                node1, node2 = edge
                if not self._is_within_distance(node1, node2):
                    return False
                if node1 == node2:
                    return False
                if self.graph.has_edge(node1, node2):
                    return False
            return True
        
        valid_options = []
        if is_valid_swap_option(swap_option1):
            valid_options.append(swap_option1)
        if is_valid_swap_option(swap_option2):
            valid_options.append(swap_option2)
        
        if not valid_options:
            return False
        
        selected_option = random.choice(valid_options)
        
        try:
            edges_backup = list(self.graph.edges())
            
            self.graph.remove_edge(i, j)
            self.graph.remove_edge(m, n)
            
            for edge in selected_option:
                node1, node2 = edge
                self.graph.add_edge(node1, node2)
            
            final_edge_count = self.graph.number_of_edges()
            final_node_count = self.graph.number_of_nodes()
            
            if final_edge_count != original_edge_count:
                logger.error(f"edge swap failed: edge count {original_edge_count} -> {final_edge_count}")
                self._rollback_graph(edges_backup)
                return False
                
            if final_node_count != original_node_count:
                logger.error(f"edge swap failed: node count {original_node_count} -> {final_node_count}")
                self._rollback_graph(edges_backup)
                return False
            
            if not self._verify_graph_properties():
                logger.error("edge swap failed: the graph invariants do not hold")
                self._rollback_graph(edges_backup)
                return False
            
            self._valid_connections_cache = None
            
            return True
            
        except Exception as e:
            logger.error(f"exception during the edge swap: {e}")
            try:
                self._rollback_graph(edges_backup)
            except:
                logger.critical("rollback failed, the graph may be inconsistent")
            return False

    def _verify_graph_properties(self):
        """Check the basic graph invariants."""
        try:
            if self.graph.number_of_nodes() != self.num_nodes:
                return False
            
            if nx.number_of_selfloops(self.graph) > 0:
                return False
            
            expected_nodes = set(range(self.num_nodes))
            actual_nodes = set(self.graph.nodes())
            if expected_nodes != actual_nodes:
                return False
            
            edges = list(self.graph.edges())
            edge_set = set()
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                if sorted_edge in edge_set:
                    return False
                edge_set.add(sorted_edge)
            
            return True
        
        except Exception as e:
            logger.error(f"exception while checking the invariants: {e}")
            return False
    
    def calculate_robustness_RG(self):
        """Robustness RG under the greedy adaptive attack."""
        if _FAST:
            return self._robustness_fast()
        graph_copy = self.graph.copy()
        total_robustness = 0
        removed_nodes = 0
        
        while graph_copy.number_of_nodes() > 0:
            if graph_copy.number_of_edges() > 0:
                components = list(nx.connected_components(graph_copy))
                max_component_size = max(len(c) for c in components) if components else 0
            else:
                max_component_size = 1 if graph_copy.number_of_nodes() == 1 else 0
            
            total_robustness += max_component_size / self.num_nodes
            
            if graph_copy.number_of_nodes() > 0:
                degrees = dict(graph_copy.degree())
                if degrees:
                    highest_degree_node = max(degrees, key=degrees.get)
                    graph_copy.remove_node(highest_degree_node)
                    removed_nodes += 1
                else:
                    break
        
        return total_robustness / (self.num_nodes + 1)

class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        if batch_size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states))
    
    def size(self):
        return len(self.buffer)

class ActorNetwork(nn.Module):
    """Actor (policy) network."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class CriticNetwork(nn.Module):
    """Critic network."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ValueNetwork(nn.Module):
    """Value network."""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class SACAgent:
    """SAC agent."""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        self.target_value = ValueNetwork(state_dim).to(self.device)
        
        self.hard_update(self.target_value, self.value)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer()
        
    def hard_update(self, target, source):
        """Hard target update."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source, tau):
        """Soft target update."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        
    def select_action(self, state):
        """Add exploration noise."""
        try:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _ = self.actor.sample(state)
            
            action_np = action.cpu().data.numpy().flatten()
            noise = np.random.normal(0, 0.2, size=action_np.shape)
            noisy_action = action_np + noise
            return np.clip(noisy_action, -1, 1)
        except Exception as e:
            logger.warning(f"Action selection error: {e}, returning random action")
            return np.random.uniform(-1, 1, size=(2,))
    
    def update(self, batch_size=64, gamma=0.99, tau=0.005):
        """Update the networks."""
        if self.replay_buffer.size() < batch_size:
            return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
        
        try:
            states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)
            
            if len(states) == 0:
                return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            
            if states.shape[0] == 0:
                return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
            
            with torch.no_grad():
                next_state_actions, next_state_log_pi = self.actor.sample(next_states)
                target_q1 = self.critic1(next_states, next_state_actions)
                target_q2 = self.critic2(next_states, next_state_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_state_log_pi
                target_value = rewards + gamma * target_q
            
            current_value = self.value(states)
            value_loss = F.mse_loss(current_value, target_value)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
            self.value_optimizer.step()
            
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            critic1_loss = F.mse_loss(current_q1, target_value)
            critic2_loss = F.mse_loss(current_q2, target_value)
            
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
            self.critic2_optimizer.step()
            
            state_actions, log_pi = self.actor.sample(states)
            q1_new = self.critic1(states, state_actions)
            q2_new = self.critic2(states, state_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_pi - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            self.soft_update(self.target_value, self.value, tau)
            
            return {
                "actor_loss": actor_loss.item(),
                "critic_loss": (critic1_loss + critic2_loss).item(),
                "value_loss": value_loss.item()
            }
            
        except Exception as e:
            logger.warning(f"Update error: {e}")
            return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}

class ROHEMOptimizer:
    """ROMEN optimiser."""
    
    def __init__(self, population_size=3, generations=50, verbose=True):
        """Initialise the optimiser."""
        self.population_size = population_size
        self.generations = generations
        self.verbose = verbose
        
        self.crossover_prob = 0.8
        self.mutation_prob = 0.9
        self.super_mutation_prob = 0.05
        self.reset_mutation_prob = 0.1
        
        self.robustness_history = []
        self.fitness_history = []
        self.loss_history = []
        
        self.interaction_times = []
        self.update_times = []
    
    def optimize_topology(self, initial_graph, positions=None, area_size=1000, 
                         comm_distance=None, max_time=None):
        """Optimise the topology; the entry point used by the wrapper."""
        start_time = time.time()
        
        if self.verbose:
            logger.info("=== ROMEN topology optimisation ===")
            logger.info(f"input graph: {initial_graph.number_of_nodes()} nodes, {initial_graph.number_of_edges()} edges")
        
        try:
            topology = NetworkTopology(
                graph=initial_graph,
                positions=positions,
                area_size=area_size,
                comm_distance=comm_distance
            )
            
            topology._compute_distance_matrix()
            
            initial_robustness = topology.calculate_robustness_RG()
            if self.verbose:
                logger.info(f"initial robustness: {initial_robustness:.4f}")
            
            state_dim = topology.num_nodes * (topology.num_nodes - 1) // 2
            action_dim = 2
            
            if self.verbose:
                logger.info(f"state dimension {state_dim}, action dimension {action_dim}")
            
            master_agent = SACAgent(state_dim, action_dim)
            population = []
            for i in range(self.population_size):
                agent = SACAgent(state_dim, action_dim)
                population.append(agent)
            
            best_robustness = initial_robustness
            best_topology = copy.deepcopy(topology)
            
            pbar = tqdm(range(self.generations), desc="optimising") if self.verbose else range(self.generations)
            
            for generation in pbar:
                generation_start_time = time.time()
                
                if max_time and (time.time() - start_time) > max_time:
                    if self.verbose:
                        logger.info(f"time limit {max_time}s reached, stopping early")
                    break
                
                try:
                    all_experiences = []
                    fitness_scores = []
                    total_successful_swaps = 0
                    
                    master_experiences, master_fitness, master_topology, master_swaps = \
                        self._topology_interaction(master_agent, topology, steps=15)
                    all_experiences.extend(master_experiences)
                    total_successful_swaps += master_swaps
                    
                    for i, agent in enumerate(population):
                        experiences, fitness, agent_topology, swaps = \
                            self._topology_interaction(agent, topology, steps=10)
                        all_experiences.extend(experiences)
                        fitness_scores.append((i, fitness, agent_topology))
                        total_successful_swaps += swaps
                        
                        for exp in experiences[-5:]:
                            agent.replay_buffer.push(*exp)
                    
                    for exp in all_experiences[-20:]:
                        master_agent.replay_buffer.push(*exp)
                    
                    if generation % 30 == 0 and len(fitness_scores) >= 2:
                        fitness_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        if random.random() < self.crossover_prob:
                            parent1_idx, parent2_idx = fitness_scores[0][0], fitness_scores[1][0]
                            parent1 = population[parent1_idx]
                            parent2 = population[parent2_idx]
                            
                            offspring = self._distillation_crossover(parent1, parent2)
                            worst_idx = fitness_scores[-1][0]
                            population[worst_idx] = offspring
                        
                        if random.random() < 0.2:
                            mutation_idx = random.randint(0, len(population) - 1)
                            population[mutation_idx] = self._gaussian_mutation(population[mutation_idx])
                    
                    if generation % 3 == 0 and master_agent.replay_buffer.size() > 16:
                        losses = master_agent.update(batch_size=16)
                        self.loss_history.append(losses)
                    
                    if generation % 10 == 0:
                        for agent in population:
                            if agent.replay_buffer.size() > 8:
                                agent.update(batch_size=8)
                    
                    current_best_fitness = -float('inf')
                    current_best_topology = None
                    
                    for _, fitness, topology_candidate in fitness_scores:
                        if fitness > current_best_fitness:
                            current_best_fitness = fitness
                            current_best_topology = topology_candidate
                    
                    if current_best_topology:
                        try:
                            current_robustness = current_best_topology.calculate_robustness_RG()
                            if current_robustness > best_robustness:
                                best_robustness = current_robustness
                                best_topology = copy.deepcopy(current_best_topology)
                                topology = copy.deepcopy(current_best_topology)
                        except Exception as e:
                            logger.warning(f"Robustness evaluation error: {e}")
                    
                    self.robustness_history.append(best_robustness)
                    if fitness_scores:
                        avg_fitness = sum(score[1] for score in fitness_scores) / len(fitness_scores)
                        self.fitness_history.append(avg_fitness)
                    
                    if self.verbose and hasattr(pbar, 'set_postfix'):
                        generation_time = time.time() - generation_start_time
                        improvement = ((best_robustness - initial_robustness) / initial_robustness * 100)
                        
                        pbar.set_postfix({
                            'Robustness': f'{best_robustness:.4f}',
                            'Improvement': f'{improvement:.1f}%',
                            'Swaps': total_successful_swaps,
                            'Time': f'{generation_time:.1f}s'
                        })
                
                except Exception as e:
                    logger.error(f"Generation {generation} error: {e}")
                    continue
            
            if hasattr(pbar, 'close'):
                pbar.close()
            
            optimization_time = time.time() - start_time
            final_improvement = ((best_robustness - initial_robustness) / initial_robustness * 100)
            
            if self.verbose:
                logger.info(f"done: robustness {best_robustness:.4f}, improvement {final_improvement:.2f}%")
                logger.info(f"elapsed {optimization_time:.2f}s")
            
            result = {
                'optimized_graph': best_topology.graph,
                'positions': best_topology.positions,
                'initial_robustness': initial_robustness,
                'final_robustness': best_robustness,
                'improvement': final_improvement,
                'optimization_time': optimization_time,
                'robustness_history': self.robustness_history,
                'fitness_history': self.fitness_history,
                'parameters': {
                    'population_size': self.population_size,
                    'generations': self.generations,
                    'area_size': area_size,
                    'comm_distance': topology.comm_distance
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"optimisation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'optimized_graph': initial_graph,
                'positions': positions,
                'initial_robustness': 0,
                'final_robustness': 0,
                'improvement': 0,
                'optimization_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _topology_interaction(self, agent, topology, steps=20):
        """Apply one topology move."""
        start_time = time.time()
        
        current_topology = copy.deepcopy(topology)
        fitness = 0
        experiences = []
        successful_swaps = 0
        
        try:
            current_robustness = current_topology.calculate_robustness_RG()
        except Exception as e:
            logger.warning(f"Robustness calculation error: {e}")
            current_robustness = 0.1
        
        for step in range(steps):
            try:
                current_state = current_topology.get_state_vector()
                action = agent.select_action(current_state)
                
                edge1, edge2 = self._map_action_to_edges(action, current_topology)
                if edge1 is None or edge2 is None:
                    continue
                
                topology_copy = copy.deepcopy(current_topology)
                swap_success = topology_copy.edge_swap(edge1, edge2)
                
                if not swap_success:
                    experiences.append((current_state, action, -1, current_state))
                    fitness -= 1
                    continue
                
                next_state = topology_copy.get_state_vector()
                next_robustness = topology_copy.calculate_robustness_RG()
                
                if next_robustness > current_robustness:
                    reward = 10
                    accept = True
                elif next_robustness == current_robustness:
                    reward = 0
                    accept = (random.random() < 0.3)
                else:
                    reward = -1
                    accept = (random.random() < 0.1)
                
                if swap_success and accept:
                    experiences.append((current_state, action, reward, next_state))
                    current_topology = topology_copy
                    current_robustness = next_robustness
                    successful_swaps += 1
                else:
                    experiences.append((current_state, action, -1, current_state))
                
                fitness += reward
                
            except Exception as e:
                logger.warning(f"Interaction step error: {e}")
                continue
        
        interaction_time = time.time() - start_time
        self.interaction_times.append(interaction_time)
        
        return experiences, fitness, current_topology, successful_swaps
    
    def _map_action_to_edges(self, action, topology):
        """Map one action to a pair of edges."""
        try:
            valid_connections = topology.get_valid_connections()
            if not valid_connections:
                return None, None
            
            action = np.array(action).flatten()
            action_normalized = (action + 1) / 2
            index1 = int(action_normalized[0] * len(valid_connections))
            index2 = int(action_normalized[1] * len(valid_connections))
            
            index1 = np.clip(index1, 0, len(valid_connections) - 1)
            index2 = np.clip(index2, 0, len(valid_connections) - 1)
            
            if index1 == index2:
                index2 = (index2 + 1) % len(valid_connections)
            
            connection_pair = valid_connections[index1]
            
            return connection_pair
        except Exception as e:
            logger.warning(f"Action mapping error: {e}")
            return None, None
    
    def _distillation_crossover(self, parent1, parent2):
        """Distillation crossover (Eq. 10 of the paper)."""
        try:
            state_dim = parent1.actor.fc1.in_features
            action_dim = parent1.actor.mean.out_features
            offspring = SACAgent(state_dim, action_dim)
            
            offspring.actor.load_state_dict(copy.deepcopy(parent1.actor.state_dict()))
            
            mixed_buffer = ReplayBuffer()
            parent1_samples = min(30, parent1.replay_buffer.size())
            parent2_samples = min(30, parent2.replay_buffer.size())
            
            for agent, sample_size in [(parent1, parent1_samples), (parent2, parent2_samples)]:
                states, actions, rewards, next_states = agent.replay_buffer.sample(sample_size)
                
                states_tensor = torch.FloatTensor(states).to(offspring.device)
                actions_tensor = torch.FloatTensor(actions).to(offspring.device)
                
                with torch.no_grad():
                    q1_values = agent.critic1(states_tensor, actions_tensor)
                    q2_values = agent.critic2(states_tensor, actions_tensor)
                    q_values = torch.min(q1_values, q2_values)
                
                advantage_threshold = torch.quantile(q_values, 0.7)
                high_value_mask = q_values > advantage_threshold
                
                for i in range(len(states)):
                    if high_value_mask[i]:
                        mixed_buffer.push(states[i], actions[i], rewards[i], next_states[i])
            
            distillation_epochs = 10
            if mixed_buffer.size() > 8:
                offspring.replay_buffer = mixed_buffer
                
                for _ in range(distillation_epochs):
                    states, actions, rewards, next_states = mixed_buffer.sample(8)
                    states_tensor = torch.FloatTensor(states).to(offspring.device)
                    
                    losses = []
                    for parent in [parent1, parent2]:
                        with torch.no_grad():
                            parent_action, _ = parent.actor.sample(states_tensor)
                        
                        offspring_action, _ = offspring.actor.sample(states_tensor)
                        
                        loss = F.mse_loss(offspring_action, parent_action)
                        
                        with torch.no_grad():
                            parent_q1 = parent.critic1(states_tensor, parent_action)
                            parent_q2 = parent.critic2(states_tensor, parent_action)
                            parent_q = torch.min(parent_q1, parent_q2)
                        
                        if parent_q.mean() > offspring.value(states_tensor).mean():
                            loss.backward()
                            losses.append(loss.item())
                            offspring.actor_optimizer.step()
                            offspring.actor_optimizer.zero_grad()
                    
                    if not losses:
                        break
            
            return offspring
            
        except Exception as e:
            logger.warning(f"distillation crossover failed: {e}, falling back to agent 1")
            state_dim = parent1.actor.fc1.in_features
            action_dim = parent1.actor.mean.out_features
            offspring = SACAgent(state_dim, action_dim)
            offspring.actor.load_state_dict(copy.deepcopy(parent1.actor.state_dict()))
            return offspring
    
    def _gaussian_mutation(self, agent):
        """Gaussian mutation."""
        try:
            state_dim = agent.actor.fc1.in_features
            action_dim = agent.actor.mean.out_features
            mutated_agent = SACAgent(state_dim, action_dim)
            
            mutated_agent.actor.load_state_dict(copy.deepcopy(agent.actor.state_dict()))
            mutated_agent.critic1.load_state_dict(copy.deepcopy(agent.critic1.state_dict()))
            mutated_agent.critic2.load_state_dict(copy.deepcopy(agent.critic2.state_dict()))
            mutated_agent.value.load_state_dict(copy.deepcopy(agent.value.state_dict()))
            mutated_agent.hard_update(mutated_agent.target_value, mutated_agent.value)
            
            mutated_agent.replay_buffer = copy.deepcopy(agent.replay_buffer)
            
            with torch.no_grad():
                for param in mutated_agent.actor.parameters():
                    if len(param.shape) > 1:
                        mask = torch.rand_like(param) < 0.03
                        noise = torch.randn_like(param) * 0.05
                        param.data = param.data + noise * mask.float()
            
            return mutated_agent
            
        except Exception as e:
            logger.warning(f"Mutation error: {e}, returning copy of original agent")
            state_dim = agent.actor.fc1.in_features
            action_dim = agent.actor.mean.out_features
            mutated_agent = SACAgent(state_dim, action_dim)
            mutated_agent.actor.load_state_dict(copy.deepcopy(agent.actor.state_dict()))
            return mutated_agent

def optimize_network_robustness(initial_graph, positions=None, population_size=3, 
                               generations=50, area_size=1000, comm_distance=None,
                               max_time=None, verbose=True):
    """Convenience wrapper around the optimiser."""
    optimizer = ROHEMOptimizer(
        population_size=population_size,
        generations=generations,
        verbose=verbose
    )
    
    return optimizer.optimize_topology(
        initial_graph=initial_graph,
        positions=positions,
        area_size=area_size,
        comm_distance=comm_distance,
        max_time=max_time
    )


 


    
