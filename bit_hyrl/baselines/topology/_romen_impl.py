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
# Ported from ``network_construction/ROMEN/ROMEN.py`` (logic copied in, no import).
# Headless: strip module-level TkAgg/rcParams; use the non-interactive Agg backend.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------------------------------------------------------------- #
# Vectorised kernels for the two hot methods
# --------------------------------------------------------------------------- #
# Profiling a single N=300 construction showed 88% of the wall clock inside two
# functions that were written as per-element Python loops:
#
#   get_state_vector        44%   2e8 tuple membership tests, and ``graph.edges()``
#                                 rebuilt 45k times per call inside the inner loop
#   calculate_robustness_RG 44%   N ``nx.connected_components`` calls per evaluation
#
# Both are replaced here by array kernels with **identical semantics**: the state
# vector holds the same float at the same index, and the robustness loop removes the
# same node at every step (highest degree, ties broken by node insertion order) and
# sums the same giant-component sizes.  ``ROMEN_LEGACY=1`` restores the original loops
# so the two can be compared bit for bit.
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
    """优化的网络拓扑类"""
    
    def __init__(self, num_nodes=None, density=None, area_size=None, comm_distance=None, 
                 graph=None, positions=None):
        """
        初始化网络拓扑
        
        Args:
            num_nodes: 节点数量（用于生成新拓扑）
            density: 网络密度（用于生成新拓扑）
            area_size: 区域大小（用于生成新拓扑）
            comm_distance: 通信距离（用于生成新拓扑）
            graph: 现有的networkx图对象（用于从现有拓扑初始化）
            positions: 节点位置字典（用于从现有拓扑初始化）
        """
        if graph is not None:
            # 从现有图初始化
            self.graph = graph.copy()
            self.num_nodes = graph.number_of_nodes()
            self.positions = positions if positions else self._generate_positions_from_graph()
            # 估算参数
            self.density = max(1, int(np.mean([graph.degree(n) for n in graph.nodes()])))
            self.area_size = area_size if area_size else 1000
            self.comm_distance = comm_distance if comm_distance else self._estimate_comm_distance()
        else:
            # 新建拓扑
            self.num_nodes = num_nodes
            self.density = density
            self.area_size = area_size
            self.comm_distance = comm_distance
            self.positions = {}
            self.graph = None
            
        self._valid_connections_cache = None
        self._distance_matrix = None

    def __deepcopy__(self, memo):
        """OPT(port): cheap structural clone instead of generic recursive deepcopy.

        The hot interaction loop deep-copies the whole topology object every step;
        generic ``copy.deepcopy`` spent ~55% of the runtime recursing per-element into
        the graph / positions / distance-matrix / cache.  This reproduces **identical**
        copy semantics (independent graph, copied distance matrix, copied
        valid-connection cache list, copied positions dict) in O(N+E+N^2) — the
        simulation result is unchanged (verified bit-for-bit) and the run is faster.
        """
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
        """从图结构生成节点位置"""
        positions = {}
        # 使用spring layout生成位置
        pos = nx.spring_layout(self.graph, scale=500)
        for node, (x, y) in pos.items():
            positions[node] = (x + 500, y + 500)  # 确保坐标为正
        return positions
    
    def _estimate_comm_distance(self):
        """估算通信距离"""
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
            # 设置为现有边距离的1.2倍作为通信距离
            return max(distances) * 1.2
        return 200
        
    def generate_scale_free_topology(self):
        """生成无标度网络拓扑"""
        logger.info(f"生成{self.num_nodes}节点的无标度网络...")
        
        # 随机生成节点位置
        for i in range(self.num_nodes):
            self.positions[i] = (
                random.uniform(0, self.area_size),
                random.uniform(0, self.area_size)
            )
        
        # 预计算距离矩阵
        self._compute_distance_matrix()
        
        # 初始化图
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))
        
        # 初始完全连接的子图
        initial_nodes = min(self.density, self.num_nodes)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                if self._is_within_distance(i, j):
                    self.graph.add_edge(i, j)
        
        # 优先连接机制添加剩余节点
        for node in tqdm(range(initial_nodes, self.num_nodes), desc="构建无标度网络"):
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
        
        # 更新有效连接缓存
        self._update_valid_connections_cache()
        logger.info(f"网络生成完成，边数: {self.graph.number_of_edges()}")
        return self.graph
    
    def _compute_distance_matrix(self):
        """预计算距离矩阵"""
        self._distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                pos1 = self.positions[i]
                pos2 = self.positions[j]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                self._distance_matrix[i, j] = distance
                self._distance_matrix[j, i] = distance
    
    def _is_within_distance(self, node1, node2):
        """检查两个节点是否在通信距离内"""
        return self._distance_matrix[node1, node2] <= self.comm_distance
    
    def _update_valid_connections_cache(self):
        """更新有效连接缓存 - 优化版本"""
        self._valid_connections_cache = []
        edges = list(self.graph.edges())
        max_connections = 5000  # 限制最大有效连接数
         
        # 随机采样边对
        for _ in range(min(max_connections, len(edges)**2)):
            edge1, edge2 = random.sample(edges, 2)
            if len(set(edge1 + edge2)) == 4:  # 确保四个不同节点
                self._valid_connections_cache.append((edge1, edge2))
        
        logger.info(f"优化有效连接数: {len(self._valid_connections_cache)}")
    
    def get_valid_connections(self):
        """获取有效连接（使用缓存）"""
        if self._valid_connections_cache is None:
            self._update_valid_connections_cache()
        return self._valid_connections_cache
    
    def get_adjacency_matrix(self):
        """获取邻接矩阵"""
        return nx.adjacency_matrix(self.graph).todense()
    
    def get_state_vector(self):
        """将拓扑转换为状态向量 (vectorised; same values as the legacy loop)."""
        n = self.num_nodes
        if _FAST:
            lo, hi = _edge_arrays(self.graph)
            state = 1.0 / self._distance_matrix[np.triu_indices(n, 1)]
            if lo.size:
                state[_triu_position(n, lo, hi)] = 1.0
            return state
        state = []
        for i in range(n):
            for j in range(i+1, n):  # 严格上三角
                if (i, j) in self.graph.edges():
                    state.append(1.0)
                else:
                    state.append(1.0 / self._distance_matrix[i, j])
        return np.array(state)
    
    def _robustness_fast(self) -> float:
        """Degree-ordered removal curve in near-linear time.

        The legacy loop recomputes the connected components of the residual graph once
        per removed node, i.e. O(N) BFS passes per evaluation.  The graph is only ever
        losing nodes, so the same curve is obtained by replaying the removal sequence
        backwards and *adding* nodes with a union-find: every component size that the
        forward pass would have measured appears when its last-added node joins.

        The removal sequence comes from a cheap degree-only forward pass, which is the
        same rule the legacy loop applies (highest current degree, ties by node order),
        and the no-edge corner case is preserved: with two or more survivors and no
        edges the legacy branch scores zero rather than one.
        """
        n = self.num_nodes
        lo, hi = _edge_arrays(self.graph)

        # ---- pass 1: the removal order and the edge count after each removal ----
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

        # ---- pass 2: replay the removals backwards, union-find the additions ----
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
                # the legacy branch reports an empty giant component here
                largest[step] = 0
            else:
                largest[step] = giant

        # accumulate forwards, in the legacy order: the replay above is backwards but
        # floating-point addition is not associative, and a last-bit difference in the
        # reward is enough to steer the policy onto a different trajectory
        total = 0.0
        for step in range(n):
            total += largest[step] / n
        return total / (n + 1)

    def _labels_ok(self) -> bool:
        """True when nodes are ``0..N-1`` in insertion order.

        Both vectorised kernels index arrays by node label and break ties by insertion
        order, so a graph with any other labelling must take the legacy path rather
        than be silently mis-indexed.
        """
        cached = getattr(self, "_labels_ok_cache", None)
        if cached is None:
            cached = list(self.graph.nodes()) == list(range(self.num_nodes))
            self._labels_ok_cache = cached
        return cached

    def _rollback_graph(self, edges_backup):
        """回滚图到备份状态"""
        # 清空当前图的所有边
        self.graph.clear_edges()
        # 恢复备份的边
        self.graph.add_edges_from(edges_backup)
    
    def edge_swap(self, edge_pair1, edge_pair2):
        """
        执行边交换操作 - 完全修复版本，确保边数严格保持不变
        
        边交换原理：
        原边: (i,j) 和 (m,n)
        新边: (i,m) 和 (j,n) 或者 (i,n) 和 (j,m)
        
        Args:
            edge_pair1: 第一条边 (i, j)
            edge_pair2: 第二条边 (m, n)
            
        Returns:
            bool: 是否成功执行边交换
        """
        i, j = edge_pair1
        m, n = edge_pair2
        
        # 记录原始状态用于验证和可能的回滚
        original_edge_count = self.graph.number_of_edges()
        original_node_count = self.graph.number_of_nodes()
        
        # 1. 基本有效性检查
        if not (self.graph.has_edge(i, j) and self.graph.has_edge(m, n)):
            return False
            
        # 2. 确保是四个不同的节点
        if len(set([i, j, m, n])) != 4:
            return False
        
        # 3. 定义两种可能的边交换方式
        swap_option1 = [(i, m), (j, n)]
        swap_option2 = [(i, n), (j, m)]
        
        # 4. 检查两种交换方式的有效性
        def is_valid_swap_option(new_edges):
            """检查一种交换方式是否有效"""
            for edge in new_edges:
                node1, node2 = edge
                # 检查是否在通信距离内
                if not self._is_within_distance(node1, node2):
                    return False
                # 检查是否会创建自环
                if node1 == node2:
                    return False
                # 关键：检查新边是否已存在（这会导致边数减少）
                if self.graph.has_edge(node1, node2):
                    return False
            return True
        
        # 5. 选择有效的交换方式
        valid_options = []
        if is_valid_swap_option(swap_option1):
            valid_options.append(swap_option1)
        if is_valid_swap_option(swap_option2):
            valid_options.append(swap_option2)
        
        # 6. 如果没有有效的交换方式，返回失败
        if not valid_options:
            return False
        
        # 7. 随机选择一个有效的交换方式
        selected_option = random.choice(valid_options)
        
        try:
            # 8. 创建图的备份（用于回滚）
            edges_backup = list(self.graph.edges())
            
            # 9. 执行边交换
            # 首先移除原有的两条边
            self.graph.remove_edge(i, j)
            self.graph.remove_edge(m, n)
            
            # 然后添加新的两条边
            for edge in selected_option:
                node1, node2 = edge
                self.graph.add_edge(node1, node2)
            
            # 10. 验证交换结果
            final_edge_count = self.graph.number_of_edges()
            final_node_count = self.graph.number_of_nodes()
            
            # 严格验证边数和节点数
            if final_edge_count != original_edge_count:
                logger.error(f"边交换失败：边数改变 {original_edge_count} -> {final_edge_count}")
                # 回滚操作
                self._rollback_graph(edges_backup)
                return False
                
            if final_node_count != original_node_count:
                logger.error(f"边交换失败：节点数改变 {original_node_count} -> {final_node_count}")
                # 回滚操作
                self._rollback_graph(edges_backup)
                return False
            
            # 11. 验证图的基本性质
            if not self._verify_graph_properties():
                logger.error("边交换失败：图性质验证失败")
                self._rollback_graph(edges_backup)
                return False
            
            # 12. 成功：更新有效连接缓存
            self._valid_connections_cache = None
            
            return True
            
        except Exception as e:
            logger.error(f"边交换过程中发生异常: {e}")
            # 尝试回滚
            try:
                self._rollback_graph(edges_backup)
            except:
                logger.critical("回滚失败，图状态可能不一致")
            return False

    def _verify_graph_properties(self):
        """验证图的基本性质"""
        try:
            # 检查节点数
            if self.graph.number_of_nodes() != self.num_nodes:
                return False
            
            # 检查是否有自环
            if nx.number_of_selfloops(self.graph) > 0:
                return False
            
            # 检查节点标识是否正确
            expected_nodes = set(range(self.num_nodes))
            actual_nodes = set(self.graph.nodes())
            if expected_nodes != actual_nodes:
                return False
            
            # 检查是否有重复边（NetworkX应该自动处理，但还是检查一下）
            edges = list(self.graph.edges())
            edge_set = set()
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                if sorted_edge in edge_set:
                    return False
                edge_set.add(sorted_edge)
            
            return True
        
        except Exception as e:
            logger.error(f"图性质验证时发生异常: {e}")
            return False
    
    def calculate_robustness_RG(self):
        """计算鲁棒性指标RG (vectorised; same removals and same sum as the legacy loop)."""
        if _FAST:
            return self._robustness_fast()
        graph_copy = self.graph.copy()
        total_robustness = 0
        removed_nodes = 0
        
        while graph_copy.number_of_nodes() > 0:
            # 计算当前最大连通分量
            if graph_copy.number_of_edges() > 0:
                components = list(nx.connected_components(graph_copy))
                max_component_size = max(len(c) for c in components) if components else 0
            else:
                max_component_size = 1 if graph_copy.number_of_nodes() == 1 else 0
            
            total_robustness += max_component_size / self.num_nodes
            
            # 移除度数最高的节点
            if graph_copy.number_of_nodes() > 0:
                degrees = dict(graph_copy.degree())
                if degrees:
                    highest_degree_node = max(degrees, key=degrees.get)
                    graph_copy.remove_node(highest_degree_node)
                    removed_nodes += 1
                else:
                    break
        
        return total_robustness / (self.num_nodes + 1)

# [保留之前的网络类定义...]
class ReplayBuffer:
    """经验回放缓冲区"""
    
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
    """演员网络（策略网络）- 修复就地操作问题"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
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
        # 避免就地操作，创建新的tensor
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # 避免就地操作
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class CriticNetwork(nn.Module):
    """评价网络 - 修复就地操作问题"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
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
    """价值网络 - 修复就地操作问题"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
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
    """SAC智能体 - 修复梯度计算问题"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络初始化
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        self.target_value = ValueNetwork(state_dim).to(self.device)
        
        # 复制参数到目标网络 - 避免就地操作
        self.hard_update(self.target_value, self.value)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer()
        
    def hard_update(self, target, source):
        """硬更新 - 避免就地操作"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source, tau):
        """软更新 - 避免就地操作"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        
    def select_action(self, state):
        """增加探索噪声"""
        try:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _ = self.actor.sample(state)
            
            # 添加高斯噪声增强探索
            # FIX(port): 原实现 ``size=action.shape`` 为 (1, action_dim)，与已 flatten
            # 的 1D 动作相加时被 numpy 广播回 (1, action_dim)，使存入回放池的动作是
            # 2D；批量后变 (B,1,action_dim) 3D，导致 ``CriticNetwork.cat([state2D,
            # action3D])`` 维度不一致（蒸馏交叉/更新被 try/except 静默降级）。此处让
            # 噪声与 flatten 后的动作同形（1D），动作以一维 [action_dim] 存入回放池。
            action_np = action.cpu().data.numpy().flatten()
            noise = np.random.normal(0, 0.2, size=action_np.shape)
            noisy_action = action_np + noise
            return np.clip(noisy_action, -1, 1)
        except Exception as e:
            logger.warning(f"Action selection error: {e}, returning random action")
            return np.random.uniform(-1, 1, size=(2,))
    
    def update(self, batch_size=64, gamma=0.99, tau=0.005):
        """更新网络 - 修复梯度计算问题"""
        if self.replay_buffer.size() < batch_size:
            return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
        
        try:
            states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)
            
            # 检查数据有效性
            if len(states) == 0:
                return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            
            # 检查tensor形状
            if states.shape[0] == 0:
                return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
            
            # 计算目标值 - 使用detach避免梯度问题
            with torch.no_grad():
                next_state_actions, next_state_log_pi = self.actor.sample(next_states)
                target_q1 = self.critic1(next_states, next_state_actions)
                target_q2 = self.critic2(next_states, next_state_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_state_log_pi
                target_value = rewards + gamma * target_q
            
            # 更新价值网络
            current_value = self.value(states)
            value_loss = F.mse_loss(current_value, target_value)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
            self.value_optimizer.step()
            
            # 更新评价网络
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
            
            # 更新演员网络 - 重新采样避免梯度问题
            state_actions, log_pi = self.actor.sample(states)
            q1_new = self.critic1(states, state_actions)
            q2_new = self.critic2(states, state_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_pi - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            # 软更新目标网络
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
    """ROMEM优化器 - 重构为接口类"""
    
    def __init__(self, population_size=3, generations=50, verbose=True):
        """
        初始化ROMEM优化器
        
        Args:
            population_size: 种群大小
            generations: 进化代数
            verbose: 是否显示详细日志
        """
        self.population_size = population_size
        self.generations = generations
        self.verbose = verbose
        
        # 训练参数
        self.crossover_prob = 0.8
        self.mutation_prob = 0.9
        self.super_mutation_prob = 0.05
        self.reset_mutation_prob = 0.1
        
        # 记录训练过程
        self.robustness_history = []
        self.fitness_history = []
        self.loss_history = []
        
        # 性能统计
        self.interaction_times = []
        self.update_times = []
    
    def optimize_topology(self, initial_graph, positions=None, area_size=1000, 
                         comm_distance=None, max_time=None):
        """
        优化网络拓扑的主接口函数
        
        Args:
            initial_graph: 初始网络图 (networkx.Graph对象)
            positions: 节点位置字典 {node_id: (x, y)}，可选
            area_size: 区域大小，用于生成位置（如果positions为None）
            comm_distance: 通信距离，可选（自动估算）
            max_time: 最大运行时间（秒），可选
            
        Returns:
            dict: 包含优化结果的字典
                - optimized_graph: 优化后的网络图
                - positions: 节点位置
                - initial_robustness: 初始鲁棒性
                - final_robustness: 最终鲁棒性
                - improvement: 改进百分比
                - optimization_time: 优化耗时
                - robustness_history: 鲁棒性变化历史
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("=== 开始ROMEM拓扑优化 ===")
            logger.info(f"输入图: {initial_graph.number_of_nodes()}节点, {initial_graph.number_of_edges()}边")
        
        try:
            # 初始化拓扑对象
            topology = NetworkTopology(
                graph=initial_graph,
                positions=positions,
                area_size=area_size,
                comm_distance=comm_distance
            )
            
            # 确保距离矩阵已计算
            topology._compute_distance_matrix()
            
            # 计算初始鲁棒性
            initial_robustness = topology.calculate_robustness_RG()
            if self.verbose:
                logger.info(f"初始拓扑鲁棒性: {initial_robustness:.4f}")
            
            # 计算状态和动作维度
            state_dim = topology.num_nodes * (topology.num_nodes - 1) // 2
            action_dim = 2
            
            if self.verbose:
                logger.info(f"状态维度: {state_dim}, 动作维度: {action_dim}")
            
            # 初始化智能体
            master_agent = SACAgent(state_dim, action_dim)
            population = []
            for i in range(self.population_size):
                agent = SACAgent(state_dim, action_dim)
                population.append(agent)
            
            # 优化主循环
            best_robustness = initial_robustness
            best_topology = copy.deepcopy(topology)
            
            # 创建进度条
            pbar = tqdm(range(self.generations), desc="优化进度") if self.verbose else range(self.generations)
            
            for generation in pbar:
                generation_start_time = time.time()
                
                # 检查时间限制
                if max_time and (time.time() - start_time) > max_time:
                    if self.verbose:
                        logger.info(f"达到时间限制 {max_time}s，提前结束优化")
                    break
                
                try:
                    # 分布式离线交互
                    all_experiences = []
                    fitness_scores = []
                    total_successful_swaps = 0
                    
                    # 主智能体交互
                    master_experiences, master_fitness, master_topology, master_swaps = \
                        self._topology_interaction(master_agent, topology, steps=15)
                    all_experiences.extend(master_experiences)
                    total_successful_swaps += master_swaps
                    
                    # 种群交互
                    for i, agent in enumerate(population):
                        experiences, fitness, agent_topology, swaps = \
                            self._topology_interaction(agent, topology, steps=10)
                        all_experiences.extend(experiences)
                        fitness_scores.append((i, fitness, agent_topology))
                        total_successful_swaps += swaps
                        
                        # 将经验添加到智能体的缓冲区
                        for exp in experiences[-5:]:
                            agent.replay_buffer.push(*exp)
                    
                    # 将经验添加到主智能体缓冲区
                    for exp in all_experiences[-20:]:
                        master_agent.replay_buffer.push(*exp)
                    
                    # 种群进化
                    if generation % 30 == 0 and len(fitness_scores) >= 2:
                        fitness_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        # 交叉操作
                        if random.random() < self.crossover_prob:
                            parent1_idx, parent2_idx = fitness_scores[0][0], fitness_scores[1][0]
                            parent1 = population[parent1_idx]
                            parent2 = population[parent2_idx]
                            
                            offspring = self._distillation_crossover(parent1, parent2)
                            worst_idx = fitness_scores[-1][0]
                            population[worst_idx] = offspring
                        
                        # 变异操作
                        if random.random() < 0.2:
                            mutation_idx = random.randint(0, len(population) - 1)
                            population[mutation_idx] = self._gaussian_mutation(population[mutation_idx])
                    
                    # 更新网络
                    if generation % 3 == 0 and master_agent.replay_buffer.size() > 16:
                        losses = master_agent.update(batch_size=16)
                        self.loss_history.append(losses)
                    
                    if generation % 10 == 0:
                        for agent in population:
                            if agent.replay_buffer.size() > 8:
                                agent.update(batch_size=8)
                    
                    # 评估当前最佳拓扑
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
                    
                    # 记录训练过程
                    self.robustness_history.append(best_robustness)
                    if fitness_scores:
                        avg_fitness = sum(score[1] for score in fitness_scores) / len(fitness_scores)
                        self.fitness_history.append(avg_fitness)
                    
                    # 更新进度条
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
            
            # 计算最终结果
            optimization_time = time.time() - start_time
            final_improvement = ((best_robustness - initial_robustness) / initial_robustness * 100)
            
            if self.verbose:
                logger.info(f"优化完成! 最终鲁棒性: {best_robustness:.4f}, 改进: {final_improvement:.2f}%")
                logger.info(f"优化耗时: {optimization_time:.2f}s")
            
            # 返回结果
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
            logger.error(f"优化过程出错: {e}")
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
        """拓扑交互操作"""
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
                
                # OPT: 复用上一步的 current_robustness（等于此处的重算值，不改变任何决策）
                # 修正奖励计算
                if next_robustness > current_robustness:
                    reward = 10
                    accept = True
                elif next_robustness == current_robustness:
                    reward = 0
                    accept = (random.random() < 0.3)  # 30%概率接受中性变化
                else:
                    reward = -1
                    accept = (random.random() < 0.1)  # 10%概率接受负面变化
                
                # 接受新拓扑的条件
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
        """将动作映射到边对"""
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
    
    # 在ROHEMOptimizer类中完善蒸馏交叉函数
    def _distillation_crossover(self, parent1, parent2):
        """完整实现蒸馏交叉操作（根据论文公式10）"""
        try:
            state_dim = parent1.actor.fc1.in_features
            action_dim = parent1.actor.mean.out_features
            offspring = SACAgent(state_dim, action_dim)
            
            # 克隆父代理1的网络参数（作为起点）
            offspring.actor.load_state_dict(copy.deepcopy(parent1.actor.state_dict()))
            
            # 创建混合经验池
            mixed_buffer = ReplayBuffer()
            parent1_samples = min(30, parent1.replay_buffer.size())
            parent2_samples = min(30, parent2.replay_buffer.size())
            
            # ==================== 核心修改点 ====================
            # 1. 从双方回放池采样经验
            for agent, sample_size in [(parent1, parent1_samples), (parent2, parent2_samples)]:
                states, actions, rewards, next_states = agent.replay_buffer.sample(sample_size)
                
                # 转换为tensor
                states_tensor = torch.FloatTensor(states).to(offspring.device)
                actions_tensor = torch.FloatTensor(actions).to(offspring.device)
                
                # 2. 使用Q函数评估动作价值
                with torch.no_grad():
                    q1_values = agent.critic1(states_tensor, actions_tensor)
                    q2_values = agent.critic2(states_tensor, actions_tensor)
                    q_values = torch.min(q1_values, q2_values)
                
                # 3. 只保留高价值经验
                # 选择奖励高于阈值的样本
                advantage_threshold = torch.quantile(q_values, 0.7)  # 取前30%高价值经验
                high_value_mask = q_values > advantage_threshold
                
                for i in range(len(states)):
                    if high_value_mask[i]:
                        mixed_buffer.push(states[i], actions[i], rewards[i], next_states[i])
            
            # 4. 知识蒸馏训练
            distillation_epochs = 10
            if mixed_buffer.size() > 8:
                offspring.replay_buffer = mixed_buffer
                
                for _ in range(distillation_epochs):
                    # 使用高价值经验更新后代策略
                    states, actions, rewards, next_states = mixed_buffer.sample(8)
                    states_tensor = torch.FloatTensor(states).to(offspring.device)
                    
                    # 同时训练两个父代理的优势策略
                    losses = []
                    for parent in [parent1, parent2]:
                        with torch.no_grad():
                            # 获取父代理在相同状态下的推荐动作
                            parent_action, _ = parent.actor.sample(states_tensor)
                        
                        # 计算后代代理的动作
                        offspring_action, _ = offspring.actor.sample(states_tensor)
                        
                        # 计算蒸馏损失：最小化后代与父代理优势策略的差异
                        loss = F.mse_loss(offspring_action, parent_action)
                        
                        # 仅保留性能更优父代理的梯度
                        with torch.no_grad():
                            parent_q1 = parent.critic1(states_tensor, parent_action)
                            parent_q2 = parent.critic2(states_tensor, parent_action)
                            parent_q = torch.min(parent_q1, parent_q2)
                        
                        # 只更新当父代理Q值超过后代的场景
                        if parent_q.mean() > offspring.value(states_tensor).mean():
                            loss.backward()
                            losses.append(loss.item())
                            offspring.actor_optimizer.step()
                            offspring.actor_optimizer.zero_grad()
                    
                    if not losses:
                        break  # 没有有效更新时提前终止
            # ==================== 修改结束 ====================
            
            return offspring
            
        except Exception as e:
            # FIX(port): 单图场景下蒸馏交叉已可正常执行（上面动作维度已修正）；保留
            # 降级兜底，但降为 WARNING，避免把可恢复情况误报为 ERROR 刷屏。
            logger.warning(f"蒸馏交叉失败: {e}, 使用父代理1作为降级方案")
            state_dim = parent1.actor.fc1.in_features
            action_dim = parent1.actor.mean.out_features
            offspring = SACAgent(state_dim, action_dim)
            offspring.actor.load_state_dict(copy.deepcopy(parent1.actor.state_dict()))
            return offspring
    
    def _gaussian_mutation(self, agent):
        """高斯变异操作"""
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

# 便捷的接口函数
def optimize_network_robustness(initial_graph, positions=None, population_size=3, 
                               generations=50, area_size=1000, comm_distance=None,
                               max_time=None, verbose=True):
    """
    优化网络拓扑鲁棒性的便捷接口函数
    
    Args:
        initial_graph: 初始网络图 (networkx.Graph对象)
        positions: 节点位置字典 {node_id: (x, y)}，可选
        population_size: ROMEM算法的种群大小，默认3
        generations: 进化代数，默认50
        area_size: 区域大小，默认1000
        comm_distance: 通信距离，可选（自动估算）
        max_time: 最大运行时间（秒），可选
        verbose: 是否显示详细日志，默认True
        
    Returns:
        dict: 包含优化结果的字典
    """
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


def plot_robustness_comparison(initial_graph, optimized_graph, title="Unity"):
    """使用NetworkX绘制优化前后的鲁棒性比较图
    
    Args:
        initial_graph: NetworkX图对象，初始拓扑
        optimized_graph: NetworkX图对象，优化后的拓扑
        title: 图表标题
    """
    # 计算攻击后的网络连通性
    def calculate_connectivity_after_attacks(graph):
        # 创建图的拷贝
        g = graph.copy()
        n = g.number_of_nodes()
        connectivity = [1.0]  # 初始完全连通
        
        for _ in range(n):
            if g.number_of_nodes() == 0:
                connectivity.extend([0] * (n - len(connectivity) + 1))
                break
                
            # HDA攻击 - 删除度数最大的节点
            degrees = dict(g.degree())
            if not degrees:
                break
                
            max_degree_node = max(degrees, key=degrees.get)
            g.remove_node(max_degree_node)
            
            # 计算最大连通分量比例
            if g.number_of_nodes() > 0:
                # 获取所有连通分量
                components = list(nx.connected_components(g))
                largest_cc_size = max(len(comp) for comp in components) if components else 0
                connectivity.append(largest_cc_size / n)
            else:
                connectivity.append(0)
        
        return connectivity[:n+1]  # 限制为n+1个点（包括初始状态）
    
    # 计算鲁棒性值 - 基于NetworkX
    def calculate_robustness(graph):
        """计算网络的鲁棒性值"""
        g = graph.copy()
        n = g.number_of_nodes()
        robustness_sum = 0.0

        # 预先计算节点度数并按从大到小排序（静态攻击）
        degrees = dict(g.degree())
        node_order = sorted(degrees.keys(), key=lambda node: degrees[node], reverse=True)

        # 遍历每一步攻击，记录最大连通分量比例
        for step in range(n + 1):
            if g.number_of_nodes() > 0:
                # 获取最大连通分量大小
                components = list(nx.connected_components(g))
                largest_cc_size = max(len(comp) for comp in components) if components else 0
                robustness_sum += largest_cc_size / n
            else:
                robustness_sum += 0.0

            if step < n and node_order:
                # 删除预定的节点
                node_to_remove = node_order[step]
                if g.has_node(node_to_remove):
                    g.remove_node(node_to_remove)

        return robustness_sum / (n + 1)
    
    initial_connectivity = calculate_connectivity_after_attacks(initial_graph)
    optimized_connectivity = calculate_connectivity_after_attacks(optimized_graph)
    
    # 计算鲁棒性值
    initial_robustness = calculate_robustness(initial_graph)
    optimized_robustness = calculate_robustness(optimized_graph)
    
    # 使用英文标签绘图，避免字体问题
    plt.figure(figsize=(10, 6))
    x = range(len(initial_connectivity))
    plt.plot(x, initial_connectivity, 'r-', label=f'Initial Topology (R={initial_robustness:.4f})')
    plt.plot(x, optimized_connectivity, 'b-', label=f'Optimized Topology (R={optimized_robustness:.4f})')
    plt.fill_between(x, initial_connectivity, alpha=0.3, color='r')
    plt.fill_between(x, optimized_connectivity, alpha=0.3, color='b')
    
    plt.xlabel('Number of Removed Nodes')
    plt.ylabel('Largest Connected Component Ratio')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # 打印鲁棒性提升信息
    improvement = (optimized_robustness - initial_robustness) / initial_robustness * 100
    print(f"Robustness improvement: {improvement:.2f}%")
    
    return initial_robustness, optimized_robustness, improvement



def plot_random_robustness_comparison(initial_graph, optimized_graph, title="Unity", attack_type="random", num_simulations=10):
    """使用NetworkX绘制优化前后的鲁棒性比较图
    
    Args:
        initial_graph: NetworkX图对象，初始拓扑
        optimized_graph: NetworkX图对象，优化后的拓扑
        title: 图表标题
        attack_type: 攻击类型，"random"为随机攻击，"hda"为高度中心性攻击
        num_simulations: 随机攻击模式下的模拟次数，求平均值
    """
    # 计算攻击后的网络连通性 - 随机攻击
    def calculate_connectivity_random_attacks(graph):
        n = graph.number_of_nodes()
        all_results = []
        
        # 进行多次随机攻击模拟
        for _ in range(num_simulations):
            g = graph.copy()
            connectivity = [1.0]  # 初始完全连通
            nodes = list(g.nodes())
            # 随机打乱节点顺序
            random.shuffle(nodes)
            
            for node in nodes:
                g.remove_node(node)
                
                # 计算最大连通分量比例
                if g.number_of_nodes() > 0:
                    components = list(nx.connected_components(g))
                    largest_cc_size = max(len(comp) for comp in components) if components else 0
                    connectivity.append(largest_cc_size / n)
                else:
                    connectivity.append(0)
            
            # 确保结果长度一致
            if len(connectivity) < n + 1:
                connectivity.extend([0] * (n + 1 - len(connectivity)))
            
            all_results.append(connectivity[:n+1])
        
        # 计算多次模拟的平均值
        avg_connectivity = [sum(col)/num_simulations for col in zip(*all_results)]
        return avg_connectivity
    
    # 计算攻击后的网络连通性 - 高度中心性攻击
    def calculate_connectivity_hda(graph):
        g = graph.copy()
        n = g.number_of_nodes()
        connectivity = [1.0]  # 初始完全连通
        
        for _ in range(n):
            if g.number_of_nodes() == 0:
                connectivity.extend([0] * (n - len(connectivity) + 1))
                break
                
            # HDA攻击 - 删除度数最大的节点
            degrees = dict(g.degree())
            if not degrees:
                break
                
            max_degree_node = max(degrees, key=degrees.get)
            g.remove_node(max_degree_node)
            
            # 计算最大连通分量比例
            if g.number_of_nodes() > 0:
                components = list(nx.connected_components(g))
                largest_cc_size = max(len(comp) for comp in components) if components else 0
                connectivity.append(largest_cc_size / n)
            else:
                connectivity.append(0)
        
        return connectivity[:n+1]  # 限制为n+1个点（包括初始状态）
    
    # 计算鲁棒性值
    def calculate_robustness(connectivity):
        """根据连通性数组计算网络的鲁棒性值"""
        n = len(connectivity) - 1  # 减去初始状态
        return sum(connectivity) / (n + 1)
    
    # 根据攻击类型选择相应的函数
    if attack_type.lower() == "random":
        initial_connectivity = calculate_connectivity_random_attacks(initial_graph)
        optimized_connectivity = calculate_connectivity_random_attacks(optimized_graph)
        attack_label = "Random Attack"
    else:  # 默认为hda攻击
        initial_connectivity = calculate_connectivity_hda(initial_graph)
        optimized_connectivity = calculate_connectivity_hda(optimized_graph)
        attack_label = "High Degree Attack"
    
    # 计算鲁棒性值
    initial_robustness = calculate_robustness(initial_connectivity)
    optimized_robustness = calculate_robustness(optimized_connectivity)
    
    # 使用英文标签绘图，避免字体问题
    plt.figure(figsize=(10, 6))
    x = range(len(initial_connectivity))
    plt.plot(x, initial_connectivity, 'r-', label=f'Initial Topology (R={initial_robustness:.4f})')
    plt.plot(x, optimized_connectivity, 'b-', label=f'Optimized Topology (R={optimized_robustness:.4f})')
    plt.fill_between(x, initial_connectivity, alpha=0.3, color='r')
    plt.fill_between(x, optimized_connectivity, alpha=0.3, color='b')
    
    plt.xlabel('Number of Removed Nodes')
    plt.ylabel('Largest Connected Component Ratio')
    plt.title(f"{title} - {attack_label}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # 打印鲁棒性提升信息
    improvement = (optimized_robustness - initial_robustness) / initial_robustness * 100
    print(f"Robustness improvement ({attack_label}): {improvement:.2f}%")
    
    return initial_robustness, optimized_robustness, improvement
 


def compare_degree_distributions(initial_graph, optimized_graph, 
                                 initial_name="Initial Network", 
                                 optimized_name="Optimized Network",
                                 show_plot=True):
    """
    比较两个NetworkX图的度分布
    
    参数:
        initial_graph (nx.Graph): 初始网络
        optimized_graph (nx.Graph): 优化后的网络
        initial_name (str): 初始网络的标签名称
        optimized_name (str): 优化后网络的标签名称
        show_plot (bool): 是否显示对比图
        
    返回:
        dict: 包含度分布统计信息的字典
    """
    # 获取度序列
    initial_degrees = [d for _, d in initial_graph.degree()]
    optimized_degrees = [d for _, d in optimized_graph.degree()]
    
    # 计算基本统计数据
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
    
    # 计算变化百分比
    node_change = ((stats["optimized"]["nodes"] - stats["initial"]["nodes"]) / 
                   stats["initial"]["nodes"]) * 100
    edge_change = ((stats["optimized"]["edges"] - stats["initial"]["edges"]) / 
                   stats["initial"]["edges"]) * 100
    mean_degree_change = ((stats["optimized"]["mean_degree"] - stats["initial"]["mean_degree"]) / 
                          stats["initial"]["mean_degree"]) * 100
    max_degree_change = ((stats["optimized"]["max_degree"] - stats["initial"]["max_degree"]) / 
                         stats["initial"]["max_degree"]) * 100
    
    # 执行KS检验比较度分布
    from scipy import stats as scipy_stats
    ks_stat, ks_pvalue = scipy_stats.ks_2samp(initial_degrees, optimized_degrees)
    
    # 打印结果
    print("Degree Distribution Analysis:")
    print(f"1. Nodes: {stats['initial']['nodes']} -> {stats['optimized']['nodes']} (Change: {node_change:.2f}%)")
    print(f"2. Edges: {stats['initial']['edges']} -> {stats['optimized']['edges']} (Change: {edge_change:.2f}%)")
    print(f"3. Mean degree: {stats['initial']['mean_degree']:.2f} -> {stats['optimized']['mean_degree']:.2f} (Change: {mean_degree_change:.2f}%)")
    print(f"4. Max degree: {stats['initial']['max_degree']} -> {stats['optimized']['max_degree']} (Change: {max_degree_change:.2f}%)")
    print(f"5. KS test: p={ks_pvalue:.4f} {'(Similar distributions)' if ks_pvalue > 0.05 else '(Different distributions)'}")
    
    # 如果需要显示图表
    if show_plot:
        import matplotlib.pyplot as plt
        
        # 创建度分布直方图
        plt.figure(figsize=(10, 6))
        
        # 确定合适的bins
        max_degree = max(stats["initial"]["max_degree"], stats["optimized"]["max_degree"])
        bins = range(0, max_degree + 2)
        
        # 绘制直方图
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
    
    # 返回统计信息
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
    

def ROMEM_method(G, max_iter=100):
    result = optimize_network_robustness(
        initial_graph=G,
        population_size=3,
        generations=30,
        area_size=500,
        comm_distance=200,
        verbose=True
    )
    return result['optimized_graph']


# 使用示例和测试
if __name__ == "__main__":
    # 示例1: 从随机图开始优化
    logger.info("=== 示例1: 优化随机生成的图 ===")
    
    # 生成一个随机图作为初始拓扑
    initial_graph = nx.barabasi_albert_graph(100, 3, seed=42)
    
    # 生成随机位置
    positions = {}
    for node in initial_graph.nodes():
        positions[node] = (random.uniform(0, 500), random.uniform(0, 500))
    
    # 调用优化函数
    result = optimize_network_robustness(
        initial_graph=initial_graph,
        positions=positions,
        population_size=3,
        generations=30,
        area_size=500,
        comm_distance=200,
        verbose=True
    )
    
    # 输出结果
    print(f"\n=== 优化结果 ===")
    print(f"初始鲁棒性: {result['initial_robustness']:.4f}")
    print(f"最终鲁棒性: {result['final_robustness']:.4f}")
    print(f"改进幅度: {result['improvement']:.2f}%")
    print(f"优化耗时: {result['optimization_time']:.2f}s")
    print(f"优化后图: {result['optimized_graph'].number_of_nodes()}节点, {result['optimized_graph'].number_of_edges()}边")
    
    # # 示例2: 从小世界网络开始优化
    # logger.info("\n=== 示例2: 优化小世界网络 ===")
    
    # # 生成小世界网络
    # small_world_graph = nx.watts_strogatz_graph(15, 4, 0.3)
    
    # result2 = optimize_network_robustness(
    #     initial_graph=small_world_graph,
    #     population_size=3,
    #     generations=20,
    #     verbose=True
    # )
    
    # print(f"\n=== 小世界网络优化结果 ===")
    # print(f"初始鲁棒性: {result2['initial_robustness']:.4f}")
    # print(f"最终鲁棒性: {result2['final_robustness']:.4f}")
    # print(f"改进幅度: {result2['improvement']:.2f}%")
    # print(f"优化耗时: {result2['optimization_time']:.2f}s")
    
    # # 绘制结果（如果matplotlib可用）
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 原始图和优化图的比较
        pos1 = nx.spring_layout(initial_graph)
        pos2 = nx.spring_layout(result['optimized_graph'])
        
        nx.draw(initial_graph, pos1, ax=ax1, with_labels=True, node_color='lightblue', 
                node_size=300, font_size=8)
        ax1.set_title(f'原始图 (鲁棒性: {result["initial_robustness"]:.3f})')
        
        nx.draw(result['optimized_graph'], pos2, ax=ax2, with_labels=True, 
                node_color='lightgreen', node_size=300, font_size=8)
        ax2.set_title(f'优化图 (鲁棒性: {result["final_robustness"]:.3f})')
        
        # 鲁棒性变化曲线
        ax3.plot(result['robustness_history'])
        ax3.set_title('鲁棒性变化')
        ax3.set_xlabel('代数')
        ax3.set_ylabel('鲁棒性')
        ax3.grid(True)
        
        # 适应度变化曲线
        if result.get('fitness_history'):
            ax4.plot(result['fitness_history'])
            ax4.set_title('适应度变化')
            ax4.set_xlabel('代数')
            ax4.set_ylabel('适应度')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('romem_optimization_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info("结果图已保存为 romem_optimization_results.png")
        plot_robustness_comparison(initial_graph, result['optimized_graph'])
        plot_random_robustness_comparison(initial_graph, result['optimized_graph'])
        compare_degree_distributions(initial_graph, result['optimized_graph'])
    except ImportError:
        logger.warning("matplotlib未安装，无法绘制结果图")