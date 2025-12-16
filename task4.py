#!/usr/bin/env python3
"""
Task 4: Three-Way TSP Solver Comparison
Nearest Neighbor vs OR-Tools vs Genetic Algorithm

Implements a Genetic Algorithm as a third AI technique and compares
all three approaches on the same 30-instance experimental protocol.
"""

import argparse
import time
import random
from pathlib import Path
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import copy

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import seaborn as sns
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Unified color scheme for all visualizations (matching Assignment-3)
COLORS = {
    # Background and grid
    'background': '#f8f9fa',
    'grid': '#e9ecef',
    
    # Algorithm-specific colors
    'nn_primary': '#FF6B6B',      # Coral Red for NN 
    'nn_light': '#FFA07A',        # Light Salmon for NN
    'or_primary': '#4ECDC4',      # Teal/Cyan for OR-Tools
    'or_light': '#95E1D3',        # Light Teal for OR-Tools
    'ga_primary': '#9B59B6',      # Purple for Genetic Algorithm
    'ga_light': '#BB8FCE',        # Light Purple for GA
    
    # Node colors
    'nodes': '#A8DADC',           # Soft Blue for regular nodes
    'start_node': '#FF6B6B',      # Coral for start node (matches NN)
    
    # Edge colors
    'all_edges': '#CED4DA',       # Light Gray for complete graph edges
    
    # Text
    'title': '#2C3E50',           # Dark Blue-Gray for titles
    'text': '#34495E'             # Slate Gray for text
}


@dataclass
class ExperimentResult:
    """Three-algorithm experiment result data structure"""
    instance_id: int
    graph_type: str
    n_nodes: int
    seed: int
    
    # Nearest Neighbor results
    nn_tour_length: float
    nn_runtime_s: float
    nn_tour: List[int]
    
    # OR-Tools results
    or_tour_length: float
    or_runtime_s: float
    or_tour: List[int]
    
    # Genetic Algorithm results
    ga_tour_length: float
    ga_runtime_s: float
    ga_tour: List[int]
    ga_generations: int
    ga_best_fitness_history: List[float]
    
    # Comparison metrics
    nn_vs_or_gap: float  # (NN - OR) / OR * 100
    ga_vs_or_gap: float  # (GA - OR) / OR * 100
    nn_vs_ga_gap: float  # (NN - GA) / GA * 100
    
    def to_dict(self):
        """Dict without full tours and fitness history."""
        d = asdict(self)
        d.pop('nn_tour')
        d.pop('or_tour')
        d.pop('ga_tour')
        d.pop('ga_best_fitness_history')
        return d


class NearestNeighborSolver:
    """Greedy heuristic TSP solver"""
    
    def solve(self, G: Any, nodes: List[Any], start_node: Any, weight: str = "length") -> Tuple[List[Any], float]:
        """Solve TSP using Nearest Neighbor heuristic"""
        start_time = time.perf_counter()
        
        unvisited = set(nodes)
        tour = [start_node]
        unvisited.remove(start_node)
        
        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda node: G[current][node][weight])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        tour.append(start_node)
        runtime = time.perf_counter() - start_time
        
        return tour, runtime


class ORToolsSolver:
    """Google OR-Tools based TSP solver"""
    
    def __init__(self, time_limit_s: int = 10):
        self.time_limit_s = time_limit_s
    
    def solve(self, G: Any, nodes: List[Any], start_node: Any, weight: str = "length") -> Tuple[List[Any], float]:
        """Solve TSP using Google OR-Tools"""
        start_time = time.perf_counter()
        
        n = len(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_node = {idx: node for idx, node in enumerate(nodes)}
        start_idx = node_to_idx[start_node]
        
        scale = 1000.0
        
        manager = pywrapcp.RoutingIndexManager(n, 1, start_idx)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            if i == j:
                return 0
            u = nodes[i]
            v = nodes[j]
            length = G[u][v][weight]
            return int(length * scale)
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(self.time_limit_s)
        
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            index = routing.Start(0)
            tour_indices = []
            while not routing.IsEnd(index):
                tour_indices.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            tour_indices.append(tour_indices[0])
            
            tour = [idx_to_node[idx] for idx in tour_indices]
        else:
            tour = [start_node, start_node]
        
        runtime = time.perf_counter() - start_time
        return tour, runtime


class GeneticAlgorithmSolver:
    """Genetic Algorithm TSP solver"""
    def __init__(self, 
                 population_size: int = 100,
                 generations: int = 200,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 tournament_size: int = 5,
                 elitism_count: int = 2,
                 seed: Optional[int] = None):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.seed = seed
        
        self.best_fitness_history = []
        
    def solve(self, G: Any, nodes: List[Any], start_node: Any, weight: str = "length") -> Tuple[List[Any], float]:
        """Solve TSP using Genetic Algorithm"""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        start_time = time.perf_counter()
        
        self.G = G
        self.nodes = nodes
        self.start_node = start_node
        self.weight = weight
        self.n_cities = len(nodes)
        
        # Initialize population
        population = self._initialize_population()
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._fitness(individual) for individual in population]
            
            # Track best solution
            best_idx = np.argmin(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            self.best_fitness_history.append(best_fitness)
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[:self.elitism_count]
            elites = [population[i] for i in elite_indices]
            
            # Create new population
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self._swap_mutation(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._swap_mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Get best solution
        fitness_scores = [self._fitness(individual) for individual in population]
        best_idx = np.argmin(fitness_scores)
        best_tour = population[best_idx]
        
        # Convert to full tour (start -> cities -> start)
        tour = [self.start_node] + best_tour + [self.start_node]
        
        runtime = time.perf_counter() - start_time
        return tour, runtime
    
    def _initialize_population(self) -> List[List[Any]]:
        """Create initial population with random permutations"""
        population = []
        # Cities excluding start node
        cities = [n for n in self.nodes if n != self.start_node]
        
        for _ in range(self.population_size):
            individual = cities.copy()
            random.shuffle(individual)
            population.append(individual)
        
        return population
    
    def _fitness(self, individual: List[Any]) -> float:
        """Calculate tour length (fitness = tour length, minimize)"""
        tour = [self.start_node] + individual + [self.start_node]
        total_length = sum(
            self.G[tour[i]][tour[i+1]][self.weight]
            for i in range(len(tour) - 1)
        )
        return total_length
    
    def _tournament_selection(self, population: List[List[Any]], fitness_scores: List[float]) -> List[Any]:
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _order_crossover(self, parent1: List[Any], parent2: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Order Crossover (OX) - preserves relative order"""
        size = len(parent1)
        
        # Choose two random crossover points
        cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
        
        # Create children
        child1 = [None] * size
        child2 = [None] * size
        
        # Copy segments between crossover points
        child1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        child2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
        
        # Fill remaining positions maintaining order from other parent
        def fill_child(child, parent):
            child_set = set(child[cx_point1:cx_point2])
            parent_remaining = [item for item in parent if item not in child_set]
            
            idx = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = parent_remaining[idx]
                    idx += 1
            return child
        
        child1 = fill_child(child1, parent2)
        child2 = fill_child(child2, parent1)
        
        return child1, child2
    
    def _swap_mutation(self, individual: List[Any]) -> List[Any]:
        """Swap two random cities"""
        mutated = individual.copy()
        if len(mutated) > 1:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated


def calculate_tour_length(G: Any, tour: List[Any], weight: str = "length") -> float:
    """Sum of edge weights along a tour"""
    return float(
        sum(G[tour[i]][tour[i + 1]][weight] for i in range(len(tour) - 1))
    )


def create_random_euclidean_graph(n: int, seed: int, area=(0.0, 100.0)) -> Tuple[Any, List[int]]:
    """Create complete Euclidean graph on n random points"""
    random.seed(seed)
    np.random.seed(seed)
    
    G = nx.Graph()
    
    for i in range(n):
        x = random.uniform(*area)
        y = random.uniform(*area)
        G.add_node(i, x=x, y=y, pos=(x, y))
    
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = G.nodes[i]['x'], G.nodes[i]['y']
            xj, yj = G.nodes[j]['x'], G.nodes[j]['y']
            distance = float(np.hypot(xi - xj, yi - yj))
            G.add_edge(i, j, length=distance)
    
    return G, list(range(n))


def run_experiment(instance_id: int, n: int, seed: int, ga_params: dict) -> Tuple[ExperimentResult, Any, List]:
    """Run three-way comparison: NN vs OR-Tools vs GA"""
    
    print(f"  Experiment {instance_id}: n={n}, seed={seed}")
    
    G, nodes = create_random_euclidean_graph(n, seed)
    start_node = nodes[0]
    
    # === Nearest Neighbor ===
    nn_solver = NearestNeighborSolver()
    nn_tour, nn_time = nn_solver.solve(G, nodes, start_node, weight="length")
    nn_length = calculate_tour_length(G, nn_tour, weight="length")
    
    # === OR-Tools ===
    or_solver = ORToolsSolver(time_limit_s=10)
    or_tour, or_time = or_solver.solve(G, nodes, start_node, weight="length")
    or_length = calculate_tour_length(G, or_tour, weight="length")
    
    # === Genetic Algorithm ===
    ga_solver = GeneticAlgorithmSolver(**ga_params, seed=seed)
    ga_tour, ga_time = ga_solver.solve(G, nodes, start_node, weight="length")
    ga_length = calculate_tour_length(G, ga_tour, weight="length")
    ga_generations = ga_params['generations']
    ga_fitness_history = ga_solver.best_fitness_history.copy()
    
    # Comparison metrics
    nn_vs_or = ((nn_length - or_length) / or_length) * 100 if or_length > 0 else 0.0
    ga_vs_or = ((ga_length - or_length) / or_length) * 100 if or_length > 0 else 0.0
    nn_vs_ga = ((nn_length - ga_length) / ga_length) * 100 if ga_length > 0 else 0.0
    
    # Create index mapping
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    nn_tour_idx = [node_to_idx[node] for node in nn_tour]
    or_tour_idx = [node_to_idx[node] for node in or_tour]
    ga_tour_idx = [node_to_idx[node] for node in ga_tour]
    
    result = ExperimentResult(
        instance_id=instance_id,
        graph_type="random",
        n_nodes=n,
        seed=seed,
        nn_tour_length=nn_length,
        nn_runtime_s=nn_time,
        nn_tour=nn_tour_idx,
        or_tour_length=or_length,
        or_runtime_s=or_time,
        or_tour=or_tour_idx,
        ga_tour_length=ga_length,
        ga_runtime_s=ga_time,
        ga_tour=ga_tour_idx,
        ga_generations=ga_generations,
        ga_best_fitness_history=ga_fitness_history,
        nn_vs_or_gap=nn_vs_or,
        ga_vs_or_gap=ga_vs_or,
        nn_vs_ga_gap=nn_vs_ga
    )
    
    print(f"     ✓ NN: {nn_length:.2f} ({nn_time:.4f}s) | OR: {or_length:.2f} ({or_time:.2f}s) | GA: {ga_length:.2f} ({ga_time:.2f}s)")
    
    return result, G, nodes


def create_visualizations(results: List[ExperimentResult], output_dir: Path):
    """Generate core solution quality visualizations (Assignment-3 style)"""
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert to DataFrame
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # Set style
    sns.set_style("whitegrid")
    
    # ==== 1. SOLUTION QUALITY COMPARISON (2 panels) ====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot - Tour Length Distribution
    ax = axes[0]
    data = [df['nn_tour_length'], df['or_tour_length'], df['ga_tour_length']]
    bp = ax.boxplot(data, labels=['Nearest Neighbor', 'OR-Tools', 'Genetic Algorithm'], patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['nn_light'])
    bp['boxes'][1].set_facecolor(COLORS['or_light'])
    bp['boxes'][2].set_facecolor(COLORS['ga_light'])
    ax.set_title('Tour Length Distribution Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Tour Length (units)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Scatter plot - Direct Comparison
    ax = axes[1]
    ax.scatter(df['nn_tour_length'], df['or_tour_length'], alpha=0.6, s=100, c=COLORS['nn_primary'], edgecolors='black', linewidth=0.5, label='NN vs OR-Tools')
    ax.scatter(df['ga_tour_length'], df['or_tour_length'], alpha=0.6, s=100, c=COLORS['ga_primary'], edgecolors='black', linewidth=0.5, marker='s', label='GA vs OR-Tools')
    min_val = min(df['nn_tour_length'].min(), df['or_tour_length'].min(), df['ga_tour_length'].min())
    max_val = max(df['nn_tour_length'].max(), df['or_tour_length'].max(), df['ga_tour_length'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal Performance Line')
    ax.set_xlabel('NN/GA Tour Length (units)', fontsize=12)
    ax.set_ylabel('OR-Tools Tour Length (units)', fontsize=12)
    ax.set_title('Solution Quality: NN/GA vs OR-Tools', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_solution_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Saved: 1_solution_quality_comparison.png")
    
    # ==== 2. RUNTIME vs QUALITY TRADE-OFF ====
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot NN points
    ax.scatter(df['nn_runtime_s'], df['nn_tour_length'], 
              alpha=0.7, s=120, c=COLORS['nn_light'], marker='o', label='Nearest Neighbor', edgecolors='black', linewidth=1)
    
    # Plot OR-Tools points
    ax.scatter(df['or_runtime_s'], df['or_tour_length'], 
              alpha=0.7, s=120, c=COLORS['or_light'], marker='s', label='OR-Tools', edgecolors='black', linewidth=1)
    
    # Plot GA points
    ax.scatter(df['ga_runtime_s'], df['ga_tour_length'], 
              alpha=0.7, s=120, c=COLORS['ga_light'], marker='^', label='Genetic Algorithm', edgecolors='black', linewidth=1)
    
    # Add average markers
    ax.scatter(df['nn_runtime_s'].mean(), df['nn_tour_length'].mean(),
              s=400, c=COLORS['nn_primary'], marker='*', label='NN Average', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(df['or_runtime_s'].mean(), df['or_tour_length'].mean(),
              s=400, c=COLORS['or_primary'], marker='*', label='OR-Tools Average', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(df['ga_runtime_s'].mean(), df['ga_tour_length'].mean(),
              s=400, c=COLORS['ga_primary'], marker='*', label='GA Average', edgecolors='black', linewidth=2, zorder=5)
    
    ax.set_xlabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tour Length (units)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime vs Tour Length', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_runtime_quality_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Saved: 2_runtime_quality_tradeoff.png")

    # ==== 3. INSTANCE-BY-INSTANCE COMPARISON ====
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(results))
    width = 0.25
    
    bars1 = ax.bar(x - width, df['nn_tour_length'], width, 
                   label='Nearest Neighbor', color=COLORS['nn_light'], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, df['or_tour_length'], width, 
                   label='OR-Tools', color=COLORS['or_light'], alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, df['ga_tour_length'], width, 
                   label='Genetic Algorithm', color=COLORS['ga_light'], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Instance ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tour Length (units)', fontsize=12, fontweight='bold')
    ax.set_title('Instance-by-Instance Tour Length Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x[::max(1, len(results)//20)])  # Show every nth label to avoid clutter
    ax.set_xticklabels(df['instance_id'].values[::max(1, len(results)//20)])
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_instance_by_instance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3_instance_by_instance.png")
    
    # ==== 4. QUALITY GAP ANALYSIS ====
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Unified color scheme - NN bars in light red, GA bars in light purple
    x = np.arange(len(results))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['nn_vs_or_gap'], width,
                   label='NN vs OR-Tools', color=COLORS['nn_light'], alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, df['ga_vs_or_gap'], width,
                   label='GA vs OR-Tools', color=COLORS['ga_light'], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Average lines
    ax.axhline(y=df['nn_vs_or_gap'].mean(), color=COLORS['nn_primary'], linestyle='--', 
              linewidth=2.5, label=f'NN Avg Gap: {df["nn_vs_or_gap"].mean():.1f}%')
    ax.axhline(y=df['ga_vs_or_gap'].mean(), color=COLORS['ga_primary'], linestyle='--', 
              linewidth=2.5, label=f'GA Avg Gap: {df["ga_vs_or_gap"].mean():.1f}%')
    
    ax.set_xlabel('Instance ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quality Gap (%)', fontsize=12, fontweight='bold')
    ax.set_title('Quality Gap Analysis by Instance', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, framealpha=0.95, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_quality_gap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 4_quality_gap_analysis.png")


def visualize_comparison_example(G, nodes, result: ExperimentResult, output_dir: Path):
    """
    Create Task 1 style three-way visualization comparing NN, OR-Tools, and GA solutions.
    Shows all three tours on the same graph topology (Assignment-3 style).
    """
    
    pos = {i: (G.nodes[i]['x'], G.nodes[i]['y']) for i in nodes}
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))
    fig.patch.set_facecolor(COLORS['background'])
    
    algorithms = [
        ('Nearest Neighbor', result.nn_tour, result.nn_tour_length, result.nn_runtime_s, COLORS['nn_primary']),
        ('OR-Tools', result.or_tour, result.or_tour_length, result.or_runtime_s, COLORS['or_primary']),
        ('Genetic Algorithm', result.ga_tour, result.ga_tour_length, result.ga_runtime_s, COLORS['ga_primary'])
    ]
    
    all_edges = list(G.edges())
    
    for idx, (ax, (algo_name, tour, length, runtime, color)) in enumerate(zip(axes, algorithms)):
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, color=COLORS['grid'], linestyle='-', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        
        # Draw all edges in background
        nx.draw_networkx_edges(G, pos, edgelist=all_edges, 
                              edge_color=COLORS['all_edges'], 
                              width=0.8, alpha=0.3, ax=ax)
        
        # Draw tour edges
        tour_edges = [(tour[i], tour[i+1]) for i in range(len(tour)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=tour_edges, 
                              edge_color=color, 
                              width=3, alpha=0.8, ax=ax)
        
        # Draw nodes (Task 1 style)
        node_colors = [COLORS['start_node'] if i == nodes[0] else COLORS['nodes'] for i in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                              node_color=node_colors, 
                              node_size=600, 
                              alpha=0.9, ax=ax,
                              edgecolors='white', linewidths=2)
        
        # Numara etiketleri (0, 1, 2, ...)
        labels = {nodes[i]: str(i) for i in range(len(nodes))}
        nx.draw_networkx_labels(G, pos, labels=labels, 
                               font_size=12, font_weight='bold', 
                               font_color='white', ax=ax)
        
        # Title
        if idx == 2:  # GA panel - show comparison
            nn_gap = result.nn_vs_or_gap
            ga_gap = result.ga_vs_or_gap
            ax.set_title(f'{algo_name}\nTour Length: {length:.2f} units | Gap vs OR: {ga_gap:.1f}%', 
                        fontsize=16, fontweight='bold', color=COLORS['title'], pad=20)
        else:
            ax.set_title(f'{algo_name}\nTour Length: {length:.2f} units', 
                        fontsize=16, fontweight='bold', color=COLORS['title'], pad=20)
        
        # Axis settings
        ax.set_aspect('equal')
        margin = 5
        ax.set_xlim(-margin, 105)
        ax.set_ylim(-margin, 105)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['start_node'], 
                      markersize=12, label='Start Point (0)', markeredgecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['nodes'], 
                      markersize=12, label='Edges', markeredgecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], color=color, linewidth=3, label=f'{algo_name} Tour'),
            plt.Line2D([0], [0], color=COLORS['all_edges'], linewidth=1, alpha=0.3, label='All Connections')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_example_tours_visualization.png', dpi=200, 
               bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    print("  Saved: 5_example_tours_visualization.png")


def main():
    parser = argparse.ArgumentParser(description="Task 4: Three-Way TSP Solver Comparison")
    parser.add_argument("--n", type=int, default=20, help="Number of nodes per instance")
    parser.add_argument("--instances", type=int, default=30, help="Number of test instances")
    parser.add_argument("--seed-start", type=int, default=100, help="Starting seed")
    parser.add_argument("--output-dir", type=str, default="Assignment-4/results", help="Output directory")
    
    # GA parameters
    parser.add_argument("--ga-pop-size", type=int, default=100, help="GA population size")
    parser.add_argument("--ga-generations", type=int, default=200, help="GA generations")
    parser.add_argument("--ga-crossover", type=float, default=0.8, help="GA crossover rate")
    parser.add_argument("--ga-mutation", type=float, default=0.2, help="GA mutation rate")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nTask 4: Three-Way TSP Solver Comparison")
    print("=" * 70)
    print("Algorithms: Nearest Neighbor | OR-Tools | Genetic Algorithm")
    print("=" * 70)
    print("Configuration:")
    print(f"  Nodes per instance    : {args.n}")
    print(f"  Number of instances   : {args.instances}")
    print(f"  Starting seed         : {args.seed_start}")
    print(f"  GA Population Size    : {args.ga_pop_size}")
    print(f"  GA Generations        : {args.ga_generations}")
    print(f"  GA Crossover Rate     : {args.ga_crossover}")
    print(f"  GA Mutation Rate      : {args.ga_mutation}")
    print(f"  Output directory      : {output_dir}")
    print()
    
    ga_params = {
        'population_size': args.ga_pop_size,
        'generations': args.ga_generations,
        'crossover_rate': args.ga_crossover,
        'mutation_rate': args.ga_mutation,
        'tournament_size': 5,
        'elitism_count': 2
    }
    
    # Run experiments
    print("Running experiments...")
    results = []
    example_data = None
    
    for i in range(args.instances):
        instance_id = i + 1
        seed = args.seed_start + i
        try:
            result, G, nodes = run_experiment(instance_id, args.n, seed, ga_params)
            results.append(result)
            
            if i == 0:
                example_data = (G, nodes, result)
                
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("\n❌ No successful experiments!")
        return
    
    print(f"\nCompleted {len(results)} experiments.")
    
    # Save results
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_csv(output_dir / 'results.csv', index=False)
    print(f"\nSaved results to: {output_dir}/results.csv")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results, output_dir)
    
    if example_data is not None:
        G, nodes, example_result = example_data
        visualize_comparison_example(G, nodes, example_result, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total Instances: {len(results)}\n")
    print("Tour Length:")
    print(f"  NN:       {df['nn_tour_length'].mean():.2f} ± {df['nn_tour_length'].std():.2f}")
    print(f"  OR-Tools: {df['or_tour_length'].mean():.2f} ± {df['or_tour_length'].std():.2f}")
    print(f"  GA:       {df['ga_tour_length'].mean():.2f} ± {df['ga_tour_length'].std():.2f}")
    print("\nRuntime:")
    print(f"  NN:       {df['nn_runtime_s'].mean():.4f}s ± {df['nn_runtime_s'].std():.4f}s")
    print(f"  OR-Tools: {df['or_runtime_s'].mean():.2f}s ± {df['or_runtime_s'].std():.2f}s")
    print(f"  GA:       {df['ga_runtime_s'].mean():.2f}s ± {df['ga_runtime_s'].std():.2f}s")
    print("\nQuality Gaps (vs OR-Tools):")
    print(f"  NN:  {df['nn_vs_or_gap'].mean():.2f}%")
    print(f"  GA:  {df['ga_vs_or_gap'].mean():.2f}%")
    print("=" * 70)
    
    print("\nTask 4 completed successfully.\n")


if __name__ == "__main__":
    main()
