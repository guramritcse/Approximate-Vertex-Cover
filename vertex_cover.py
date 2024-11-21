import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
import re
import os
import time
import yaml
import shutil


def clear_directory(directory_path):
    """
    Deletes all files and subdirectories inside the given directory without deleting the directory itself.
    
    :param directory_path: Path to the directory to be cleared.
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


def opt_vertex_cover(weights, edges):
    """
    Computes the exact optimal weighted vertex cover for a general graph using ILP.
    
    Args:
        edges (list of tuples): List of edges, where each edge is represented as a tuple (u, v).
        weights (list): List of weights for the vertices. weights[i] is the weight of vertex i.
    
    Returns:
        set, float: A vertex cover (set of vertices) and the total weight of the vertex cover.
    """
    # Number of vertices
    num_vertices = len(weights)

    # Create an ILP problem
    prob = LpProblem("Weighted_Vertex_Cover", LpMinimize)

    # Decision variables: x_u for each vertex u
    x = [LpVariable(f"x_{u}", cat=LpBinary) for u in range(num_vertices)]

    # Objective function: Minimize the total weight of the vertex cover
    prob += lpSum(weights[u] * x[u] for u in range(num_vertices)), "Total_Weight"

    # Constraints: Each edge must be covered
    for u, v in edges:
        prob += x[u] + x[v] >= 1, f"Edge_{u}_{v}_Covered"

    # Solve the ILP
    prob.solve(PULP_CBC_CMD(msg=False))

    # Extract the solution
    vertex_cover = {u for u in range(num_vertices) if x[u].varValue == 1}
    total_weight = sum(weights[u] for u in vertex_cover)

    return vertex_cover, total_weight


def alg_vertex_cover(weights, edges):
    """
    Implements the approximate primal-dual algorithm for weighted vertex cover using numpy arrays.
    
    Args:
        edges (list of tuples): List of edges, where each edge is represented as a tuple (u, v).
        weights (list): List of weights for the vertices. weights[i] is the weight of vertex i.
    
    Returns:
        set, float: A vertex cover (set of vertices) and the total weight of the vertex cover.
    """
    # Seed the random number generator
    np.random.seed(42)
    
    # Number of vertices
    num_vertices = len(weights)

    # Initialize the dual variable matrix
    ye = np.zeros((num_vertices, num_vertices))  # Dual variables for all edges, ye[u, v] is the dual variable for edge (u, v), for easy computation, we have ye[u, v] = ye[v, u] for all u, v
    xu = np.zeros(num_vertices)  # Primal variables for vertices
    vertex_cover = set()         # Vertex cover set
    
    # Create a matrix that represents what edges are yet to be covered and initialize it, note that for easy computation, we have edges_not_covered[u, v] = edges_not_covered[v, u] for all u, v
    edges_not_covered = np.zeros((num_vertices, num_vertices))
    for u, v in edges:
        edges_not_covered[u, v] = edges_not_covered[v, u] = 1
    
    while True:
        # Check if all edges are covered
        if np.sum(edges_not_covered) == 0:
            break
        
        # Pick a random edge that is not covered
        idxs = np.where(edges_not_covered)
        idx = np.random.randint(len(idxs[0]))
        u, v = idxs[0][idx], idxs[1][idx]
        
        # Calculate the increment for the dual variable
        increment_possible = [weights[u] - np.sum(ye[u, :]), weights[v] - np.sum(ye[v, :])]
        min_idx = np.argmin(increment_possible)
        increment = increment_possible[min_idx]
        primal_var_idx = u if min_idx == 0 else v

        # Update dual variable
        ye[u, v] += increment
        ye[v, u] += increment

        # Update primal variable and vertex cover
        xu[primal_var_idx] = 1
        vertex_cover.add(primal_var_idx)
        edges_not_covered[primal_var_idx, :] = edges_not_covered[:, primal_var_idx] = 0
    
    # Calculate the total weight of the vertex cover
    total_weight = sum(weights[v] for v in vertex_cover)
    
    return vertex_cover, total_weight


def process_graphs(dataset_dir):
    """
    Process a list of graph files and compute the optimal and approximate weighted vertex covers.

    Args:
        dataset_dir (str): Directory containing the graph files.

    Returns:
        dict: A dictionary containing the optimal and approximate weighted vertex covers for each graph.
    """
    # Dictionary to store the results
    alg_opt_dict = dict()

    for file in os.listdir(dataset_dir):
        # Read the graph file
        with open(os.path.join(dataset_dir, file), "r") as f:
            n, m = map(int, f.readline().split())
            weights = [int(f.readline()) for _ in range(n)]
            edges = [tuple(map(int, f.readline().split())) for _ in range(m)]

        # Compute the optimal weighted vertex cover
        opt_time_start = time.time()
        opt_cover, opt_weight = opt_vertex_cover(weights, edges)
        opt_time_end = time.time()
        opt_time = (opt_time_end - opt_time_start) * 1e3

        # Compute the approximate weighted vertex cover
        alg_time_start = time.time()
        alg_cover, alg_weight = alg_vertex_cover(weights, edges)
        alg_time_end = time.time()
        alg_time = (alg_time_end - alg_time_start) * 1e3

        # Store the results in a dictionary
        alg_opt_dict[file] = {"opt": (opt_cover, opt_weight, opt_time), "alg": (alg_cover, alg_weight, alg_time)}

    return alg_opt_dict


def print_results(alg_opt_dict, output_dir):
    """
    Print the results of the optimal and approximate weighted vertex covers.

    Args:
        alg_opt_dict (dict): A dictionary containing the optimal and approximate weighted vertex covers for each graph.
        output_dir (str): Directory to save the results.
    """
    # Print the results in output_dir/results.txt
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        alg_time = []
        opt_time = []
        alg_opt_time_ratio = []
        alg_opt_weight_ratio = []
        for file, values in alg_opt_dict.items():
            _, aw, at = values["alg"]
            _, ow, ot = values["opt"]
            alg_time.append(at)
            opt_time.append(ot)
            alg_opt_time_ratio.append(at / ot)
            alg_opt_weight_ratio.append(aw / ow)
            f.write(f"Graph: {file}\n")
            f.write(f"Algorithm Time: {at:.2f} ms\n")
            f.write(f"Optimal Time: {ot:.2f} ms\n")
            f.write(f"Algorithm Time / Optimal Time Ratio: {at / ot:.2f}\n")
            f.write(f"Algorithm Weight / Optimal Weight Ratio: {aw / ow:.2f}\n")
            f.write("\n")
        avg_alg_time = sum(alg_time) / len(alg_time)
        avg_opt_time = sum(opt_time) / len(opt_time)
        avg_alg_opt_time_ratio = sum(alg_opt_time_ratio) / len(alg_opt_time_ratio)
        avg_alg_opt_weight_ratio = sum(alg_opt_weight_ratio) / len(alg_opt_weight_ratio)
        f.write(f"Average Algorithm Time: {avg_alg_time:.2f} ms\n")
        f.write(f"Average Optimal Time: {avg_opt_time:.2f} ms\n")
        f.write(f"Average Algorithm Time / Optimal Time Ratio: {avg_alg_opt_time_ratio:.2f}\n")
        f.write(f"Average Algorithm Weight / Optimal Weight Ratio: {avg_alg_opt_weight_ratio:.2f}\n")
    
    # Print alg_opt_dict in output_dir/dict.txt
    with open(os.path.join(output_dir, "dict.txt"), "w") as f:
        f.write(str(alg_opt_dict))


if __name__ == "__main__":
    # Load the YAML configuration
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    # Set the regex pattern for parsing the graph files
    gen_model = config["graph_generation"]["gen_model"]
    n = config["graph_generation"]["parameters"]["n"]
    k = config["graph_generation"]["parameters"]["k"]
    dataset_dir = f"{config["dataset"]["dir"]}/{gen_model}_{n}_{k}"
    output_dir = f"{config["output"]["dir"]}/{gen_model}_{n}_{k}"

    # Copy the graph files to the output directory
    os.makedirs(output_dir, exist_ok=True)
    clear_directory(output_dir)
    for file in os.listdir(dataset_dir):
        shutil.copyfile(os.path.join(dataset_dir, file), os.path.join(output_dir, file))

    # Process each graph file
    alg_opt_dict = process_graphs(dataset_dir)

    # Print the results
    print_results(alg_opt_dict, output_dir)
