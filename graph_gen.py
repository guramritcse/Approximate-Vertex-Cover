import numpy as np
import os
import yaml
import shutil
from tqdm import tqdm


def clear_directory(directory_path):
    """
    Deletes all files and subdirectories inside the given directory without deleting the directory itself.
    
    Args:
        directory_path (str): Path to the directory to be cleared.
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


def random_seed(seed):
    """
    Seed the random number generator.

    Args:
        seed (int): Random seed.
    """
    np.random.seed(seed)


def erdos_renyi_gnp(n, k, dataset_dir, name):
    """
    Generate a random graph using the Erdős-Rényi model G(n, p) with random node weights.
    Save the graph in the dataset directory in a specified format.

    Args:
        n (int): Number of vertices.
        k (float): Sparsity parameter.
        dataset_dir (str): Directory to save the generated graph.
        name (int): Name of the graph file.
    """
    # Probability of edge creation
    p = k / n

    # Create an upper triangular adjacency matrix with probabilities
    adjacency_matrix = np.triu(np.random.random(size=(n, n)) < p, k=1)

    # Extract edges from the adjacency matrix
    edges = np.column_stack(np.where(adjacency_matrix))

    # Generate random weights for vertices
    weights = np.random.randint(1, 101, size=n)

    # Ensure dataset directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    # Save the graph to file
    file_path = f"{dataset_dir}/ergnp_{n}_{k}_{name}.txt"
    with open(file_path, "w") as f:
        f.write(f"{n} {len(edges)}\n")  # Write number of vertices and edges
        np.savetxt(f, weights, fmt="%d")  # Write weights
        np.savetxt(f, edges, fmt="%d")  # Write edges


if __name__ == "__main__":
    # Set random seed for reproducibility
    random_seed(42)

    # Load the YAML configuration
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    # Set the regex pattern for parsing the graph files
    gen_model = config["graph_generation"]["gen_model"]
    n = config["graph_generation"]["parameters"]["n"]
    k = config["graph_generation"]["parameters"]["k"]
    size = config["graph_generation"]["size"]

    # Ensure dataset directory exists
    dataset_dir = config["dataset"]["dir"]
    os.makedirs(dataset_dir, exist_ok=True)

    # Sanity checks
    assert gen_model in ["ergnp"], "Invalid graph generation model."
    assert n > 0, "Number of vertices should be positive."
    assert k >= 0 and k <= n, "Probability of edge creation (n/k) should be in the range [0, 1]."
    assert size > 0, "Number of graphs should be positive."

    if gen_model == "ergnp":
        subdir = f"{dataset_dir}/ergnp_{n}_{k}"
        os.makedirs(subdir, exist_ok=True)
        clear_directory(subdir)
        for i in tqdm(range(size), desc="Generating graphs"):
            erdos_renyi_gnp(n, k, subdir, i)

    print(f"Generated {size} graphs using the {gen_model} model with {n} vertices.")
