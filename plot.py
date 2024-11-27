import matplotlib.pyplot as plt
import numpy as np
import os

def process_file_and_plot_histogram(file_path, model, n, k):
    """
    Reads the file, extracts 'Algorithm Weight / Optimal Weight Ratio', and plots a histogram.

    Parameters:
    file_path (str): Path to the file containing the data.
    n (int): Number of nodes in the graph.
    k (int): Sparsity parameter for the graph.
    """
    weight_ratios = []
    
    # Read the file and extract weight ratios
    with open(file_path, 'r') as file:
        for line in file:
            if "Algorithm Weight / Optimal Weight Ratio" in line:
                # Extract the ratio value from the line
                ratio = float(line.split(":")[1].strip())
                weight_ratios.append(ratio)
    
    # Calculate the mean
    mean_ratio = sum(weight_ratios) / len(weight_ratios)
    
    # Define histogram bins
    bins = np.arange(1.0, 2.0, 0.05)  
    
    # Compute the histogram data
    bin_counts, bin_edges = np.histogram(weight_ratios, bins=bins)
    
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    bar_colors = plt.cm.plasma(np.linspace(0.3, 0.8, len(bin_counts))) 
    
    bars = plt.bar(bin_edges[:-1], bin_counts, width=np.diff(bins), color=bar_colors, edgecolor='black', alpha=0.8)
    
    # Add frequency labels inside bars
    for bar, count in zip(bars, bin_counts):
        if count > 0:  
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2, 
                str(count),
                ha='center',
                va='center',
                fontsize=10,
                color='white' 
            )
    
    # Highlight the mean with a vertical line
    plt.axvline(mean_ratio, color='blue', linestyle='--', label=f'Mean: {mean_ratio:.2f}')
    
    # Set the background color to light grey
    plt.gca().set_facecolor('#f0f0f0') 
    
    # Add titles and labels
    plt.xlabel('Algorithm Weight / Optimal Weight Ratio', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(np.arange(1.0, 2.0, 0.1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/histogram_{model}_{n}_{k}.png')


if __name__ == "__main__":
    model = 'ergnp'
    for n, k in [(200, 1), (200, 2), (200, 3), (200, 4), (200, 5), (200, 10), (2000, 3)]:
        file_path = f'output/{model}_{n}_{k}/results.txt'
        process_file_and_plot_histogram(file_path, model, n, k)
