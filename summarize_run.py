import os
import numpy as np
import matplotlib.pyplot as plt

# Set the directory where run data is stored
run_dir = "sonnet_IFNG/demo_runs"

def get_step_indices(prefix, suffix):
    """Find all step indices for given file prefix/suffix."""
    indices = []
    pattern = re.compile(rf"{prefix}(\d+).*{suffix}")
    for fname in os.listdir(run_dir):
        match = pattern.match(fname)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)

def summarize_sampled_genes():
    """Load and summarize sampled gene sets for each step."""
    print("\n=== Sampled Genes per Step ===")
    step_counts = []
    step_indices = get_step_indices("sampled_genes_", ".npy")
    all_steps = []
    for i in step_indices:
        gene_file = os.path.join(run_dir, f"sampled_genes_{i}.npy")
        if os.path.exists(gene_file):
            genes = np.load(gene_file)
            step_counts.append(len(genes))
            all_steps.append(set(genes))
            print(f"Step {i}: {', '.join(genes[:10])} ... ({len(genes)} total genes)")
        else:
            print(f"sampled_genes_{i}.npy not found.")
            step_counts.append(0)
            all_steps.append(set())
    return step_counts, step_indices, all_steps

def summarize_step_logs():
    """Print summary (first 15 lines) of each step's log file."""
    print("\n=== Step Log Summary ===")
    step_indices = get_step_indices("step_", "_log.log")
    for i in step_indices:
        log_file = os.path.join(run_dir, f"step_{i}_log.log")
        if os.path.exists(log_file):
            print(f"\n--- Step {i} Log ---")
            with open(log_file, "r") as f:
                lines = f.readlines()
                preview = lines[:15]
                for line in preview:
                    print(line.strip())
                if len(lines) > 15:
                    print("... (truncated)")
        else:
            print(f"step_{i}_log.log not found.")

def plot_gene_counts(step_counts, step_indices):
    """Plot number of sampled genes per step."""
    plt.figure(figsize=(8, 5))
    plt.plot(step_indices, step_counts, marker='o', linestyle='-', color='blue')
    plt.xticks(step_indices)
    plt.xlabel("Step")
    plt.ylabel("Number of Genes")
    plt.title("Gene Count per Step (BioDiscoveryAgent)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "gene_count_plot.png"))
    plt.show()

def plot_new_gene_discovery(all_steps, step_indices):
    """Plot number of newly introduced genes at each step."""
    new_gene_counts = []
    seen = set()
    for genes in all_steps:
        new_genes = genes - seen
        new_gene_counts.append(len(new_genes))
        seen |= genes

    plt.figure(figsize=(8, 5))
    plt.plot(step_indices, new_gene_counts, marker='o', linestyle='-', color='green')
    plt.xticks(step_indices)
    plt.xlabel("Step")
    plt.ylabel("New Genes Introduced")
    plt.title("New Gene Discovery per Step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "new_gene_discovery_plot.png"))
    plt.show()

if __name__ == "__main__":
    step_counts, step_indices, all_steps = summarize_sampled_genes()
    summarize_step_logs()
    plot_gene_counts(step_counts, step_indices)
    plot_new_gene_discovery(all_steps, step_indices)