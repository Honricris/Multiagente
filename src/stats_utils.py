import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_wilcoxon_test(results_df):
    """
    Perform the Wilcoxon signed-ranks test for 2 algorithms.
    This is the recommended non-parametric alternative for pairwise comparisons.
    """
    if results_df.shape[1] != 2:
        raise ValueError("Wilcoxon test is intended for exactly 2 algorithms.")
    
    scores_a = results_df.iloc[:, 0]
    scores_b = results_df.iloc[:, 1]
    
    statistic, p_value = stats.wilcoxon(scores_a, scores_b)
    
    ranks = results_df.rank(axis=1, ascending=False, method='average')
    average_ranks = ranks.mean()
    
    return statistic, p_value, average_ranks

def compute_friedman_test(results_df):
    """
    Perform the Friedman test to compare multiple algorithms.
    Returns the F-statistic (Iman-Davenport correction) and p-value.
    """
    n_datasets, k_algorithms = results_df.shape
    if k_algorithms < 2:
        raise ValueError("Friedman test requires at least 2 algorithms.")

    # Execute Friedman test (standard chi-square)
    chi2_f, p_chi2 = stats.friedmanchisquare(*[results_df[col] for col in results_df.columns])

    # Iman and Davenport correction for a less conservative F-statistic
    f_stat = ((n_datasets - 1) * chi2_f) / (n_datasets * (k_algorithms - 1) - chi2_f)
    p_value = stats.f.sf(f_stat, k_algorithms - 1, (k_algorithms - 1) * (n_datasets - 1))

    # Rank algorithms (1 is best, k is worst)
    # Note: Higher fitness is worse in optimization, but ranks should reflect 1 as best
    ranks = results_df.rank(axis=1, ascending=True, method='average')
    average_ranks = ranks.mean()

    return f_stat, p_value, average_ranks

def plot_critical_difference_diagram(average_ranks, n_datasets, alpha=0.05, output_dir="."):
    """
    Generate a Critical Difference (CD) diagram using Nemenyi post-hoc test.
    Improved version to prevent label overlapping using staggered heights.
    """
    k = len(average_ranks)
    
    # Critical values (q_alpha) for Nemenyi test (Demšar, 2006, Table 5a)
    q_alpha_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    
    q_val = q_alpha_005.get(k, 3.0) 
    cd = q_val * np.sqrt((k * (k + 1)) / (6 * n_datasets))

    # Sorting for visualization
    sorted_ranks = average_ranks.sort_values()
    names = sorted_ranks.index
    values = sorted_ranks.values

    # Increased figure size for clarity
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    
    # Ranks: 1 (best) on the right
    ax.set_xlim(max(k, values.max()) + 0.5, 0.5) 
    ax.set_ylim(-1.5, 1.2)
    
    # Main axis
    plt.axhline(0, color='black', lw=1.5)
    ticks = np.arange(1, int(max(k, values.max())) + 1)
    plt.xticks(ticks, ticks, fontsize=11)
    plt.title(f"Critical Difference Diagram (CD={cd:.3f}, alpha={alpha})", pad=20, fontsize=13)

    # Plot algorithms and their ranks with dynamic staggered heights
    for i, (name, rank) in enumerate(zip(names, values)):
        # Cycles through 3 different vertical levels to prevent text overlap
        y_level = -0.3 - (i % 3) * 0.3 
        
        plt.plot([rank, rank], [0, y_level], color='black', lw=1, ls='--')
        plt.text(rank, y_level - 0.05, f"{name} ({rank:.2f})", 
                 ha='right', va='top', rotation=35, fontsize=10, fontweight='bold')

    # Plot CD bar at the top
    cd_x_start = max(ticks)
    plt.plot([cd_x_start, cd_x_start - cd], [0.8, 0.8], color='red', lw=4)
    plt.text(cd_x_start - cd/2, 0.9, "CD", color='red', ha='center', fontweight='bold')

    # Connect groups that are NOT significantly different (Blue lines)
    for i in range(k):
        for j in range(i + 1, k):
            if abs(values[i] - values[j]) <= cd:
                # Vertical offset for each connection line to keep them distinct
                plt.plot([values[i], values[j]], [0.2 + (i*0.07), 0.2 + (i*0.07)], 
                         color='blue', lw=3, alpha=0.5)

    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, "cd_diagram.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_boxplot_comparison(results_df, output_dir):
    """
    Generate a boxplot to visualize performance distribution.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=results_df)
    plt.yscale('symlog', linthresh=1e-3) # Support for negative values
    plt.title("Performance Distribution Comparison")
    plt.ylabel("Fitness Value (Symlog Scale)")
    plt.xlabel("Algorithm Configuration")
    
    save_path = os.path.join(output_dir, "boxplot_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()