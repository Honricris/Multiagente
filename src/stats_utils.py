import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_wilcoxon_test(results_df):
    """
    Perform the Wilcoxon signed-ranks test for 2 algorithms.
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
    """
    n_datasets, k_algorithms = results_df.shape
    if k_algorithms < 2:
        raise ValueError("Friedman test requires at least 2 algorithms.")

    # Execute Friedman test
    chi2_f, p_chi2 = stats.friedmanchisquare(*[results_df[col] for col in results_df.columns])

    # Iman and Davenport correction
    f_stat = ((n_datasets - 1) * chi2_f) / (n_datasets * (k_algorithms - 1) - chi2_f)
    p_value = stats.f.sf(f_stat, k_algorithms - 1, (k_algorithms - 1) * (n_datasets - 1))

    ranks = results_df.rank(axis=1, ascending=True, method='average')
    average_ranks = ranks.mean()

    return f_stat, p_value, average_ranks

def plot_critical_difference_diagram(average_ranks, n_datasets, alpha=0.05, output_dir="."):
    """
    Generate a Critical Difference (CD) diagram using Nemenyi post-hoc test.
    """
    k = len(average_ranks)
    q_alpha_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_val = q_alpha_005.get(k, 3.0) 
    cd = q_val * np.sqrt((k * (k + 1)) / (6 * n_datasets))

    sorted_ranks = average_ranks.sort_values()
    names = sorted_ranks.index
    values = sorted_ranks.values

    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.set_xlim(max(k, values.max()) + 0.7, 0.3) 
    ax.set_ylim(-2.0, 1.5)
    plt.axhline(0, color='black', lw=1.5)
    ticks = np.arange(1, int(max(k, values.max())) + 1)
    plt.xticks(ticks, ticks, fontsize=11)
    plt.title(f"Critical Difference Diagram (CD={cd:.3f}, alpha={alpha})", pad=20, fontsize=13)

    for i, (name, rank) in enumerate(zip(names, values)):
        y_level = -0.4 - (i % 3) * 0.45 
        plt.plot([rank, rank], [0, y_level], color='black', lw=1, ls='--')
        plt.text(rank, y_level - 0.1, f"{name} ({rank:.2f})", ha='center', va='top', rotation=30, fontsize=10, fontweight='bold')

    cd_x_start = max(ticks)
    plt.plot([cd_x_start, cd_x_start - cd], [1.1, 1.1], color='red', lw=4)
    plt.text(cd_x_start - cd/2, 1.2, "CD", color='red', ha='center', fontweight='bold')

    for i in range(k):
        for j in range(i + 1, k):
            if abs(values[i] - values[j]) <= cd:
                plt.plot([values[i], values[j]], [0.25 + (i*0.1), 0.25 + (i*0.1)], color='blue', lw=3, alpha=0.5)

    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, "cd_diagram.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_boxplot_comparison(results_df, output_dir, filename="boxplot_comparison.png", title="Performance Distribution", y_scale='symlog'):
    """
    Generate a boxplot to visualize performance distribution.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=results_df)
    
    if y_scale == 'symlog':
        plt.yscale('symlog', linthresh=1e-3)
    else:
        plt.yscale(y_scale)
        
    plt.title(title)
    plt.ylabel(f"Fitness Value ({y_scale.capitalize()} Scale)")
    plt.xlabel("Algorithm Configuration")
    plt.grid(True, axis='y', alpha=0.3)
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()