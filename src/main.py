import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import os
import time
import optuna.visualization as vis 
from benchmarks import BenchmarkFactory
from optimizers import GreyWolfOptimizer
from stats_utils import compute_friedman_test, plot_critical_difference_diagram

# Global variables for experiment control
DIM = 10 
MAX_ITER = 50 
N_REPETITIONS = 5 
OPTUNA_TRIALS = 20 
SEED_TRANSFORM = 42 

# Success rate configuration
SUCCESS_THRESHOLD = 1e-2
KNOWN_OPTIMA = {
    'sphere': 0.0, 'rosenbrock': 0.0, 'rastrigin': 0.0, 'schwefel': 0.0,
    'ackley': 0.0, 'griewank': 0.0, 'michalewicz': -9.66015, 'zakharov': 0.0,
    'dixon_price': 0.0, 'levy': 0.0
}

# Directory structure
BASE_GRAPHICS_DIR = "graphics/GWO"
STRAT_OUTPUT_DIR = os.path.join(BASE_GRAPHICS_DIR, "Strategies")
POP_OUTPUT_DIR = os.path.join(BASE_GRAPHICS_DIR, "Population")

# Benchmark problems to analyze
PROBLEM_NAMES = [
    'sphere', 'rosenbrock', 'rastrigin', 'schwefel', 'ackley', 
    'griewank', 'michalewicz', 'zakharov', 'dixon_price', 'levy'
]

# Parameters for comparative study
STRATEGIES = ['linear', 'exp', 'log', 'sin']
POP_SIZES = [10, 20, 40, 80, 160]

def calculate_and_save_metrics(raw_data, problem_list, config_list, output_path):
    """Computes Mean, Median, STD, and Success Rate relative to known optima"""
    summary = []
    for name in problem_list:
        target = KNOWN_OPTIMA.get(name, 0.0)
        for config in config_list:
            vals = np.array(raw_data[name][config])
            # Success is defined as distance to the global minimum
            successes = np.abs(vals - target) < SUCCESS_THRESHOLD
            summary.append({
                'Problem': name.capitalize(),
                'Config': config,
                'Mean': np.mean(vals),
                'Median': np.median(vals),
                'Std Dev': np.std(vals),
                'Success Rate (%)': np.mean(successes) * 100
            })
    pd.DataFrame(summary).to_csv(output_path, index=False)

def global_objective(trial):
    """
    Optuna objective function to optimize population size and strategy globally.
    """
    pop_size = trial.suggest_int('pop_size', 10, 80)
    strategy = trial.suggest_categorical('strategy', STRATEGIES)
    scores = []
    
    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        reps = []
        for r in range(3):
            np.random.seed(trial.number * 100 + r)
            gwo = GreyWolfOptimizer(problem, pop_size, MAX_ITER)
            _, score, _ = gwo.optimize(strategy=strategy)
            reps.append(score)
        scores.append(np.mean(reps))
        
    return np.mean(scores)

def main():
    # Initialize output directory structure
    os.makedirs(STRAT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(POP_OUTPUT_DIR, exist_ok=True)

    # 1. OPTUNA PARAMETER TUNING
    print("Step 1: Finding optimal global parameters using Optuna")
    study = optuna.create_study(direction='minimize')
    study.optimize(global_objective, n_trials=OPTUNA_TRIALS)
    best_params = study.best_params
    print(f"Optimal configuration: {best_params}")

    # Export parameter importance plot
    try:
        vis.plot_param_importances(study).write_html(os.path.join(BASE_GRAPHICS_DIR, "param_importances.html"))
    except Exception as e:
        print(f"Skipping importance plot: {e}")

    # 2. STRATEGY COMPARISON
    print("\nStep 2: Analyzing decay strategies")
    results_strat = {s: [] for s in STRATEGIES}
    conv_strat = {p: {s: [] for s in STRATEGIES} for p in PROBLEM_NAMES}
    raw_strat_data = {p: {s: [] for s in STRATEGIES} for p in PROBLEM_NAMES} # For metrics

    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        for strat in STRATEGIES:
            f_reps, h_reps = [], []
            for rep in range(N_REPETITIONS):
                np.random.seed(rep)
                gwo = GreyWolfOptimizer(problem, best_params['pop_size'], MAX_ITER)
                _, score, history = gwo.optimize(strategy=strat)
                f_reps.append(score)
                h_reps.append(history)
            results_strat[strat].append(np.mean(f_reps))
            conv_strat[name][strat] = np.mean(h_reps, axis=0)
            raw_strat_data[name][strat] = f_reps # Collect raw data

    # Save Strategy metrics table
    calculate_and_save_metrics(raw_strat_data, PROBLEM_NAMES, STRATEGIES, os.path.join(STRAT_OUTPUT_DIR, "strategy_metrics.csv"))

    # Strategy Statistical Validation using Iman-Davenport correction
    df_strat = pd.DataFrame(results_strat, index=PROBLEM_NAMES)
    f_stat_s, p_val_s, ranks_s = compute_friedman_test(df_strat)
    print(f"Strategy Friedman Result: F = {f_stat_s:.4f}, p = {p_val_s:.4e}")

    if p_val_s < 0.05:
        print("Significant strategy difference found. Generating CD Diagram")
        plot_critical_difference_diagram(ranks_s, len(PROBLEM_NAMES), output_dir=STRAT_OUTPUT_DIR)
    else:
        print("No significant strategy differences detected")

    # 3. POPULATION SWEEP
    print("\nStep 3: Analyzing population scale impact")
    results_pop = {f"N_{n}": [] for n in POP_SIZES}
    times_pop = {f"N_{n}": [] for n in POP_SIZES}
    conv_pop = {p: {f"N_{n}": [] for n in POP_SIZES} for p in PROBLEM_NAMES}
    raw_pop_data = {p: {f"N_{n}": [] for n in POP_SIZES} for p in PROBLEM_NAMES} # For metrics
    
    target_strat = best_params['strategy']

    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        for n in POP_SIZES:
            f_reps, t_reps, h_reps = [], [], []
            for rep in range(N_REPETITIONS):
                np.random.seed(rep)
                gwo = GreyWolfOptimizer(problem, n, MAX_ITER)
                # Measure computational cost using process time
                start_t = time.process_time()
                _, score, history = gwo.optimize(strategy=target_strat)
                t_reps.append(time.process_time() - start_t)
                f_reps.append(score)
                h_reps.append(history)
            
            results_pop[f"N_{n}"].append(np.mean(f_reps))
            times_pop[f"N_{n}"].append(np.mean(t_reps))
            conv_pop[name][f"N_{n}"] = np.mean(h_reps, axis=0)
            raw_pop_data[name][f"N_{n}"] = f_reps # Collect raw data

    # Save Population metrics table
    calculate_and_save_metrics(raw_pop_data, PROBLEM_NAMES, [f"N_{n}" for n in POP_SIZES], os.path.join(POP_OUTPUT_DIR, "population_metrics.csv"))

    # Population Statistical Validation
    df_pop = pd.DataFrame(results_pop, index=PROBLEM_NAMES)
    f_stat_p, p_val_p, ranks_p = compute_friedman_test(df_pop)
    print(f"Population Friedman Result: F = {f_stat_p:.4f}, p = {p_val_p:.4e}")

    if p_val_p < 0.05:
        print("Significant population difference found. Generating CD Diagram")
        plot_critical_difference_diagram(ranks_p, len(PROBLEM_NAMES), output_dir=POP_OUTPUT_DIR)
    else:
        print("No significant population differences detected")

    # 4. ENHANCED VISUALIZATIONS
    print("\nStep 4: Generating plots and trend analysis")

    # A. Accuracy vs Cost Trend Plot
    plt.figure(figsize=(10, 6))
    avg_fitness = [np.mean(results_pop[f"N_{n}"]) for n in POP_SIZES]
    avg_times = [np.mean(times_pop[f"N_{n}"]) for n in POP_SIZES]
    plt.plot(avg_times, avg_fitness, '-o', color='darkred', lw=2, label='Population Scale Trend')
    
    for i, n in enumerate(POP_SIZES):
        plt.annotate(f"N={n}", (avg_times[i], avg_fitness[i]), xytext=(5, 5), textcoords="offset points")
    
    plt.yscale('symlog', linthresh=1e-2)
    plt.title("GWO Resource Trade-off: Precision vs CPU Time")
    plt.xlabel("Mean CPU Time (s)")
    plt.ylabel("Mean Fitness Value")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(POP_OUTPUT_DIR, "tradeoff_line_trend.png"))
    plt.close()

    # B. Population Convergence per Problem
    for name in PROBLEM_NAMES:
        plt.figure(figsize=(9, 6))
        vals = np.concatenate([conv_pop[name][f"N_{n}"] for n in POP_SIZES])
        for n in POP_SIZES:
            plt.plot(conv_pop[name][f"N_{n}"], label=f"N={n}")
        plt.title(f"Convergence by Population: {name.capitalize()}")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Score")

        if name.lower() == 'michalewicz':
            plt.yscale('linear')  # Michalewicz se ve mejor en escala lineal
        elif np.any(vals <= 0):
            plt.yscale('symlog', linthresh=1e-2)
        else:
            plt.yscale('log')

        plt.legend()
        plt.grid(True, which="both", alpha=0.2)
        plt.savefig(os.path.join(POP_OUTPUT_DIR, f"pop_conv_{name}.png"))
        plt.close()

    # C. Global Population Efficiency (Normalized)
    plt.figure(figsize=(10, 6))
    for n in POP_SIZES:
        norms = []
        for name in PROBLEM_NAMES:
            h = np.array(conv_pop[name][f"N_{n}"])
            h_min, h_max = h.min(), h.max()
            norms.append((h - h_min) / (h_max - h_min + 1e-12) if h_max > h_min else np.zeros_like(h))
        plt.plot(np.mean(norms, axis=0), label=f"N={n}")
    
    plt.title("Normalized Global Efficiency by Population Size")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Mean Fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(POP_OUTPUT_DIR, "global_pop_efficiency.png"))
    plt.close()

    # D. Strategy Convergence per Problem
    for name in PROBLEM_NAMES:
        plt.figure(figsize=(9, 6))
        vals = np.concatenate([conv_strat[name][strat] for strat in STRATEGIES])
        for strat in STRATEGIES:
            plt.plot(conv_strat[name][strat], label=strat)
        plt.title(f"Strategy Convergence: {name.capitalize()}")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Score")
        
        if name.lower() == 'michalewicz':
            plt.yscale('linear')  
        elif np.any(vals <= 0):
            plt.yscale('symlog', linthresh=1e-2)
        else:
            plt.yscale('log')

        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(os.path.join(STRAT_OUTPUT_DIR, f"conv_{name}.png"))
        plt.close()

    # E. Global Strategy Efficiency (Normalized Across All Problems)
    plt.figure(figsize=(10, 6))
    for strat in STRATEGIES:
        norms = []
        for name in PROBLEM_NAMES:
            h = np.array(conv_strat[name][strat])
            h_min, h_max = h.min(), h.max()
            norms.append((h - h_min) / (h_max - h_min + 1e-12) if h_max > h_min else np.zeros_like(h))
        plt.plot(np.mean(norms, axis=0), label=strat, lw=2.5)
    
    plt.title("Normalized Global Strategy Efficiency")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Mean Fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(STRAT_OUTPUT_DIR, "global_strategy_efficiency.png"))
    plt.close()

    print("All tasks finished. Results saved in 'graphics/GWO/'")

if __name__ == "__main__":
    main()