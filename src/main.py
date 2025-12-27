import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import time
import optuna.visualization as vis 
import seaborn as sns 
from benchmarks import BenchmarkFactory
from optimizers import GreyWolfOptimizer, WhaleOptimizationAlgorithm 
from stats_utils import compute_friedman_test, plot_critical_difference_diagram, compute_wilcoxon_test, plot_boxplot_comparison

# --- Global Experiment Configuration ---
DIM = 30 # Search space dimensionality
MAX_ITER = 500 # Maximum iterations per optimization run
N_REPETITIONS = 30 # Number of independent runs for statistical significance
OPTUNA_TRIALS = 50 # Number of hyperparameter optimization trials
SEED_TRANSFORM = 42 # Seed for benchmark objective function transformations

# --- GWO Optimization Ranges ---
GWO_POP_RANGE = (10, 200) # Population size range for GWO tuning

# --- WOA Parameters ---
WOA_POP_RANGE = (10, 200) # Population size range for WOA tuning
WOA_B_RANGE = (0.5, 3.0) # Range for the constant b in spiral update
WOA_P_RANGE = (0.2, 0.8) # Range for the probability p switch

# --- Parameter Sweep Values ---
STRATEGIES = ['linear', 'exp', 'log', 'sin'] # Convergence parameter decay strategies
POP_SIZES = [10, 20, 40, 80, 100, 160, 200] # Population sizes for scalability analysis
WOA_B_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5] # Fixed b values for sensitivity analysis
WOA_P_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7] # Fixed p values for sensitivity analysis

# --- Success Thresholds and Known Optima ---
SUCCESS_THRESHOLDS = {
    'sphere': 1e-6,
    'rosenbrock': 1e-4,
    'rastrigin': 1e-2,
    'schwefel': 1e-2,
    'ackley': 1e-3,
    'griewank': 1e-2,
    'michalewicz': 1e-1,
    'zakharov': 1e-5,
    'dixon_price': 1e-4,
    'levy': 1e-5
}
KNOWN_OPTIMA = {
    'sphere': 0.0, 'rosenbrock': 0.0, 'rastrigin': 0.0, 'schwefel': 0.0,
    'ackley': 0.0, 'griewank': 0.0, 'michalewicz': -9.66015, 'zakharov': 0.0,
    'dixon_price': 0.0, 'levy': 0.0
}

# --- Directory Structure ---
BASE_GRAPHICS_DIR_GWO = "graphics/GWO"
STRAT_OUTPUT_DIR_GWO = os.path.join(BASE_GRAPHICS_DIR_GWO, "Strategies")
POP_OUTPUT_DIR_GWO = os.path.join(BASE_GRAPHICS_DIR_GWO, "Population")

BASE_GRAPHICS_DIR_WOA = "graphics/WOA"
STRAT_OUTPUT_DIR_WOA = os.path.join(BASE_GRAPHICS_DIR_WOA, "Strategies")
POP_OUTPUT_DIR_WOA = os.path.join(BASE_GRAPHICS_DIR_WOA, "Population")
B_OUTPUT_DIR_WOA = os.path.join(BASE_GRAPHICS_DIR_WOA, "Parameter_b")
P_OUTPUT_DIR_WOA = os.path.join(BASE_GRAPHICS_DIR_WOA, "Parameter_p")

COMPARISON_DIR = "graphics/Comparison"
TRAJECTORY_DIR = "graphics/Trajectories"

PROBLEM_NAMES = [
    'sphere', 'rosenbrock', 'rastrigin', 'schwefel', 'ackley', 
    'griewank', 'michalewicz', 'zakharov', 'dixon_price', 'levy'
]

def calculate_xpl_xpt_numeric(pos_history):
    """Calculates exploration and exploitation percentages based on population diversity"""
    pos_history = np.array(pos_history)
    
    # Measure diversity as average absolute difference from the median
    medians = np.median(pos_history, axis=1, keepdims=True)
    div_t = np.mean(np.abs(pos_history - medians), axis=(1, 2))
    
    div_max = np.max(div_t) if np.max(div_t) > 0 else 1e-12
    
    # Calculate iteration-wise percentages
    xpl_t = (div_t / div_max) * 100
    xpt_t = (np.abs(div_t - div_max) / div_max) * 100
    
    # Return average values of the entire run
    return np.mean(xpl_t), np.mean(xpt_t)

def calculate_and_save_metrics(raw_data, problem_list, config_list, output_path):
    """Calcula promedios, medianas y tasas de éxito usando umbrales específicos"""
    summary = []
    for name in problem_list:
        target = KNOWN_OPTIMA.get(name, 0.0)
        threshold = SUCCESS_THRESHOLDS.get(name, 1e-2)
        
        for config in config_list:
            vals = np.array(raw_data[name][config])
            successes = np.abs(vals - target) < threshold
            
            summary.append({
                'Problem': name.capitalize(),
                'Config': config,
                'Mean': np.mean(vals),
                'Median': np.median(vals),
                'Std Dev': np.std(vals),
                'Success Rate (%)': np.mean(successes) * 100
            })
    pd.DataFrame(summary).to_csv(output_path, index=False)

def gwo_global_objective(trial):
    """Optuna objective function for GWO hyperparameter tuning"""
    pop_size = trial.suggest_int('pop_size', GWO_POP_RANGE[0], GWO_POP_RANGE[1])
    strategy = trial.suggest_categorical('strategy', STRATEGIES)
    scores = []
    
    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        reps = []
        for r in range(10):
            np.random.seed(trial.number * 100 + r)
            gwo = GreyWolfOptimizer(problem, pop_size, MAX_ITER)
            _, score, _, _ = gwo.optimize(strategy=strategy)
            reps.append(score)
        scores.append(np.mean(reps))
        
    return np.mean(scores)

def woa_global_objective(trial):
    """Optuna objective function for WOA hyperparameter tuning"""
    pop_size = trial.suggest_int('pop_size', WOA_POP_RANGE[0], WOA_POP_RANGE[1])
    strategy = trial.suggest_categorical('strategy', STRATEGIES)
    b_val = trial.suggest_float('b', WOA_B_RANGE[0], WOA_B_RANGE[1])
    p_val = trial.suggest_float('p', WOA_P_RANGE[0], WOA_P_RANGE[1])
    
    scores = []
    
    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        reps = []
        for r in range(10):
            np.random.seed(trial.number * 100 + r)
            woa = WhaleOptimizationAlgorithm(problem, pop_size, MAX_ITER)
            _, score, _, _ = woa.optimize(b=b_val, p_switch=p_val, strategy=strategy)
            reps.append(score)
        scores.append(np.mean(reps))
        
    return np.mean(scores)

def get_mean_trajectory(problem, pos_history_list):
    """Extracts the average best fitness and coordinate progress across repetitions"""
    all_z1_paths = []
    all_fit_paths = []

    for pos_history in pos_history_list:
        z1_traj = []
        fit_traj = []
        current_best_fit = np.inf
        current_best_z = None

        for pop in pos_history:
            scores = [problem.compute(ind) for ind in pop]
            min_idx = np.argmin(scores)
            
            if scores[min_idx] < current_best_fit:
                current_best_fit = scores[min_idx]
                current_best_z = problem._transform(pop[min_idx])
            
            z1_traj.append(current_best_z[0])
            fit_traj.append(current_best_fit)
        
        all_z1_paths.append(z1_traj)
        all_fit_paths.append(fit_traj)

    # Compute average across repetitions
    mean_z1 = np.mean(all_z1_paths, axis=0)
    mean_fit = np.mean(all_fit_paths, axis=0)
    
    return mean_z1, mean_fit

def plot_mean_trajectory_comparison(z1_gwo, fit_gwo, z1_woa, fit_woa, name, save_path):
    """Plots a 2D comparison of algorithm trajectories in the search space"""
    plt.figure(figsize=(10, 7))
    
    # GWO Visualization
    plt.plot(z1_gwo, fit_gwo, color='red', label='GWO Mean Progress', 
             alpha=0.8, lw=2, marker='o', markevery=max(1, len(z1_gwo)//10))
    
    # WOA Visualization
    plt.plot(z1_woa, fit_woa, color='blue', label='WOA Mean Progress', 
             alpha=0.8, lw=2, ls='--', marker='^', markevery=max(1, len(z1_woa)//10))

    # Mark final average points
    plt.scatter(z1_gwo[-1], fit_gwo[-1], color='red', edgecolor='black', 
                marker='X', s=150, zorder=5, label='GWO Mean Final')
    plt.scatter(z1_woa[-1], fit_woa[-1], color='blue', edgecolor='black', 
                marker='X', s=150, zorder=5, label='WOA Mean Final')

    plt.title(f"Average Trajectory (n={N_REPETITIONS}): {name.capitalize()}", fontsize=12, fontweight='bold')
    plt.xlabel("Average Principal Dimension ($z_1$)")
    plt.ylabel("Average Best Fitness Found")
    
    all_fits = np.concatenate([fit_gwo, fit_woa])
    if name.lower() == 'michalewicz' or np.any(all_fits <= 0):
        plt.yscale('linear')
    else:
        plt.yscale('log')
    
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    # Initialize output directory structure
    for d in [STRAT_OUTPUT_DIR_GWO, POP_OUTPUT_DIR_GWO, 
              STRAT_OUTPUT_DIR_WOA, POP_OUTPUT_DIR_WOA, 
              B_OUTPUT_DIR_WOA, P_OUTPUT_DIR_WOA, COMPARISON_DIR, TRAJECTORY_DIR]:
        os.makedirs(d, exist_ok=True)

    print("="*60)
    print("STARTING GWO ANALYSIS")
    print("="*60)

    # 1. OPTUNA TUNING (GWO)
    print("Step 1 (GWO): Finding optimal global parameters using Optuna")
    study_gwo = optuna.create_study(direction='minimize')
    study_gwo.optimize(gwo_global_objective, n_trials=OPTUNA_TRIALS)
    best_params_gwo = study_gwo.best_params
    print(f"Optimal GWO configuration: {best_params_gwo}")

    try:
        vis.plot_param_importances(study_gwo).write_html(os.path.join(BASE_GRAPHICS_DIR_GWO, "param_importances.html"))
    except Exception as e:
        print(f"Skipping GWO importance plot: {e}")

    # 2. STRATEGY COMPARISON (GWO)
    print("\nStep 2 (GWO): Analyzing decay strategies")
    results_strat_gwo = {s: [] for s in STRATEGIES}
    conv_strat_gwo = {p: {s: [] for s in STRATEGIES} for p in PROBLEM_NAMES}
    raw_strat_data_gwo = {p: {s: [] for s in STRATEGIES} for p in PROBLEM_NAMES}

    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        for strat in STRATEGIES:
            f_reps, h_reps = [], []
            for rep in range(N_REPETITIONS):
                np.random.seed(rep)
                gwo = GreyWolfOptimizer(problem, best_params_gwo['pop_size'], MAX_ITER)
                _, score, history, _ = gwo.optimize(strategy=strat)
                f_reps.append(score)
                h_reps.append(history)
            results_strat_gwo[strat].append(np.mean(f_reps))
            conv_strat_gwo[name][strat] = np.mean(h_reps, axis=0)
            raw_strat_data_gwo[name][strat] = f_reps

    calculate_and_save_metrics(raw_strat_data_gwo, PROBLEM_NAMES, STRATEGIES, os.path.join(STRAT_OUTPUT_DIR_GWO, "strategy_metrics.csv"))

    df_strat_gwo = pd.DataFrame(results_strat_gwo, index=PROBLEM_NAMES)
    f_stat_s, p_val_s, ranks_s = compute_friedman_test(df_strat_gwo)
    print(f"GWO Strategy Friedman Result: F = {f_stat_s:.4f}, p = {p_val_s:.4e}")

    if p_val_s < 0.05:
        print("Significant GWO strategy difference found. Generating CD Diagram")
        plot_critical_difference_diagram(ranks_s, len(PROBLEM_NAMES), output_dir=STRAT_OUTPUT_DIR_GWO)

    # 3. POPULATION SWEEP (GWO)
    print("\nStep 3 (GWO): Analyzing population scale impact")
    results_pop_gwo = {f"N_{n}": [] for n in POP_SIZES}
    times_pop_gwo = {f"N_{n}": [] for n in POP_SIZES}
    conv_pop_gwo = {p: {f"N_{n}": [] for n in POP_SIZES} for p in PROBLEM_NAMES}
    raw_pop_data_gwo = {p: {f"N_{n}": [] for n in POP_SIZES} for p in PROBLEM_NAMES}
    
    target_strat_gwo = best_params_gwo['strategy']

    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        for n in POP_SIZES:
            f_reps, t_reps, h_reps = [], [], []
            for rep in range(N_REPETITIONS):
                np.random.seed(rep)
                gwo = GreyWolfOptimizer(problem, n, MAX_ITER)
                start_t = time.process_time()
                _, score, history, _ = gwo.optimize(strategy=target_strat_gwo)
                t_reps.append(time.process_time() - start_t)
                f_reps.append(score)
                h_reps.append(history)
            
            results_pop_gwo[f"N_{n}"].append(np.mean(f_reps))
            times_pop_gwo[f"N_{n}"].append(np.mean(t_reps))
            conv_pop_gwo[name][f"N_{n}"] = np.mean(h_reps, axis=0)
            raw_pop_data_gwo[name][f"N_{n}"] = f_reps

    calculate_and_save_metrics(raw_pop_data_gwo, PROBLEM_NAMES, [f"N_{n}" for n in POP_SIZES], os.path.join(POP_OUTPUT_DIR_GWO, "population_metrics.csv"))

    df_pop_gwo = pd.DataFrame(results_pop_gwo, index=PROBLEM_NAMES)
    f_stat_p, p_val_p, ranks_p = compute_friedman_test(df_pop_gwo)
    print(f"GWO Population Friedman Result: F = {f_stat_p:.4f}, p = {p_val_p:.4e}")

    if p_val_p < 0.05:
        print("Significant GWO population difference found. Generating CD Diagram")
        plot_critical_difference_diagram(ranks_p, len(PROBLEM_NAMES), output_dir=POP_OUTPUT_DIR_GWO)

    # 4. GWO VISUALIZATIONS
    print("\nStep 4 (GWO): Generating plots")
    
    plt.figure(figsize=(10, 6))
    avg_fit_gwo = [np.mean(results_pop_gwo[f"N_{n}"]) for n in POP_SIZES]
    avg_time_gwo = [np.mean(times_pop_gwo[f"N_{n}"]) for n in POP_SIZES]
    plt.plot(avg_time_gwo, avg_fit_gwo, '-o', color='darkred', lw=2, label='Population Scale Trend')
    for i, n in enumerate(POP_SIZES):
        plt.annotate(f"N={n}", (avg_time_gwo[i], avg_fit_gwo[i]), xytext=(5, 5), textcoords="offset points")
    plt.yscale('symlog', linthresh=1e-2)
    plt.title("GWO Resource Trade-off")
    plt.xlabel("Mean CPU Time (s)")
    plt.ylabel("Mean Fitness Value")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(POP_OUTPUT_DIR_GWO, "tradeoff_line_trend.png"))
    plt.close()

    for name in PROBLEM_NAMES:
        plt.figure(figsize=(9, 6))
        vals = np.concatenate([conv_pop_gwo[name][f"N_{n}"] for n in POP_SIZES])
        for n in POP_SIZES:
            plt.plot(conv_pop_gwo[name][f"N_{n}"], label=f"N={n}")
        plt.title(f"GWO Convergence by Population: {name.capitalize()}")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Score")
        if name.lower() == 'michalewicz': plt.yscale('linear')
        elif np.any(vals <= 0): plt.yscale('symlog', linthresh=1e-2)
        else: plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", alpha=0.2)
        plt.savefig(os.path.join(POP_OUTPUT_DIR_GWO, f"pop_conv_{name}.png"))
        plt.close()

    for name in PROBLEM_NAMES:
        plt.figure(figsize=(9, 6))
        vals = np.concatenate([conv_strat_gwo[name][strat] for strat in STRATEGIES])
        for strat in STRATEGIES:
            plt.plot(conv_strat_gwo[name][strat], label=strat)
        plt.title(f"GWO Strategy Convergence: {name.capitalize()}")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Score")
        if name.lower() == 'michalewicz': plt.yscale('linear')
        elif np.any(vals <= 0): plt.yscale('symlog', linthresh=1e-2)
        else: plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(os.path.join(STRAT_OUTPUT_DIR_GWO, f"conv_{name}.png"))
        plt.close()

    plt.figure(figsize=(10, 6))
    for n in POP_SIZES:
        norms = []
        for name in PROBLEM_NAMES:
            h = np.array(conv_pop_gwo[name][f"N_{n}"])
            h_min, h_max = h.min(), h.max()
            norms.append((h - h_min) / (h_max - h_min + 1e-12) if h_max > h_min else np.zeros_like(h))
        plt.plot(np.mean(norms, axis=0), label=f"N={n}")
    plt.title("GWO Normalized Global Efficiency (Population)")
    plt.legend()
    plt.savefig(os.path.join(POP_OUTPUT_DIR_GWO, "global_pop_efficiency.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for strat in STRATEGIES:
        norms = []
        for name in PROBLEM_NAMES:
            h = np.array(conv_strat_gwo[name][strat])
            h_min, h_max = h.min(), h.max()
            norms.append((h - h_min) / (h_max - h_min + 1e-12) if h_max > h_min else np.zeros_like(h))
        plt.plot(np.mean(norms, axis=0), label=strat, lw=2.5)
    plt.title("GWO Normalized Global Efficiency (Strategy)")
    plt.legend()
    plt.savefig(os.path.join(STRAT_OUTPUT_DIR_GWO, "global_strategy_efficiency.png"))
    plt.close()

    print("\n" + "="*60)
    print("STARTING WOA ANALYSIS")
    print("="*60)

    # 1. OPTUNA TUNING (WOA)
    print("Step 1 (WOA): Finding optimal global parameters using Optuna")
    study_woa = optuna.create_study(direction='minimize')
    study_woa.optimize(woa_global_objective, n_trials=OPTUNA_TRIALS)
    best_params_woa = study_woa.best_params
    print(f"Optimal WOA configuration: {best_params_woa}")

    try:
        vis.plot_param_importances(study_woa).write_html(os.path.join(BASE_GRAPHICS_DIR_WOA, "param_importances.html"))
    except Exception as e:
        print(f"Skipping WOA importance plot: {e}")

    # 2. STRATEGY COMPARISON (WOA)
    print("\nStep 2 (WOA): Analyzing decay strategies")
    results_strat_woa = {s: [] for s in STRATEGIES}
    conv_strat_woa = {p: {s: [] for s in STRATEGIES} for p in PROBLEM_NAMES}
    raw_strat_data_woa = {p: {s: [] for s in STRATEGIES} for p in PROBLEM_NAMES}

    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        for strat in STRATEGIES:
            f_reps, h_reps = [], []
            for rep in range(N_REPETITIONS):
                np.random.seed(rep)
                woa = WhaleOptimizationAlgorithm(problem, best_params_woa['pop_size'], MAX_ITER)
                _, score, history, _ = woa.optimize(b=best_params_woa['b'], 
                                                 p_switch=best_params_woa['p'], 
                                                 strategy=strat)
                f_reps.append(score)
                h_reps.append(history)
            results_strat_woa[strat].append(np.mean(f_reps))
            conv_strat_woa[name][strat] = np.mean(h_reps, axis=0)
            raw_strat_data_woa[name][strat] = f_reps

    calculate_and_save_metrics(raw_strat_data_woa, PROBLEM_NAMES, STRATEGIES, os.path.join(STRAT_OUTPUT_DIR_WOA, "strategy_metrics.csv"))

    df_strat_woa = pd.DataFrame(results_strat_woa, index=PROBLEM_NAMES)
    f_stat_s, p_val_s, ranks_s = compute_friedman_test(df_strat_woa)
    print(f"WOA Strategy Friedman Result: F = {f_stat_s:.4f}, p = {p_val_s:.4e}")

    if p_val_s < 0.05:
        print("Significant WOA strategy difference found. Generating CD Diagram")
        plot_critical_difference_diagram(ranks_s, len(PROBLEM_NAMES), output_dir=STRAT_OUTPUT_DIR_WOA)

    # 3. POPULATION SWEEP (WOA)
    print("\nStep 3 (WOA): Analyzing population scale impact")
    results_pop_woa = {f"N_{n}": [] for n in POP_SIZES}
    times_pop_woa = {f"N_{n}": [] for n in POP_SIZES}
    conv_pop_woa = {p: {f"N_{n}": [] for n in POP_SIZES} for p in PROBLEM_NAMES}
    raw_pop_data_woa = {p: {f"N_{n}": [] for n in POP_SIZES} for p in PROBLEM_NAMES}
    
    target_strat_woa = best_params_woa['strategy']

    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        for n in POP_SIZES:
            f_reps, t_reps, h_reps = [], [], []
            for rep in range(N_REPETITIONS):
                np.random.seed(rep)
                woa = WhaleOptimizationAlgorithm(problem, n, MAX_ITER)
                start_t = time.process_time()
                _, score, history, _ = woa.optimize(b=best_params_woa['b'], 
                                                 p_switch=best_params_woa['p'], 
                                                 strategy=target_strat_woa)
                t_reps.append(time.process_time() - start_t)
                f_reps.append(score)
                h_reps.append(history)
            
            results_pop_woa[f"N_{n}"].append(np.mean(f_reps))
            times_pop_woa[f"N_{n}"].append(np.mean(t_reps))
            conv_pop_woa[name][f"N_{n}"] = np.mean(h_reps, axis=0)
            raw_pop_data_woa[name][f"N_{n}"] = f_reps

    calculate_and_save_metrics(raw_pop_data_woa, PROBLEM_NAMES, [f"N_{n}" for n in POP_SIZES], os.path.join(POP_OUTPUT_DIR_WOA, "population_metrics.csv"))

    df_pop_woa = pd.DataFrame(results_pop_woa, index=PROBLEM_NAMES)
    f_stat_p, p_val_p, ranks_p = compute_friedman_test(df_pop_woa)
    print(f"WOA Population Friedman Result: F = {f_stat_p:.4f}, p = {p_val_p:.4e}")

    if p_val_p < 0.05:
        print("Significant WOA population difference found. Generating CD Diagram")
        plot_critical_difference_diagram(ranks_p, len(PROBLEM_NAMES), output_dir=POP_OUTPUT_DIR_WOA)
    
    # 4. PARAMETER B SWEEP (WOA)
    print("\nStep 4 (WOA): Analyzing Parameter b impact")
    results_b_woa = {f"b_{v}": [] for v in WOA_B_VALUES}
    conv_b_woa = {p: {f"b_{v}": [] for v in WOA_B_VALUES} for p in PROBLEM_NAMES}
    raw_b_data_woa = {p: {f"b_{v}": [] for v in WOA_B_VALUES} for p in PROBLEM_NAMES}

    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        for b_val in WOA_B_VALUES:
            f_reps, h_reps = [], []
            for rep in range(N_REPETITIONS):
                np.random.seed(rep)
                woa = WhaleOptimizationAlgorithm(problem, best_params_woa['pop_size'], MAX_ITER)
                _, score, history, _ = woa.optimize(b=b_val, 
                                                 p_switch=best_params_woa['p'], 
                                                 strategy=target_strat_woa)
                f_reps.append(score)
                h_reps.append(history)
            results_b_woa[f"b_{b_val}"].append(np.mean(f_reps))
            conv_b_woa[name][f"b_{b_val}"] = np.mean(h_reps, axis=0)
            raw_b_data_woa[name][f"b_{b_val}"] = f_reps

    calculate_and_save_metrics(raw_b_data_woa, PROBLEM_NAMES, [f"b_{v}" for v in WOA_B_VALUES], os.path.join(B_OUTPUT_DIR_WOA, "parameter_b_metrics.csv"))

    df_b_woa = pd.DataFrame(results_b_woa, index=PROBLEM_NAMES)
    f_stat_b, p_val_b, ranks_b = compute_friedman_test(df_b_woa)
    print(f"WOA Parameter b Friedman Result: F = {f_stat_b:.4f}, p = {p_val_b:.4e}")
    if p_val_b < 0.05:
        plot_critical_difference_diagram(ranks_b, len(PROBLEM_NAMES), output_dir=B_OUTPUT_DIR_WOA)

    # 5. PARAMETER P SWEEP (WOA)
    print("\nStep 5 (WOA): Analyzing Parameter p impact")
    results_p_woa = {f"p_{v}": [] for v in WOA_P_VALUES}
    conv_p_woa = {p: {f"p_{v}": [] for v in WOA_P_VALUES} for p in PROBLEM_NAMES}
    raw_p_data_woa = {p: {f"p_{v}": [] for v in WOA_P_VALUES} for p in PROBLEM_NAMES}

    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        for p_val in WOA_P_VALUES:
            f_reps, h_reps = [], []
            for rep in range(N_REPETITIONS):
                np.random.seed(rep)
                woa = WhaleOptimizationAlgorithm(problem, best_params_woa['pop_size'], MAX_ITER)
                _, score, history, _ = woa.optimize(b=best_params_woa['b'], 
                                                 p_switch=p_val, 
                                                 strategy=target_strat_woa)
                f_reps.append(score)
                h_reps.append(history)
            results_p_woa[f"p_{p_val}"].append(np.mean(f_reps))
            conv_p_woa[name][f"p_{p_val}"] = np.mean(h_reps, axis=0)
            raw_p_data_woa[name][f"p_{p_val}"] = f_reps

    calculate_and_save_metrics(raw_p_data_woa, PROBLEM_NAMES, [f"p_{v}" for v in WOA_P_VALUES], os.path.join(P_OUTPUT_DIR_WOA, "parameter_p_metrics.csv"))

    df_p_woa = pd.DataFrame(results_p_woa, index=PROBLEM_NAMES)
    f_stat_p_val, p_val_p_val, ranks_p_val = compute_friedman_test(df_p_woa)
    print(f"WOA Parameter p Friedman Result: F = {f_stat_p_val:.4f}, p = {p_val_p_val:.4e}")
    if p_val_p_val < 0.05:
        plot_critical_difference_diagram(ranks_p_val, len(PROBLEM_NAMES), output_dir=P_OUTPUT_DIR_WOA)

    # 6. WOA VISUALIZATIONS
    print("\nStep 6 (WOA): Generating plots")
    
    plt.figure(figsize=(10, 6))
    avg_fit_woa = [np.mean(results_pop_woa[f"N_{n}"]) for n in POP_SIZES]
    avg_time_woa = [np.mean(times_pop_woa[f"N_{n}"]) for n in POP_SIZES]
    plt.plot(avg_time_woa, avg_fit_woa, '-o', color='darkblue', lw=2, label='Population Scale Trend')
    for i, n in enumerate(POP_SIZES):
        plt.annotate(f"N={n}", (avg_time_woa[i], avg_fit_woa[i]), xytext=(5, 5), textcoords="offset points")
    plt.yscale('symlog', linthresh=1e-2)
    plt.title("WOA Resource Trade-off")
    plt.xlabel("Mean CPU Time (s)")
    plt.ylabel("Mean Fitness Value")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(POP_OUTPUT_DIR_WOA, "tradeoff_line_trend.png"))
    plt.close()

    for name in PROBLEM_NAMES:
        plt.figure(figsize=(9, 6))
        vals = np.concatenate([conv_pop_woa[name][f"N_{n}"] for n in POP_SIZES])
        for n in POP_SIZES:
            plt.plot(conv_pop_woa[name][f"N_{n}"], label=f"N={n}")
        plt.title(f"WOA Convergence by Population: {name.capitalize()}")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Score")
        if name.lower() == 'michalewicz': plt.yscale('linear')
        elif np.any(vals <= 0): plt.yscale('symlog', linthresh=1e-2)
        else: plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", alpha=0.2)
        plt.savefig(os.path.join(POP_OUTPUT_DIR_WOA, f"pop_conv_{name}.png"))
        plt.close()

    for name in PROBLEM_NAMES:
        plt.figure(figsize=(9, 6))
        vals = np.concatenate([conv_strat_woa[name][strat] for strat in STRATEGIES])
        for strat in STRATEGIES:
            plt.plot(conv_strat_woa[name][strat], label=strat)
        plt.title(f"WOA Strategy Convergence: {name.capitalize()}")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Score")
        if name.lower() == 'michalewicz': plt.yscale('linear')
        elif np.any(vals <= 0): plt.yscale('symlog', linthresh=1e-2)
        else: plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(os.path.join(STRAT_OUTPUT_DIR_WOA, f"conv_{name}.png"))
        plt.close()

    for name in PROBLEM_NAMES:
        plt.figure(figsize=(9, 6))
        vals = np.concatenate([conv_b_woa[name][f"b_{v}"] for v in WOA_B_VALUES])
        for v in WOA_B_VALUES:
            plt.plot(conv_b_woa[name][f"b_{v}"], label=f"b={v}")
        plt.title(f"WOA Parameter b Convergence: {name.capitalize()}")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Score")
        if name.lower() == 'michalewicz': plt.yscale('linear')
        elif np.any(vals <= 0): plt.yscale('symlog', linthresh=1e-2)
        else: plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(os.path.join(B_OUTPUT_DIR_WOA, f"param_b_conv_{name}.png"))
        plt.close()

    for name in PROBLEM_NAMES:
        plt.figure(figsize=(9, 6))
        vals = np.concatenate([conv_p_woa[name][f"p_{v}"] for v in WOA_P_VALUES])
        for v in WOA_P_VALUES:
            plt.plot(conv_p_woa[name][f"p_{v}"], label=f"p={v}")
        plt.title(f"WOA Parameter p Convergence: {name.capitalize()}")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Score")
        if name.lower() == 'michalewicz': plt.yscale('linear')
        elif np.any(vals <= 0): plt.yscale('symlog', linthresh=1e-2)
        else: plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(os.path.join(P_OUTPUT_DIR_WOA, f"param_p_conv_{name}.png"))
        plt.close()

    plt.figure(figsize=(10, 6))
    for n in POP_SIZES:
        norms = []
        for name in PROBLEM_NAMES:
            h = np.array(conv_pop_woa[name][f"N_{n}"])
            h_min, h_max = h.min(), h.max()
            norms.append((h - h_min) / (h_max - h_min + 1e-12) if h_max > h_min else np.zeros_like(h))
        plt.plot(np.mean(norms, axis=0), label=f"N={n}")
    plt.title("WOA Normalized Global Efficiency (Population)")
    plt.legend()
    plt.savefig(os.path.join(POP_OUTPUT_DIR_WOA, "global_pop_efficiency.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for strat in STRATEGIES:
        norms = []
        for name in PROBLEM_NAMES:
            h = np.array(conv_strat_woa[name][strat])
            h_min, h_max = h.min(), h.max()
            norms.append((h - h_min) / (h_max - h_min + 1e-12) if h_max > h_min else np.zeros_like(h))
        plt.plot(np.mean(norms, axis=0), label=strat, lw=2.5)
    plt.title("WOA Normalized Global Efficiency (Strategy)")
    plt.legend()
    plt.savefig(os.path.join(STRAT_OUTPUT_DIR_WOA, "global_strategy_efficiency.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for v in WOA_B_VALUES:
        norms = []
        for name in PROBLEM_NAMES:
            h = np.array(conv_b_woa[name][f"b_{v}"])
            h_min, h_max = h.min(), h.max()
            norms.append((h - h_min) / (h_max - h_min + 1e-12) if h_max > h_min else np.zeros_like(h))
        plt.plot(np.mean(norms, axis=0), label=f"b={v}")
    plt.title("WOA Normalized Global Efficiency (Parameter b)")
    plt.legend()
    plt.savefig(os.path.join(B_OUTPUT_DIR_WOA, "global_param_b_efficiency.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for v in WOA_P_VALUES:
        norms = []
        for name in PROBLEM_NAMES:
            h = np.array(conv_p_woa[name][f"p_{v}"])
            h_min, h_max = h.min(), h.max()
            norms.append((h - h_min) / (h_max - h_min + 1e-12) if h_max > h_min else np.zeros_like(h))
        plt.plot(np.mean(norms, axis=0), label=f"p={v}")
    plt.title("WOA Normalized Global Efficiency (Parameter p)")
    plt.legend()
    plt.savefig(os.path.join(P_OUTPUT_DIR_WOA, "global_param_p_efficiency.png"))
    plt.close()

    # 7. FINAL COMPARATIVE ANALYSIS
    print("\n" + "="*60)
    print("STARTING FINAL COMPARISON: GWO vs WOA")
    print("="*60)

    comparison_stats = []
    gwo_means_for_wilcoxon = []
    woa_means_for_wilcoxon = []
    global_gwo_times, global_gwo_fits = [], []
    global_woa_times, global_woa_fits = [], []

    for name in PROBLEM_NAMES:
        print(f"Comparing algorithms on {name}...")
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        target = KNOWN_OPTIMA.get(name, 0.0)
        
        gwo_reps_score, gwo_reps_time, gwo_histories = [], [], []
        woa_reps_score, woa_reps_time, woa_histories = [], [], []
        
        for rep in range(N_REPETITIONS):
            np.random.seed(rep)
            gwo = GreyWolfOptimizer(problem, best_params_gwo['pop_size'], MAX_ITER)
            start = time.process_time()
            _, score, history, _ = gwo.optimize(strategy=best_params_gwo['strategy'])
            end = time.process_time()
            gwo_reps_score.append(score)
            gwo_reps_time.append(end - start)
            gwo_histories.append(history)
            
        for rep in range(N_REPETITIONS):
            np.random.seed(rep)
            woa = WhaleOptimizationAlgorithm(problem, best_params_woa['pop_size'], MAX_ITER)
            start = time.process_time()
            _, score, history, _ = woa.optimize(b=best_params_woa['b'], 
                                             p_switch=best_params_woa['p'], 
                                             strategy=best_params_woa['strategy'])
            end = time.process_time()
            woa_reps_score.append(score)
            woa_reps_time.append(end - start)
            woa_histories.append(history)
            
        gwo_mean = np.mean(gwo_reps_score)
        woa_mean = np.mean(woa_reps_score)

        current_threshold = SUCCESS_THRESHOLDS.get(name, 1e-2)

        gwo_success = np.mean(np.abs(np.array(gwo_reps_score) - target) < current_threshold) * 100
        woa_success = np.mean(np.abs(np.array(woa_reps_score) - target) < current_threshold) * 100
        
        gwo_means_for_wilcoxon.append(gwo_mean)
        woa_means_for_wilcoxon.append(woa_mean)
        global_gwo_times.append(np.mean(gwo_reps_time))
        global_gwo_fits.append(gwo_mean)
        global_woa_times.append(np.mean(woa_reps_time))
        global_woa_fits.append(woa_mean)
        
        winner = "GWO" if gwo_mean < woa_mean else "WOA"
        
        comparison_stats.append({
            'Problem': name.capitalize(),
            'GWO Mean': gwo_mean, 'WOA Mean': woa_mean,
            'GWO Median': np.median(gwo_reps_score), 'WOA Median': np.median(woa_reps_score),
            'GWO Std': np.std(gwo_reps_score), 'WOA Std': np.std(woa_reps_score),
            'GWO Success (%)': gwo_success, 'WOA Success (%)': woa_success,
            'GWO Time (s)': np.mean(gwo_reps_time), 'WOA Time (s)': np.mean(woa_reps_time),
            'Winner': winner
        })
        
        plt.figure(figsize=(8, 6))
        plt.plot(np.mean(gwo_histories, axis=0), label='GWO', color='red', linewidth=2)
        plt.plot(np.mean(woa_histories, axis=0), label='WOA', color='blue', linewidth=2, linestyle='--')
        plt.title(f"Convergence Comparison: {name.capitalize()}")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Best Fitness")
        if name.lower() == 'michalewicz': plt.yscale('linear')
        else: plt.yscale('symlog', linthresh=1e-2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(COMPARISON_DIR, f"compare_conv_{name}.png"))
        plt.close()
        
        df_box = pd.DataFrame({'GWO': gwo_reps_score, 'WOA': woa_reps_score})
        scale = 'linear' if name.lower() == 'michalewicz' else 'symlog'
        plot_boxplot_comparison(df_box, COMPARISON_DIR, 
                                filename=f"compare_box_{name}.png", 
                                title=f"Stability Analysis: {name.capitalize()}", 
                                y_scale=scale)

    df_comp = pd.DataFrame(comparison_stats)
    df_comp.to_csv(os.path.join(COMPARISON_DIR, "comparison_summary.csv"), index=False)

    print("\nPerforming Wilcoxon Signed-Rank Test (Global)...")
    try:
        df_wilcoxon = pd.DataFrame({'GWO': gwo_means_for_wilcoxon, 'WOA': woa_means_for_wilcoxon}, index=PROBLEM_NAMES)
        stat, p_value, _ = compute_wilcoxon_test(df_wilcoxon)
        print(f"Wilcoxon Statistic: {stat}, P-Value: {p_value:.4e}")
        
        with open(os.path.join(COMPARISON_DIR, "wilcoxon_results.txt"), "w") as f:
            f.write(f"Wilcoxon Statistic: {stat}\nP-Value: {p_value:.4e}\n")
            f.write("Significant: Yes" if p_value < 0.05 else "Significant: No")
    except Exception as e:
        print(f"Error calculating Wilcoxon test: {e}")

    # Final comparative plots
    plt.figure(figsize=(10, 6))
    plt.scatter(global_gwo_times, global_gwo_fits, color='red', label='GWO', marker='o', s=80, alpha=0.7)
    plt.scatter(global_woa_times, global_woa_fits, color='blue', label='WOA', marker='^', s=80, alpha=0.7)
    for i in range(len(PROBLEM_NAMES)):
        plt.annotate("", xy=(global_woa_times[i], global_woa_fits[i]), 
                     xytext=(global_gwo_times[i], global_gwo_fits[i]),
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5, lw=1.5))
    plt.yscale('symlog', linthresh=1e-2)
    plt.title("Global Trade-off Movement (GWO -> WOA)")
    plt.xlabel("Mean CPU Time (s)")
    plt.ylabel("Mean Fitness (Symlog)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(COMPARISON_DIR, "global_tradeoff_comparison.png"))
    plt.close()

    rpd_data = []
    for i in range(len(PROBLEM_NAMES)):
        g_val, w_val = gwo_means_for_wilcoxon[i], woa_means_for_wilcoxon[i]
        denom = abs(g_val) if abs(g_val) > 1e-9 else 1e-9
        rpd_data.append(((w_val - g_val) / denom) * 100)

    plt.figure(figsize=(8, 6))
    sns.boxplot(y=rpd_data, color='lightblue', width=0.4)
    plt.axhline(0, color='red', linestyle='--', label="GWO Baseline")
    plt.title("Performance Deviation of WOA relative to GWO")
    plt.ylabel("RPD (%) (>0 means WOA is worse)")
    plt.legend()
    plt.savefig(os.path.join(COMPARISON_DIR, "global_boxplot_normalized.png"))
    plt.close()

# --- TRAJECTORIES AND XPL/XPT BALANCE ---
    print("\n" + "="*60)
    print("STARTING XPL/XPT NUMERIC ANALYSIS")
    print("="*60)
    
    xpl_xpt_data = []

    for name in PROBLEM_NAMES:
        print(f"Analyzing XPL/XPT balance for {name}...")
        np.random.seed(SEED_TRANSFORM)
        prob_traj = BenchmarkFactory.create(name, DIM)
        
        gwo_reps_xpl, gwo_reps_xpt = [], []
        woa_reps_xpl, woa_reps_xpt = [], []
        
        # Save histories for average trajectory visualization
        gwo_reps_hist = []
        woa_reps_hist = []

        for r in range(N_REPETITIONS):
            np.random.seed(r)
            
            # GWO execution
            gwo_t = GreyWolfOptimizer(prob_traj, best_params_gwo['pop_size'], MAX_ITER)
            _, _, _, g_hist = gwo_t.optimize(strategy=best_params_gwo['strategy'])
            g_xpl, g_xpt = calculate_xpl_xpt_numeric(g_hist)
            gwo_reps_xpl.append(g_xpl)
            gwo_reps_xpt.append(g_xpt)
            gwo_reps_hist.append(g_hist)
            
            # WOA execution
            woa_t = WhaleOptimizationAlgorithm(prob_traj, best_params_woa['pop_size'], MAX_ITER)
            _, _, _, w_hist = woa_t.optimize(b=best_params_woa['b'], 
                                            p_switch=best_params_woa['p'], 
                                            strategy=best_params_woa['strategy'])
            w_xpl, w_xpt = calculate_xpl_xpt_numeric(w_hist)
            woa_reps_xpl.append(w_xpl)
            woa_reps_xpt.append(w_xpt)
            woa_reps_hist.append(w_hist)

        # Store mean values across repetitions
        xpl_xpt_data.append({
            'Problem': name.capitalize(),
            'GWO XPL%': np.mean(gwo_reps_xpl),
            'GWO XPT%': np.mean(gwo_reps_xpt),
            'WOA XPL%': np.mean(woa_reps_xpl),
            'WOA XPT%': np.mean(woa_reps_xpt)
        })

        # Generate average trajectory plots
        z1_gwo_m, fit_gwo_m = get_mean_trajectory(prob_traj, gwo_reps_hist)
        z1_woa_m, fit_woa_m = get_mean_trajectory(prob_traj, woa_reps_hist)
        plot_mean_trajectory_comparison(z1_gwo_m, fit_gwo_m, z1_woa_m, fit_woa_m, name, 
                                        os.path.join(TRAJECTORY_DIR, f"mean_trajectory_{name}.png"))

    # Output balance results to terminal
    df_balance = pd.DataFrame(xpl_xpt_data)
    print("\nSummary of Exploration (XPL) vs Exploitation (XPT) - Means of 5 reps:")
    print(df_balance.to_string(index=False))
                                     

    print("All tasks finished successfully.")

if __name__ == "__main__":
    main()