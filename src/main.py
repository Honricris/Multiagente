import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import os
import time
import optuna.visualization as vis 
from benchmarks import BenchmarkFactory
# Se añade WhaleOptimizationAlgorithm a la importación
from optimizers import GreyWolfOptimizer, WhaleOptimizationAlgorithm 
from stats_utils import compute_friedman_test, plot_critical_difference_diagram

# Global variables for experiment control
DIM = 10 
MAX_ITER = 50 
N_REPETITIONS = 5 
OPTUNA_TRIALS = 20 
SEED_TRANSFORM = 42 

# --- OPTIMIZATION RANGES (GLOBAL VARIABLES) ---
# GWO Parameters
GWO_POP_RANGE = (10, 80)

# WOA Parameters
WOA_POP_RANGE = (10, 80)
WOA_B_RANGE = (0.5, 2.0)      # Spiral shape constant range
WOA_P_RANGE = (0.4, 0.6)      # Mechanism probability switch range

# Discrete values for Parameter Sweeps
STRATEGIES = ['linear', 'exp', 'log', 'sin']
POP_SIZES = [10, 20, 40, 80, 160]
WOA_B_VALUES = [0.5, 1.0, 1.5, 2.0]
WOA_P_VALUES = [0.4, 0.5, 0.6]

# Success rate configuration
SUCCESS_THRESHOLD = 1e-2
KNOWN_OPTIMA = {
    'sphere': 0.0, 'rosenbrock': 0.0, 'rastrigin': 0.0, 'schwefel': 0.0,
    'ackley': 0.0, 'griewank': 0.0, 'michalewicz': -9.66015, 'zakharov': 0.0,
    'dixon_price': 0.0, 'levy': 0.0
}

# Directory structure
BASE_GRAPHICS_DIR_GWO = "graphics/GWO"
STRAT_OUTPUT_DIR_GWO = os.path.join(BASE_GRAPHICS_DIR_GWO, "Strategies")
POP_OUTPUT_DIR_GWO = os.path.join(BASE_GRAPHICS_DIR_GWO, "Population")

# New WOA Directories
BASE_GRAPHICS_DIR_WOA = "graphics/WOA"
STRAT_OUTPUT_DIR_WOA = os.path.join(BASE_GRAPHICS_DIR_WOA, "Strategies")
POP_OUTPUT_DIR_WOA = os.path.join(BASE_GRAPHICS_DIR_WOA, "Population")
B_OUTPUT_DIR_WOA = os.path.join(BASE_GRAPHICS_DIR_WOA, "Parameter_b")
P_OUTPUT_DIR_WOA = os.path.join(BASE_GRAPHICS_DIR_WOA, "Parameter_p")

# Benchmark problems to analyze
PROBLEM_NAMES = [
    'sphere', 'rosenbrock', 'rastrigin', 'schwefel', 'ackley', 
    'griewank', 'michalewicz', 'zakharov', 'dixon_price', 'levy'
]

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

def gwo_global_objective(trial):
    """
    Optuna objective function to optimize population size and strategy globally for GWO.
    """
    # Using Global Variable for Range
    pop_size = trial.suggest_int('pop_size', GWO_POP_RANGE[0], GWO_POP_RANGE[1])
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

def woa_global_objective(trial):
    """
    Optuna objective function to optimize parameters globally for WOA.
    """
    # Using Global Variables for Ranges
    pop_size = trial.suggest_int('pop_size', WOA_POP_RANGE[0], WOA_POP_RANGE[1])
    strategy = trial.suggest_categorical('strategy', STRATEGIES)
    b_val = trial.suggest_float('b', WOA_B_RANGE[0], WOA_B_RANGE[1])
    p_val = trial.suggest_float('p', WOA_P_RANGE[0], WOA_P_RANGE[1])
    
    scores = []
    
    for name in PROBLEM_NAMES:
        np.random.seed(SEED_TRANSFORM)
        problem = BenchmarkFactory.create(name, DIM)
        reps = []
        for r in range(3):
            np.random.seed(trial.number * 100 + r)
            woa = WhaleOptimizationAlgorithm(problem, pop_size, MAX_ITER)
            # Pass optimized parameters to WOA
            _, score, _ = woa.optimize(b=b_val, p_switch=p_val, strategy=strategy)
            reps.append(score)
        scores.append(np.mean(reps))
        
    return np.mean(scores)

def main():
    # Initialize output directory structure
    for d in [STRAT_OUTPUT_DIR_GWO, POP_OUTPUT_DIR_GWO, 
              STRAT_OUTPUT_DIR_WOA, POP_OUTPUT_DIR_WOA, 
              B_OUTPUT_DIR_WOA, P_OUTPUT_DIR_WOA]:
        os.makedirs(d, exist_ok=True)

    # ==========================================
    # PART 1: GREY WOLF OPTIMIZER (GWO) STUDY
    # ==========================================
    print("="*60)
    print("STARTING GWO ANALYSIS")
    print("="*60)

    # 1. OPTUNA PARAMETER TUNING (GWO)
    print("Step 1 (GWO): Finding optimal global parameters using Optuna")
    study_gwo = optuna.create_study(direction='minimize')
    study_gwo.optimize(gwo_global_objective, n_trials=OPTUNA_TRIALS)
    best_params_gwo = study_gwo.best_params
    print(f"Optimal GWO configuration: {best_params_gwo}")

    # Export parameter importance plot (GWO)
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
                _, score, history = gwo.optimize(strategy=strat)
                f_reps.append(score)
                h_reps.append(history)
            results_strat_gwo[strat].append(np.mean(f_reps))
            conv_strat_gwo[name][strat] = np.mean(h_reps, axis=0)
            raw_strat_data_gwo[name][strat] = f_reps

    # Save GWO Strategy metrics
    calculate_and_save_metrics(raw_strat_data_gwo, PROBLEM_NAMES, STRATEGIES, os.path.join(STRAT_OUTPUT_DIR_GWO, "strategy_metrics.csv"))

    # GWO Strategy Statistical Validation
    df_strat_gwo = pd.DataFrame(results_strat_gwo, index=PROBLEM_NAMES)
    f_stat_s, p_val_s, ranks_s = compute_friedman_test(df_strat_gwo)
    print(f"GWO Strategy Friedman Result: F = {f_stat_s:.4f}, p = {p_val_s:.4e}")

    if p_val_s < 0.05:
        print("Significant GWO strategy difference found. Generating CD Diagram")
        plot_critical_difference_diagram(ranks_s, len(PROBLEM_NAMES), output_dir=STRAT_OUTPUT_DIR_GWO)
    else:
        print("No significant GWO strategy differences detected")

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
                _, score, history = gwo.optimize(strategy=target_strat_gwo)
                t_reps.append(time.process_time() - start_t)
                f_reps.append(score)
                h_reps.append(history)
            
            results_pop_gwo[f"N_{n}"].append(np.mean(f_reps))
            times_pop_gwo[f"N_{n}"].append(np.mean(t_reps))
            conv_pop_gwo[name][f"N_{n}"] = np.mean(h_reps, axis=0)
            raw_pop_data_gwo[name][f"N_{n}"] = f_reps

    # Save GWO Population metrics
    calculate_and_save_metrics(raw_pop_data_gwo, PROBLEM_NAMES, [f"N_{n}" for n in POP_SIZES], os.path.join(POP_OUTPUT_DIR_GWO, "population_metrics.csv"))

    # GWO Population Statistical Validation
    df_pop_gwo = pd.DataFrame(results_pop_gwo, index=PROBLEM_NAMES)
    f_stat_p, p_val_p, ranks_p = compute_friedman_test(df_pop_gwo)
    print(f"GWO Population Friedman Result: F = {f_stat_p:.4f}, p = {p_val_p:.4e}")

    if p_val_p < 0.05:
        print("Significant GWO population difference found. Generating CD Diagram")
        plot_critical_difference_diagram(ranks_p, len(PROBLEM_NAMES), output_dir=POP_OUTPUT_DIR_GWO)

    # 4. GWO VISUALIZATIONS
    print("\nStep 4 (GWO): Generating plots")
    
    # Trade-off
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

    # Population Convergence
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

    # Strategy Convergence
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

    # Global Efficiency Plots
    # Pop
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

    # Strat
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


    # ==========================================
    # PART 2: WHALE OPTIMIZATION ALGORITHM (WOA) STUDY
    # ==========================================
    print("\n" + "="*60)
    print("STARTING WOA ANALYSIS")
    print("="*60)

    # 1. OPTUNA PARAMETER TUNING (WOA)
    print("Step 1 (WOA): Finding optimal global parameters using Optuna")
    study_woa = optuna.create_study(direction='minimize')
    study_woa.optimize(woa_global_objective, n_trials=OPTUNA_TRIALS)
    best_params_woa = study_woa.best_params
    print(f"Optimal WOA configuration: {best_params_woa}")

    # Export parameter importance plot (WOA)
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
                _, score, history = woa.optimize(b=best_params_woa['b'], 
                                                 p_switch=best_params_woa['p'], 
                                                 strategy=strat)
                f_reps.append(score)
                h_reps.append(history)
            results_strat_woa[strat].append(np.mean(f_reps))
            conv_strat_woa[name][strat] = np.mean(h_reps, axis=0)
            raw_strat_data_woa[name][strat] = f_reps

    # Save WOA Strategy metrics
    calculate_and_save_metrics(raw_strat_data_woa, PROBLEM_NAMES, STRATEGIES, os.path.join(STRAT_OUTPUT_DIR_WOA, "strategy_metrics.csv"))

    # WOA Strategy Statistical Validation
    df_strat_woa = pd.DataFrame(results_strat_woa, index=PROBLEM_NAMES)
    f_stat_s, p_val_s, ranks_s = compute_friedman_test(df_strat_woa)
    print(f"WOA Strategy Friedman Result: F = {f_stat_s:.4f}, p = {p_val_s:.4e}")

    if p_val_s < 0.05:
        print("Significant WOA strategy difference found. Generating CD Diagram")
        plot_critical_difference_diagram(ranks_s, len(PROBLEM_NAMES), output_dir=STRAT_OUTPUT_DIR_WOA)
    else:
        print("No significant WOA strategy differences detected")

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
                _, score, history = woa.optimize(b=best_params_woa['b'], 
                                                 p_switch=best_params_woa['p'], 
                                                 strategy=target_strat_woa)
                t_reps.append(time.process_time() - start_t)
                f_reps.append(score)
                h_reps.append(history)
            
            results_pop_woa[f"N_{n}"].append(np.mean(f_reps))
            times_pop_woa[f"N_{n}"].append(np.mean(t_reps))
            conv_pop_woa[name][f"N_{n}"] = np.mean(h_reps, axis=0)
            raw_pop_data_woa[name][f"N_{n}"] = f_reps

    # Save WOA Population metrics
    calculate_and_save_metrics(raw_pop_data_woa, PROBLEM_NAMES, [f"N_{n}" for n in POP_SIZES], os.path.join(POP_OUTPUT_DIR_WOA, "population_metrics.csv"))

    # WOA Population Statistical Validation
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
                # Fix others, vary b
                _, score, history = woa.optimize(b=b_val, 
                                                 p_switch=best_params_woa['p'], 
                                                 strategy=target_strat_woa)
                f_reps.append(score)
                h_reps.append(history)
            results_b_woa[f"b_{b_val}"].append(np.mean(f_reps))
            conv_b_woa[name][f"b_{b_val}"] = np.mean(h_reps, axis=0)
            raw_b_data_woa[name][f"b_{b_val}"] = f_reps

    # Save WOA Parameter b metrics
    calculate_and_save_metrics(raw_b_data_woa, PROBLEM_NAMES, [f"b_{v}" for v in WOA_B_VALUES], os.path.join(B_OUTPUT_DIR_WOA, "parameter_b_metrics.csv"))

    # Stats for b
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
                # Fix others, vary p
                _, score, history = woa.optimize(b=best_params_woa['b'], 
                                                 p_switch=p_val, 
                                                 strategy=target_strat_woa)
                f_reps.append(score)
                h_reps.append(history)
            results_p_woa[f"p_{p_val}"].append(np.mean(f_reps))
            conv_p_woa[name][f"p_{p_val}"] = np.mean(h_reps, axis=0)
            raw_p_data_woa[name][f"p_{p_val}"] = f_reps

    # Save WOA Parameter p metrics
    calculate_and_save_metrics(raw_p_data_woa, PROBLEM_NAMES, [f"p_{v}" for v in WOA_P_VALUES], os.path.join(P_OUTPUT_DIR_WOA, "parameter_p_metrics.csv"))

    # Stats for p
    df_p_woa = pd.DataFrame(results_p_woa, index=PROBLEM_NAMES)
    f_stat_p_val, p_val_p_val, ranks_p_val = compute_friedman_test(df_p_woa)
    print(f"WOA Parameter p Friedman Result: F = {f_stat_p_val:.4f}, p = {p_val_p_val:.4e}")
    if p_val_p_val < 0.05:
        plot_critical_difference_diagram(ranks_p_val, len(PROBLEM_NAMES), output_dir=P_OUTPUT_DIR_WOA)

    # 6. WOA VISUALIZATIONS
    print("\nStep 6 (WOA): Generating plots")
    
    # Trade-off
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

    # WOA Population Convergence
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

    # WOA Strategy Convergence
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

    # WOA Parameter b Convergence
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

    # WOA Parameter p Convergence
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

    # WOA Global Efficiency Plots
    # Pop
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

    # Strat
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

    # Parameter b
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

    # Parameter p
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

    print("All WOA tasks finished. Results saved in 'graphics/WOA/'")

if __name__ == "__main__":
    main()