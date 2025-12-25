import numpy as np

class SwarmOptimizer:
    """Base class for swarm intelligence algorithms"""
    
    def __init__(self, problem, pop_size, max_iter):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = problem.dimension
        self.bounds = problem.bounds
        self.population = None
        self.fitness = None

    def _initialize(self):
        """Generates random initial positions within bounds"""
        low, high = self.bounds
        self.population = np.random.uniform(low, high, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def optimize(self):
        raise NotImplementedError

class GreyWolfOptimizer(SwarmOptimizer):
    """Full implementation of the Grey Wolf Optimizer search strategy"""

    def _update_a(self, t, strategy='linear'):
        """
        Governs the exploration-exploitation balance.
        t: current iteration, T: max_iterations.
        """
        T = self.max_iter
        if strategy == 'linear':
            return 2 - 2 * (t / T)
        if strategy == 'exp':
            return 2 * np.exp(-3 * t / T)
        if strategy == 'log':
            # Logarithmic decay
            return 2 * np.log10(1 + (10 - 1) * (1 - t / T))
        if strategy == 'sin':
            # Sinusoidal decay
            return 2 * np.cos((np.pi / 2) * (t / T))
        return 0 

    def optimize(self, strategy='linear'):
        self._initialize()
        
        # History to store the best fitness at each iteration
        convergence_history = []

        # Initialize leaders
        alpha_pos = np.zeros(self.dim)
        alpha_score = np.inf
        
        beta_pos = np.zeros(self.dim)
        beta_score = np.inf
        
        delta_pos = np.zeros(self.dim)
        delta_score = np.inf

        for t in range(self.max_iter):
            # Update leaders based on current population fitness
            for i in range(self.pop_size):
                # Ensure agents stay within search space
                self.population[i] = np.clip(self.population[i], self.bounds[0], self.bounds[1])
                
                # Evaluate fitness on the transformed landscape
                fitness = self.problem.compute(self.population[i])
                
                # Update hierarchy: Alpha, Beta, Delta
                if fitness < alpha_score:
                    alpha_score, alpha_pos = fitness, self.population[i].copy()
                elif fitness < beta_score:
                    beta_score, beta_pos = fitness, self.population[i].copy()
                elif fitness < delta_score:
                    delta_score, delta_pos = fitness, self.population[i].copy()

            # Record the best fitness of current iteration
            convergence_history.append(alpha_score)
            a = self._update_a(t, strategy)

            # Update position of all search agents (omega wolves)
            # Vectorized calculation for efficiency
            r1 = np.random.random((self.pop_size, self.dim))
            r2 = np.random.random((self.pop_size, self.dim))
            
            # Calculate movement relative to Alpha
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - self.population)
            X1 = alpha_pos - A1 * D_alpha
            
            # Calculate movement relative to Beta
            r1 = np.random.random((self.pop_size, self.dim))
            r2 = np.random.random((self.pop_size, self.dim))
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - self.population)
            X2 = beta_pos - A2 * D_beta
            
            # Calculate movement relative to Delta
            r1 = np.random.random((self.pop_size, self.dim))
            r2 = np.random.random((self.pop_size, self.dim))
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - self.population)
            X3 = delta_pos - A3 * D_delta
            
            # Average movement from the three leaders
            self.population = (X1 + X2 + X3) / 3

        return alpha_pos, alpha_score, convergence_history


class WhaleOptimizationAlgorithm(SwarmOptimizer):
    def _update_a(self, t, strategy='linear'):
        """
        Calculates decay coefficient 'a'.
        """
        T = self.max_iter
        if strategy == 'linear':
            return 2 - 2 * (t / T)
        if strategy == 'exp':
            return 2 * np.exp(-3 * t / T)
        if strategy == 'log':
            return 2 * np.log10(1 + (10 - 1) * (1 - t / T))
        if strategy == 'sin':
            return 2 * np.cos((np.pi / 2) * (t / T))
        return 0

    def optimize(self, b=1, p_switch=0.5, strategy='linear'):
        self._initialize()
        convergence_history = []
        
        # Initialize best solution
        best_pos = np.zeros(self.dim)
        best_score = np.inf

        for t in range(self.max_iter):
            # 1. CLIP: Keep agents inside bounds
            self.population = np.clip(self.population, self.bounds[0], self.bounds[1])

            # 2. FITNESS EVALUATION
            current_fitness = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                current_fitness[i] = self.problem.compute(self.population[i])
            
            # Update global best
            min_fitness_idx = np.argmin(current_fitness)
            if current_fitness[min_fitness_idx] < best_score:
                best_score = current_fitness[min_fitness_idx]
                best_pos = self.population[min_fitness_idx].copy()

            convergence_history.append(best_score)

            # 3. POSITION UPDATE (Vectorized)
            a = self._update_a(t, strategy)

            # Generate random params for the whole population
            # Shape (N, 1) allows broadcasting across dimensions
            r1 = np.random.random((self.pop_size, 1))
            r2 = np.random.random((self.pop_size, 1))
            
            A = 2 * a * r1 - a
            C = 2 * r2
            
            p = np.random.random((self.pop_size, 1))
            l = np.random.uniform(-1, 1, (self.pop_size, 1))

            # --- LOGICAL MASKS (Who does what?) ---
            
            # Condition 1: Spiral (Bubble-net attack) -> p >= 0.5
            mask_spiral = (p >= p_switch).flatten()

            # Condition 2: Linear (Search or Encircle) -> p < 0.5
            mask_linear = (p < p_switch).flatten()
            
            # 2.1: Encircle (Exploitation) -> |A| < 1
            mask_encircle = mask_linear & (np.abs(A).flatten() < 1)
            
            # 2.2: Search (Exploration) -> |A| >= 1
            mask_search = mask_linear & (np.abs(A).flatten() >= 1)

            # --- APPLY MOVEMENTS ---

            # A) Spiral Update
            if np.any(mask_spiral):
                dist_to_best = np.abs(best_pos - self.population[mask_spiral])
                # Formula: D' * e^(bl) * cos(2pi*l) + best
                spiral_move = dist_to_best * np.exp(b * l[mask_spiral]) * np.cos(2 * np.pi * l[mask_spiral])
                self.population[mask_spiral] = spiral_move + best_pos

            # B) Encircling Update
            if np.any(mask_encircle):
                # Formula: best - A * |C * best - current|
                D = np.abs(C[mask_encircle] * best_pos - self.population[mask_encircle])
                self.population[mask_encircle] = best_pos - A[mask_encircle] * D

            # C) Search Update (Random exploration)
            if np.any(mask_search):
                count_search = np.sum(mask_search)
                # Pick random whales for reference
                rand_indices = np.random.randint(0, self.pop_size, size=count_search)
                X_rand = self.population[rand_indices]
                
                # Formula: X_rand - A * |C * X_rand - current|
                D = np.abs(C[mask_search] * X_rand - self.population[mask_search])
                self.population[mask_search] = X_rand - A[mask_search] * D

        return best_pos, best_score, convergence_history