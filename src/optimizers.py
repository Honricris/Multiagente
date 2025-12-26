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
        """Governs the exploration-exploitation balance"""
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

    def optimize(self, strategy='linear'):
        self._initialize()
        
        convergence_history = []
        pos_history = [] # Store population state at each iteration

        # Initialize leader positions and scores
        alpha_pos = np.zeros(self.dim)
        alpha_score = np.inf
        
        beta_pos = np.zeros(self.dim)
        beta_score = np.inf
        
        delta_pos = np.zeros(self.dim)
        delta_score = np.inf

        for t in range(self.max_iter):
            # Record current population for trajectory mapping
            pos_history.append(self.population.copy())

            for i in range(self.pop_size):
                # Apply search space boundaries
                self.population[i] = np.clip(self.population[i], self.bounds[0], self.bounds[1])
                
                # Evaluate fitness on transformed landscape
                fitness = self.problem.compute(self.population[i])
                
                # Update hierarchy based on fitness
                if fitness < alpha_score:
                    alpha_score, alpha_pos = fitness, self.population[i].copy()
                elif fitness < beta_score:
                    beta_score, beta_pos = fitness, self.population[i].copy()
                elif fitness < delta_score:
                    delta_score, delta_pos = fitness, self.population[i].copy()

            convergence_history.append(alpha_score)
            a = self._update_a(t, strategy)

            # Update wolf positions using Alpha, Beta, and Delta guidance
            r1 = np.random.random((self.pop_size, self.dim))
            r2 = np.random.random((self.pop_size, self.dim))
            
            # Move towards Alpha
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - self.population)
            X1 = alpha_pos - A1 * D_alpha
            
            # Move towards Beta
            r1 = np.random.random((self.pop_size, self.dim))
            r2 = np.random.random((self.pop_size, self.dim))
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - self.population)
            X2 = beta_pos - A2 * D_beta
            
            # Move towards Delta
            r1 = np.random.random((self.pop_size, self.dim))
            r2 = np.random.random((self.pop_size, self.dim))
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - self.population)
            X3 = delta_pos - A3 * D_delta
            
            # Calculate new positions as the average of leaders influence
            self.population = (X1 + X2 + X3) / 3
            # Enforce search boundaries immediately after update
            self.population = np.clip(self.population, self.bounds[0], self.bounds[1])

        return alpha_pos, alpha_score, convergence_history, pos_history


class WhaleOptimizationAlgorithm(SwarmOptimizer):
    """Full implementation of the Whale Optimization Algorithm"""

    def _update_a(self, t, strategy='linear'):
        """Calculates time-dependent decay coefficient a"""
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
        pos_history = [] # Store population state at each iteration
        
        best_pos = np.zeros(self.dim)
        best_score = np.inf

        for t in range(self.max_iter):
            # Enforce search boundaries and record positions
            self.population = np.clip(self.population, self.bounds[0], self.bounds[1])
            pos_history.append(self.population.copy())

            # Fitness evaluation loop
            current_fitness = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                current_fitness[i] = self.problem.compute(self.population[i])
            
            # Track global best solution
            min_idx = np.argmin(current_fitness)
            if current_fitness[min_idx] < best_score:
                best_score = current_fitness[min_idx]
                best_pos = self.population[min_idx].copy()

            convergence_history.append(best_score)
            a = self._update_a(t, strategy)

            # Parameter broadcasting for vectorized position updates
            r1 = np.random.random((self.pop_size, 1))
            r2 = np.random.random((self.pop_size, 1))
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.random((self.pop_size, 1))
            l = np.random.uniform(-1, 1, (self.pop_size, 1))

            # Define movement strategies via boolean masks
            mask_spiral = (p >= p_switch).flatten()
            mask_linear = (p < p_switch).flatten()
            mask_encircle = mask_linear & (np.abs(A).flatten() < 1)
            mask_search = mask_linear & (np.abs(A).flatten() >= 1)

            # Apply Spiral bubble-net attack
            if np.any(mask_spiral):
                dist = np.abs(best_pos - self.population[mask_spiral])
                spiral = dist * np.exp(b * l[mask_spiral]) * np.cos(2 * np.pi * l[mask_spiral])
                self.population[mask_spiral] = spiral + best_pos

            # Apply Encircling prey movement
            if np.any(mask_encircle):
                D = np.abs(C[mask_encircle] * best_pos - self.population[mask_encircle])
                self.population[mask_encircle] = best_pos - A[mask_encircle] * D

            # Apply Search for prey (random exploration)
            if np.any(mask_search):
                rand_idx = np.random.randint(0, self.pop_size, size=np.sum(mask_search))
                X_rand = self.population[rand_idx]
                D = np.abs(C[mask_search] * X_rand - self.population[mask_search])
                self.population[mask_search] = X_rand - A[mask_search] * D

        return best_pos, best_score, convergence_history, pos_history