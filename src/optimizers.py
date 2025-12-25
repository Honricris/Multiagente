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
        T = self.max_iter
        if strategy == 'linear':
            return 2 - 2 * (t / T)
        return 2 * (1 - t / T)

    def optimize(self, b=1, p_switch=0.5, strategy='linear'):
        self._initialize()
        convergence_history = []
        best_pos = np.zeros(self.dim)
        best_score = np.inf

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                self.population[i] = np.clip(self.population[i], self.bounds[0], self.bounds[1])
                fitness = self.problem.compute(self.population[i])
                
                if fitness < best_score:
                    best_score = fitness
                    best_pos = self.population[i].copy()

            convergence_history.append(best_score)
            a = self._update_a(t, strategy)
            
            for i in range(self.pop_size):
                r = np.random.random()
                A = 2 * a * r - a
                C = 2 * np.random.random()
                p = np.random.random()
                l = np.random.uniform(-1, 1)
                
                if p < p_switch:
                    if np.abs(A) < 1:
                        D = np.abs(C * best_pos - self.population[i])
                        self.population[i] = best_pos - A * D
                    else:
                        rand_idx = np.random.randint(0, self.pop_size)
                        x_rand = self.population[rand_idx]
                        D = np.abs(C * x_rand - self.population[i])
                        self.population[i] = x_rand - A * D
                else:
                    distance_to_best = np.abs(best_pos - self.population[i])
                    spiral = distance_to_best * np.exp(b * l) * np.cos(2 * np.pi * l)
                    self.population[i] = spiral + best_pos

        return best_pos, best_score, convergence_history