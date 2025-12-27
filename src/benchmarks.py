import numpy as np
from utils import generate_shift, generate_rotation, apply_transform

class BenchmarkFunction:
    """Base class for optimization benchmark functions."""
    def __init__(self, dimension, bounds, shift=True, rotate=True):
        self.dimension = dimension
        self.bounds = bounds
        self.use_shift = shift
        self.use_rotate = rotate
        self.shift_vector = None
        self.rotation_matrix = None
        self._setup_transformations()

    def _setup_transformations(self):
        # Initialize shift vector
        if self.use_shift:
            self.shift_vector = generate_shift(self.dimension, self.bounds)
        else:
            self.shift_vector = np.zeros(self.dimension)
            
        # Initialize rotation matrix
        if self.use_rotate:
            self.rotation_matrix = generate_rotation(self.dimension)
        else:
            self.rotation_matrix = np.eye(self.dimension)

    def _transform(self, x):
        """Applies shift and rotation to the input vector."""
        return apply_transform(x, self.shift_vector, self.rotation_matrix)

    def compute(self, x):
        """Main method to evaluate the function at point x."""
        z = self._transform(x)
        return self._objective(z)

    def _objective(self, z):
        raise NotImplementedError("This method must be implemented by the subclass.")


class Sphere(BenchmarkFunction):
    def _objective(self, z):
        return np.sum(z**2)

class Rosenbrock(BenchmarkFunction):  
    def _objective(self, z):
        return np.sum(100 * (z[1:] - z[:-1]**2)**2 + (1 - z[:-1])**2)
    
class Rastrigin(BenchmarkFunction):
    def _objective(self, z):
        n = len(z)
        return 10 * n + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

class Schwefel(BenchmarkFunction):
    def _objective(self, z):
        n = len(z)
        return 418.9829 * n + np.sum(-z * np.sin(np.sqrt(np.abs(z))))

class Ackley(BenchmarkFunction):
    def _objective(self, z):
        n = len(z)
        a, b, c = 20, 0.2, 2 * np.pi
        s1 = np.sum(z**2)
        s2 = np.sum(np.cos(c * z))
        return -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.exp(1)

class Griewank(BenchmarkFunction):
    def _objective(self, z):
        indices = np.arange(1, len(z) + 1)
        s = np.sum(z**2) / 4000
        p = np.prod(np.cos(z / np.sqrt(indices)))
        return s - p + 1

class Michalewicz(BenchmarkFunction):
    def _objective(self, z):
        m = 10
        indices = np.arange(1, len(z) + 1)
        s = np.sum(np.sin(z) * (np.sin(indices * z**2 / np.pi))**(2 * m))
        return -s

class Zakharov(BenchmarkFunction):
    def _objective(self, z):
        indices = np.arange(1, len(z) + 1)
        s1 = np.sum(z**2)
        s2 = np.sum(0.5 * indices * z)
        return s1 + s2**2 + s2**4

class DixonPrice(BenchmarkFunction):
    def _objective(self, z):
        indices = np.arange(2, len(z) + 1)
        term1 = (z[0] - 1)**2
        term2 = np.sum(indices * (2 * z[1:]**2 - z[:-1])**2)
        return term1 + term2

class Levy(BenchmarkFunction):
    def _objective(self, z):
        w = 1 + (z - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        return term1 + term2 + term3


class BenchmarkFactory:
    """Factory class to instantiate benchmark functions by name."""
    @staticmethod
    def create(name, dimension):
        config = {
            'sphere': (Sphere, [-5.12, 5.12]),
            'rosenbrock': (Rosenbrock, [-5.0, 10.0]),
            'rastrigin': (Rastrigin, [-5.12, 5.12]),
            'schwefel': (Schwefel, [-500.0, 500.0]),
            'ackley': (Ackley, [-32.768, 32.768]),
            'griewank': (Griewank, [-600.0, 600.0]),
            'michalewicz': (Michalewicz, [0.0, np.pi]),
            'zakharov': (Zakharov, [-5.0, 10.0]),
            'dixon_price': (DixonPrice, [-10.0, 10.0]),
            'levy': (Levy, [-10.0, 10.0])
        }
        
        name = name.lower()
        if name not in config:
            raise ValueError(f"Function {name} not found")
            
        cls, bounds = config[name]
        return cls(dimension, bounds)