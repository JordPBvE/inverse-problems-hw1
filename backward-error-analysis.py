import numpy as np
import scipy.linalg as la

# Parameters and grid
n = 100
a = 100
x = np.linspace(0, 1, n)

# Define forward operator
c = np.exp(-a * x**2) / ((n - 1) * np.sqrt(np.pi / a))
K = la.toeplitz(c)

# Singular Value Decomposition (SVD) for pseudoinverse computation
U, S, Vt = np.linalg.svd(K)
Kinv = Vt.T @ np.diag(1 / S) @ U.T

# Define functions
functions = {
    "Step Function": abs(x - 0.5) < 0.2,
    "Parabola": x * (1 - x)
}

# Delta values
delta_values = [0.1, 0.01, 0.001]

# Compute backward error for each function and delta
results = []

for label, u in functions.items():
    f = K @ u  # Forward transform
    
    for delta in delta_values:
        fd = K @ u + delta * np.random.randn(n)  # Noisy data
        ud = Kinv @ fd  # Reconstruction
        backward_error = np.linalg.norm(u - ud)  # Compute error
        
        results.append((label, delta, backward_error))

# Print results as a table
print(f"{'Function':<15}{'Delta':<10}{'Backward Error'}")
print("-" * 40)
for label, delta, error in results:
    print(f"{label:<15}{delta:<10.3g}{error:.3e}")
