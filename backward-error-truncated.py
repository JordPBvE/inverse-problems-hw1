import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Parameters and grid
n = 100
a = 100
delta = 0.001
x = np.linspace(0, 1, n)

# Define forward operator
c = np.exp(-a * x**2) / ((n - 1) * np.sqrt(np.pi / a))
K = la.toeplitz(c)

# Define functions
functions = {
    "Step Function": abs(x - 0.5) < 0.2,
    "Parabola": x * (1 - x)
}

# Truncated SVD parameters
ranks = [6, 9, 12, 14, 15, 16, 18]  # Different k values

# Compute SVD
U, S, Vt = np.linalg.svd(K)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (label, u) in zip(axes, functions.items()):
    for rank in ranks:
        # Truncate SVD
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vt_trunc = Vt[:rank, :]
        
        # Compute pseudo-inverse using truncated SVD
        S_inv_trunc = np.diag(1 / S_trunc)
        Kinv_trunc = Vt_trunc.T @ S_inv_trunc @ U_trunc.T
        
        fd = K @ u + delta * np.random.randn(n)
        ud = Kinv_trunc @ fd
        
        backward_error = np.linalg.norm(u - ud)
        ax.plot(x, ud, label=f"k={rank}, err={round(backward_error, 3)}")
    
    ax.plot(x, u, label="Original Function", linestyle='dashed')
    ax.set_xlabel("x")
    ax.set_ylabel("Value")
    ax.set_title(f"{label} (delta={delta})")
    ax.legend()

plt.show()
