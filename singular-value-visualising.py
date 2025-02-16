import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Parameters and grid
n = 100
a = 100
x = np.linspace(0, 1, n)

# Define forward operator
c = np.exp(-a * x**2) / ((n - 1) * np.sqrt(np.pi / a))
K = la.toeplitz(c)

# Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(K)

# Compute inverse singular values
inverse_S = 1 / S

# Plot first 20 inverse singular values
plt.figure(figsize=(8, 5))
plt.plot(inverse_S[:20], 'o-', label='1/S')

# Mark specific points and draw lines to axes
indices = [7, 13, 15]
for k in indices:
    plt.axhline(y=inverse_S[k], xmin=0, xmax=k/20, color='gray', linestyle='--')
    plt.scatter(k, inverse_S[k], color='red', zorder=3)

plt.xlabel('Index')
plt.ylabel('Inverse Singular Value')
plt.yscale('log')
plt.title('First 20 Inverse Singular Values of K')
plt.legend()
plt.grid()
plt.show()
