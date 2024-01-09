import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def posterior_grid_compare(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = (grid <= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

def posterior_grid_abs_diff(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = np.abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

grid_compare, posterior_compare = posterior_grid_compare()
grid_abs_diff, posterior_abs_diff = posterior_grid_abs_diff()

plt.figure(figsize=(12, 6))

# comparare cu 0.5
plt.subplot(1, 2, 1)
plt.plot(grid_compare, posterior_compare)
plt.title('Prior bazat pe comparare cu 0.5')
plt.xlabel('Valoare pe grid')
plt.ylabel('Probabilitate posteriora')

# diferența față de 0.5
plt.subplot(1, 2, 2)
plt.plot(grid_abs_diff, posterior_abs_diff)
plt.title('Prior bazat pe diferența absoluta fata de 0.5')
plt.xlabel('Valoare pe grid')
plt.ylabel('Probabilitate posteriora')

plt.tight_layout()
plt.show()
