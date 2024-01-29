import numpy as np
from scipy.stats import geom

# informatii din cerinta
p_X = 0.3
p_Y = 0.5
iterations = 10000

# simulam variabilele aleatoare repartizate Geometric
X = geom.rvs(p_X, size=iterations)
Y = geom.rvs(p_Y, size=iterations)

# calculam prob estimata
prob_estimate = np.mean(X > Y ** 2)
print("Aprox prob P(X > Y^2):", prob_estimate)

k = 30
prob_estimates = []


for _ in range(k):
    X = geom.rvs(p_X, size=iterations)
    Y = geom.rvs(p_Y, size=iterations)
    prob_estimate = np.mean(X > Y ** 2)
    prob_estimates.append(prob_estimate)

# calculam media si deviatia pe probabilitatile estimate
mean_estimate = np.mean(prob_estimates)
std_deviation = np.std(prob_estimates)

print("Med:", mean_estimate)
print("Dev std:", std_deviation)