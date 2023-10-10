import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

lambda1 = 4
lambda2 = 6

p = 0.4

nr_simulari = 1000

X = np.zeros(nr_simulari)
for i in range(nr_simulari):
    u = np.random.uniform()
    if u < p:
        X[i] = expon.rvs(scale=1/lambda1)
    else:
        X[i] = expon.rvs(scale=1/lambda2)

media = np.mean(X)
deviatia = np.std(X)

print("Media lui X este:", media)
print("Deviația standard a lui X este:", deviatia)


plt.hist(X, bins=50, density=True)
plt.xlabel("Timpul de servire (ore)")
plt.ylabel("Densitatea")
plt.title("Distribuția timpului de servire pentru un client")
plt.show()
