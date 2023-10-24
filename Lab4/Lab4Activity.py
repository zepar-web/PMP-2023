import numpy as np
from scipy.stats import norm, expon, poisson

lambd = 20
X = poisson.rvs(lambd)

def estimate_alpha(X):
    def total_time(alpha):
        T_plasare_plata = norm.rvs(loc=2, scale=0.5, size=X)
        T_gatit = expon.rvs(scale=1/alpha, size=X)
        T_total = T_plasare_plata + T_gatit
        return T_total

    alpha = 10
    while True:
        T_total = total_time(alpha)
        success = (T_total < 15).all()/X
        if success >= 0.95:
            alpha -= 0.1
        else:
            return alpha

alpha_maxim = estimate_alpha(X)

T_plasare_plata = norm.rvs(loc=2, scale=0.5, size=X)
T_gatit = expon.rvs(scale=alpha_maxim, size=X)
T_total = T_plasare_plata + T_gatit

print("Numărul de clienți:", X)
print("α maxim pentru timp sub 15 minute:", alpha_maxim)
print("Timpul mediu de așteptare:", T_total.mean())
