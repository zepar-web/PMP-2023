import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

alpha = [4, 4, 5, 5]
lambda_values = [3, 2, 2, 3]

probabilitati_servere = [0.25, 0.25, 0.30, 0.20]

lambda_latenta = 4

numar_simulari = 100000
timp_peste_3 = []
# Generăm aleator valori pentru timpul necesar servirii unui client pentru fiecare server
timp_servire = []
for i in range(numar_simulari):
    server_ales = np.random.choice([0, 1, 2, 3], p=probabilitati_servere)
    timp_gamma = np.random.gamma(alpha[server_ales], scale=1/lambda_values[server_ales])
    timp_latenta = np.random.exponential(scale=1/lambda_latenta)
    timp_total = timp_gamma + timp_latenta
    timp_servire.append(timp_total)
    if(timp_total > 3):
        timp_peste_3.append(timp_total)

# prob ca timpul de servire sa fie mai mare de 3ms
probabilitate_finala = len(timp_peste_3)/len(timp_servire)


print("Probabilitatea ca timpul de servire să fie mai mare de 3 milisecunde:", probabilitate_finala)

plt.figure(figsize=(8, 6))
plt.hist(timp_servire, bins=50, density=True, alpha=0.6, color='g', edgecolor='black')
plt.xlabel('Timp de Servire (milisecunde)')
plt.ylabel('Densitate')
plt.title('Densitatea distribuției lui X')
plt.show()
