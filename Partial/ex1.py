import random
import numpy as np
from scipy import stats
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# 1 stema
# 0 cap

j0_castiga = 0
j1_castiga = 0

for i in range(20000):
    j0 = 0
    j1 = 0

    moneda = random.choice([0, 1])  # aruncam o moneda normala sa vedem cine incepe
    # 0 ---> incepe primul jucator j0
    # 1 ---> incepe al doilea jucator j1

    if moneda == 0:
        j0 = 1
    else:
        j1 = 1

    if j0 == 1:
        stema_m1 = stats.binom.rvs(1, 1/3)  # arunca cu prob 1/3 sa nimereasca stema
    elif j1 == 1:
        stema_m1 = stats.binom.rvs(1, 0.5)  # arunca cu prob de 1/2 sa nimereasca stema

    if stema_m1 == 1:
        n = 1
    else:
        n = 0

    m = 0
    if j0 == 1:
        stema_m2 = stats.binom.rvs(1, 1/3, size=n + 1)  # a doua aruncare pentru primul jucator
    elif j1 == 1:
        stema_m2 = stats.binom.rvs(1, 0.5, size=n + 1)  # a doua aruncare pentru al doilea jucator

    # numaram numarul de steme
    for i in range(n+1):
        if stema_m2[i] == 1:
            m += 1

    if n > m:  # daca n>m castiga jucatorul din prima runda
        if j0 == 1:
            j0_castiga += 1  # castiga primul jucator
        elif j1 == 1:
            j1_castiga += 1  # castiga al doilea jucator

# sansele de castig pentru j0
sanse_castig_j0 = j0_castiga/20000 * 100
# sansele de castig pentru j1
sanse_castig_j1 = j1_castiga/20000 * 100

print(f"Primul jucator castiga cu sansele {sanse_castig_j0}%")
print(f"Al doilea jucator jucator castiga cu sansele {sanse_castig_j1}%")

if sanse_castig_j0 > sanse_castig_j1:
    print("Primul jucator are sanse mai mari sa castige")
else:
    print("Al doilea jucator are sanse mai mari sa castige")
model = BayesianNetwork([('j', 'r1'),
                         ('j', 'r2'),
                         ('r1', 'r2')  # a2-a runda este conditionata de prima
                         ])

cpd_j = TabularCPD('j', 2, [[0.5], [0.5]])

cpd_r1 = TabularCPD('r1', 2, [[1 / 3, 0.5], [2 / 3, 0.5]], evidence=['j'],evidence_card=[2])

cpd_r2 = TabularCPD('r2', 2, [[1 / 3, 2 / 3, 0.5, 0.5], [2 / 3, 1 / 3, 0.5, 0.5]],
                               evidence=['r1', 'j'], evidence_card=[2, 2])

model.add_cpds(cpd_j, cpd_r1, cpd_r2)

# verificam daca modelul este valid
model.check_model()


inferenta = VariableElimination(model)

#punctul 3
turn1 = inferenta.query(variables=['r1'], evidence={'r2': 0})

# probabilitate fata prima runda, stiind ca nr de steme in a 2-a rund este 0
prob_ban = turn1.values[0]
prob_stema = turn1.values[1]

if prob_ban > prob_stema:
    print("Mai probabil sa obtinem ban, stiind ca in a-a am obtinut stema")
else:
    print("Mai probabil sa obtinem stema, stiind ca in a-a am obtinut stema")
