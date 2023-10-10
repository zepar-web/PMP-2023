import random
import matplotlib.pyplot as plt

probabilitate_stea = 0.3

num_experimente = 100

rezultate_ss = []
rezultate_sb = []
rezultate_bs = []
rezultate_bb = []

for _ in range(num_experimente):
    rezultat_experiment = ''
    for _ in range(10):
        prima_monedă = random.choice(['s', 'b'])
        a_doua_monedă = random.choices(['s', 'b'], weights=[probabilitate_stea, 1 - probabilitate_stea])[0]

        rezultat_experiment += prima_monedă + a_doua_monedă

    rezultate_ss.append(rezultat_experiment.count('ss'))
    rezultate_sb.append(rezultat_experiment.count('sb'))
    rezultate_bs.append(rezultat_experiment.count('bs'))
    rezultate_bb.append(rezultat_experiment.count('bb'))

plt.pie([sum(rezultate_ss) / len(rezultate_ss), sum(rezultate_sb) / len(rezultate_sb),
            sum(rezultate_bs) / len(rezultate_bs), sum(rezultate_bb) / len(rezultate_bb)],
        labels=['SS', 'SB', 'BS', 'BB'], autopct='%1.1f%%')
plt.title('Probabilități de obținere a rezultatelor')
plt.show()



