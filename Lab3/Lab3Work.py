from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# 1. Definiți variabilele aleatoare
model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Incendiu', 'Alarma'), ('Cutremur', 'Alarma')])

# Probabilitatea cutremurului
cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])

# Probabilitatea incendiului în funcție de cutremur
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.99, 0.03], [0.01, 0.97]],
                          evidence=['Cutremur'], evidence_card=[2])

# Probabilitatea alarmei în funcție de incendiu și cutremur
cpd_alarmă = TabularCPD(variable='Alarma', variable_card=2, values=[[0.05, 0.02, 0.999, 0.98],
                                                                    [0.95, 0.98, 0.001, 0.02]],
                        evidence=['Incendiu', 'Cutremur'], evidence_card=[2, 2])


model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarmă)

# Verificați dacă modelul este valid și coerent
assert model.check_model()

inference = VariableElimination(model)
result = inference.query(variables=['Cutremur'], evidence={'Alarma': 1})
print(result)

result = inference.query(variables=['Incendiu'], evidence={'Alarma': 0})
print(result)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()