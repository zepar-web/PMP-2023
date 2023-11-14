import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az


def main():
    data = pd.read_csv('auto-mpg.csv')

    data = data[data['horsepower'] != '?']
    data = data[data['mpg'] != '?']

    print(data.info)
    data['horsepower'] = pd.to_numeric(data['horsepower'])
    data['mpg'] = pd.to_numeric(data['mpg'])

    plt.scatter(data['horsepower'], data['mpg'])
    plt.title('Relatia dintre cai putere si mpg')
    plt.xlabel('Cai putere')
    plt.ylabel('Consum MPG')
    plt.show()
    # -------

    x = np.array(data['horsepower'])
    y = data['mpg']

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)

        # Modelul liniar
        mu = alpha + beta * x

        # Variabilitatea datelor
        sigma = pm.HalfNormal('sigma', sigma=10)

        # Definirea distribuției pentru variabila dependentă
        mpg = pm.Normal('mpg', mu=mu, sigma=sigma, observed=y)

        # Sampling
        trace = pm.sample(100, tune=100)


if __name__ == '__main__':
    main()


