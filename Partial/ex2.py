import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

if __name__ == "__main__":
    miu = 10
    alpha = 2

    # Generam 200 de timpi de asteptare cu miu si alpha
    timpi_medii_asteptare = np.random.normal(miu, alpha, 200)

    # Vizualizăm primele câteva valori generate
    print(timpi_medii_asteptare[:10])  # Afisez primele 10 valori din lista generată

    plt.hist(timpi_medii_asteptare)
    plt.show()

    observed_data = timpi_medii_asteptare  # Timpii medii de așteptare generate în pasul anterior

    # Definim modelul în PyMC
    with pm.Model() as model:
        # Alegem distribuțiile a priori pentru parametrii μ și α
        mu_prior = pm.Normal('mu', mu=10, sigma=5)  # Distribuție normală pentru μ
        alpha_prior = pm.HalfNormal('alpha', sigma=2)  # Distribuție exponențială pentru α

        # Definim distribuția a posteriori folosind distribuția observată
        likelihood = pm.Normal('likelihood', mu=mu_prior, sigma=alpha_prior, observed=observed_data)


    with model:
        trace = pm.sample(200, tune=100, cores=1)


    az.plot_trace(trace)
    plt.show()

    az.plot_posterior(trace['mu_prior'])
    plt.show()

    az.plot_posterior(trace['alpha_prior'])
    plt.show()
