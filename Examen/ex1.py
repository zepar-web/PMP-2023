import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt
def ex1():
    '''Punctul A'''

    # incarcam csv-ul
    df = pd.read_csv("Titanic.csv")
    df = df.drop(["PassengerId", "Name", "Sex", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

    # stergem randurile care au valori lipsa
    df = df.dropna()

    survived_out = df["Survived"].values

    Pclass = df["Pclass"].values
    age = df["Age"].values
    age_mean = age.mean()
    age_std = age.std()

    # standardizam datele pentru age
    age = (age - age_mean) / age_std

    X = np.column_stack((age, Pclass))
    X_mean = X.mean(axis=0, keepdims=True)

    print(f"Media var independente : {X_mean}")
    print(f"Media output {survived_out.mean()}")
    print(f"Dev standard {X.std(axis=0, keepdims=True)}")
    print(f"Output dev standard : {survived_out.std()}")

    '''Punctul B'''
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1, shape=2)
        X_shared = pm.MutableData('x_shared', X)
        miu = pm.Deterministic('miu', alpha + pm.math.dot(X_shared, beta))
        theta = pm.Deterministic('theta', pm.math.sigmoid(miu))

        bd = pm.Deterministic('bd', -alpha / beta[1] + beta[0] / beta[1] * X_shared[:, 0].mean())

        y_pred = pm.Bernoulli('y_pred', p=theta, observed=survived_out)

        idata = pm.sample(1250, return_inferencedata=True)

    '''Punctul C'''
    # verificam care variabila are cea mai mare influenta pentru output
    az.plot_forest(idata, hdi_prob=0.95, var_names=['beta'])
    plt.show()
    print(az.summary(idata, hdi_prob=0.95, var_names=['beta']))
    # se poate observa ca beta[1](Pclass) are o influenta mai mare asupra output-ului

    '''Punctul D'''
    # persoana de 30 de ani, de la clasa a2-a
    obs_std2 = [(30 - age_mean) / age_mean, 2]
    pm.set_data({"x_shared": [obs_std2]}, model=model)

    pos_pred = pm.sample_posterior_predictive(idata, model=model, var_names=["theta"])
    y_pos_pred = pos_pred.posterior_predictive['theta'].stack(sample=("chain", "draw")).values
    # construim intervalul de incredere de 90%, daca va supravietui sau nu
    az.plot_posterior(y_pos_pred, hdi_prob=0.9)
    plt.show()


if __name__ == '__main__':
    ex1();








