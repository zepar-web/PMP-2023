import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

# Define Y and θ values
Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]
first_prob = 10

# Create a subplot grid
num_rows = len(Y_values)
num_cols = len(theta_values)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# Create the model
for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        with pm.Model() as model:  # Use pm.Model() as a context manager
            n = pm.Poisson("n", mu=10)
            y_obs = pm.Binomial(f"y_obs_{Y}{theta}", n=n, p=theta, observed=Y)
            trace = pm.sample(1000, tune=1000, cores=1, progressbar=False)
            az.plot_posterior(trace, var_names=["n"], point_estimate="mean", ax=axes[i, j])
            axes[i, j].set_title(f"Y={Y}, θ={theta}")

# Remove empty subplots
for i in range(num_rows):
    for j in range(num_cols):
        if i == 0 and j == 0:
            continue
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()
