# Import necessary libraries
import pandas as pd
import numpy as np
import pymc as pm
import csv

# Load and preprocess data
def load_traffic_data(file_name):
    with open(file_name, "r") as f:
        data = csv.reader(f)
        header = next(data)
        rows = [int(row[1]) for row in data]
    return np.array(rows)

def main():
    # Load traffic data from a CSV file
    traffic = load_traffic_data("trafic.csv")

    # Define parameters
    num_intervals = 5
    interval_lengths = [60 * (end - start) for start, end in [(4, 7), (4, 8), (4, 16), (4, 19)]]

    # Create a Bayesian model
    with pm.Model() as traffic_model:
        alpha = 1.0 / traffic.mean()
        lambdas = [pm.Exponential(f"lambda_{i}", alpha) for i in range(num_intervals)]
        index = np.arange(1200)
        l = lambdas[0]
        for i in range(1, num_intervals):
            l = pm.math.switch(interval_lengths[i - 1] < index, lambdas[i], l)

        # Define the observation model
        observation = pm.Poisson("obs", l, observed=traffic)

        # Perform Bayesian inference
        step = pm.Metropolis()
        trace = pm.sample(100, tune=100, step=step, return_inferencedata=False, cores=1)

    # Extract and print results
    lambdas_samples = [trace[f'lambda_{i}'] for i in range(num_intervals)]
    for i, lambda_samples in enumerate(lambdas_samples):
        print(f"Estimated lambda_{i + 1} mean: {lambda_samples.mean()}")

if __name__ == "__main__":
    main()
