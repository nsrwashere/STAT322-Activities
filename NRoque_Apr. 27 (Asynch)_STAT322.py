import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Sample data 
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Define the model
with pm.Model() as model:
  # Intercept and slope priors (weakly informative)
  alpha = pm.Normal("alpha", mu=0, sd=10)
  beta = pm.Normal("beta", mu=0, sd=10)

  # Linear regression with error term
  y_pred = alpha + beta * x
  epsilon = pm.Normal("epsilon", sd=1)
  y_obs = y_pred + epsilon

# Perform MCMC sampling (adjust number of samples as needed)
trace = pm.sample(1000)

# Extract posterior means 
alpha_mean = trace["alpha"].mean()
beta_mean = trace["beta"].mean()

# New data for prediction (optional)
x_new = np.linspace(min(x), max(x) + 1, 10)  # Extend range slightly

# Predictive distribution for new data points (optional)
if x_new is not None:
  y_pred_new = alpha_mean + beta_mean * x_new

# Plot results
plt.scatter(x, y)

# Plot regression line
if x_new is not None:
  plt.plot(x_new, alpha_mean + beta_mean * x_new, label="Posterior Mean")

plt.xlabel("x")
plt.ylabel("y")

if x_new is not None:
  plt.legend()

plt.show()
