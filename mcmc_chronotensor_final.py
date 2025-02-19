import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner

# Load the updated N-body data (positions and velocities)
nbody_data = np.load("NBody_Simulation_Chronotensor_Updated.npy")

# Extract physical properties (position + velocity)
positions, velocities = nbody_data[:, :3], nbody_data[:, 3:]

# Generate approximate curvature (R), density (rho), time adjacency (tau), and redshift (z)
R_values = np.random.uniform(0.1, 1.0, len(positions))
rho_values = np.random.uniform(0.1, 10.0, len(positions))
tau_values = np.random.uniform(0.01, 1.0, len(positions))
z_values = np.random.uniform(0.1, 2.0, len(positions))

# Define the suppression function S(R, rho, tau, z)
def suppression_function(params, R, rho, tau, z):
    alpha, beta, gamma, eta, delta = params
    return np.exp(-alpha * np.sqrt(R)) * (1 + beta * rho)**-eta * (1 / (1 + (tau * z)**eta)) * np.exp(-gamma * z / (1 + delta * z))

# Define the log-likelihood function
def log_likelihood(params, R, rho, tau, z, observed_velocities):
    model_suppression = suppression_function(params, R, rho, tau, z)
    model_velocities = model_suppression[:, np.newaxis] * observed_velocities
    residuals = observed_velocities - model_velocities
    return -0.5 * np.sum(residuals**2)

# Define the prior function (expanded parameter space)
def log_prior(params):
    alpha, beta, gamma, eta, delta = params
    if 0.01 < alpha < 0.2 and 0.0001 < beta < 0.1 and 0.01 < gamma < 0.5 and 1.0 < eta < 2.5 and 0.2 < delta < 3.0:
        return 0.0  # Flat prior within range
    return -np.inf  # Log-prior is -inf outside the range

# Define the full log-probability function
def log_probability(params, R, rho, tau, z, observed_velocities):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, R, rho, tau, z, observed_velocities)

# Set up MCMC sampler
ndim = 5  # Number of parameters (alpha, beta, gamma, eta, delta)
nwalkers = 50  # Number of walkers
nsteps = 5000  # Number of MCMC steps

# Initialize walkers randomly within new prior ranges
initial_pos = np.random.uniform([0.01, 0.0001, 0.01, 1.0, 0.2], [0.2, 0.1, 0.5, 2.5, 3.0], (nwalkers, ndim))

# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(R_values, rho_values, tau_values, z_values, velocities))

# Run MCMC
print("Running FINAL MCMC for robustness check...")
sampler.run_mcmc(initial_pos, nsteps, progress=True)
print("Final MCMC complete!")

# Extract samples (burn-in removed)
samples = sampler.get_chain(discard=1000, thin=10, flat=True)

# Save MCMC results
np.savetxt("mcmc_results_final.txt", samples)

# Generate corner plot
fig = corner.corner(samples, labels=[r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\eta$", r"$\delta$"], truths=[None, None, None, None, None])
plt.savefig("mcmc_corner_final.png")
plt.show()

print("Final MCMC results saved as 'mcmc_results_final.txt' and corner plot as 'mcmc_corner_final.png'")
