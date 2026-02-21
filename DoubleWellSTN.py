import jax
import jax.numpy as jnp
from jax import random, jacfwd, jit
import numpy as np
N = 16            
a = 0.5           
lam = 1.0         
v = 1.0           
t_sep = N // 2    
def action(z):
    z_next = jnp.roll(z, -1)
    kinetic = 0.5 * ((z_next - z) / a)**2
    potential = lam * (z**2 - v**2)**2
    return jnp.sum(kinetic + potential) * a
def contour(x, alpha):
    y = alpha * jnp.sin(x) 
    return x + 1j * y
@jax.jit
def get_integrand_components(x, alpha):
    z = contour(x, alpha)
    S_z = action(z)
    J = jacfwd(contour)(x, alpha)
    detJ = jnp.linalg.det(J)
    S_eff = jnp.real(S_z) - jnp.log(jnp.abs(detJ))
    phase = jnp.exp(-1j * jnp.imag(S_z)) * (detJ / jnp.abs(detJ))
    return S_eff, phase, z
def run_mcmc(alpha, num_samples, key):
    x_current = jnp.zeros(N) 
    S_eff_current, phase_current, z_current = get_integrand_components(x_current, alpha)
    samples_z = []
    phases = []
    accepted = 0
    for i in range(num_samples):
        key, subkey, accept_key = random.split(key, 3)
        x_proposed = x_current + 0.5 * random.normal(subkey, shape=(N,))
        S_eff_proposed, phase_proposed, z_proposed = get_integrand_components(x_proposed, alpha)
        delta_S = S_eff_proposed - S_eff_current
        if delta_S < 0 or jnp.exp(-delta_S) > random.uniform(accept_key):
            x_current = x_proposed
            S_eff_current = S_eff_proposed
            phase_current = phase_proposed
            z_current = z_proposed
            accepted += 1
        if i % 10 == 0: 
            samples_z.append(z_current)
            phases.append(phase_current)
        if i > 0 and i % 2000 == 0:
            print(f"   Completed {i}/{num_samples} steps...")
    print(f"Acceptance Rate: {accepted / num_samples:.2f}")
    return jnp.array(samples_z), jnp.array(phases)
key = random.PRNGKey(42)
alpha_test = 0.2
num_steps = 10000
print(f"Running MCMC for contour parameter alpha = {alpha_test}...")
z_samples, phase_samples = run_mcmc(alpha_test, num_steps, key)
O_samples = jnp.mean(z_samples * jnp.roll(z_samples, -t_sep, axis=1), axis=1)
numerator = jnp.mean(O_samples * phase_samples)
denominator = jnp.mean(phase_samples)
signal = jnp.abs(numerator / denominator)
noise_squared = jnp.mean(jnp.abs(O_samples)**2) - jnp.abs(numerator)**2
noise = jnp.sqrt(noise_squared)
stn_ratio = signal / noise
print("\n--- Results ---")
print(f"Signal (Estimate of C(t)): {signal:.6f}")
print(f"Noise (Standard Deviation): {noise:.6f}")
print(f"Signal-to-Noise Ratio: {stn_ratio:.6f}")
