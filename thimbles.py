import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from scipy.special import airy
from scipy.signal import argrelextrema
z_param = np.exp(1j * (-0.5) * np.pi)
x_min, x_max = -3, 3
y_min, y_max = -3, 3
grid_points = 300
x = np.linspace(x_min, x_max, grid_points)
y = np.linspace(y_min, y_max, grid_points)
X, Y = np.meshgrid(x, y)
X_complex = X + 1j * Y
def action(x):
    return 1j * ((x**3)/3 + z_param * x)
def d_action(x):
    return 1j * (x**2 + z_param)
saddle_1 = 1j * np.sqrt(z_param)
saddle_2 = -1j * np.sqrt(z_param)
def flow_stable(t, p): 
    z = p[0] + 1j * p[1]
    d = -np.conjugate(d_action(z))
    return [d.real, d.imag]
def flow_unstable(t, p):
    z = p[0] + 1j * p[1]
    d = np.conjugate(d_action(z))
    return [d.real, d.imag]
def get_start_points(saddle, radius=0.01):
    theta = np.linspace(0, 2*np.pi, 100)
    circle = saddle + radius * np.exp(1j * theta)
    vals = action(circle).real
    descent_idx = argrelextrema(vals, np.less)[0]
    ascent_idx = argrelextrema(vals, np.greater)[0]
    return circle[descent_idx], circle[ascent_idx]
plt.figure(figsize=(10, 8))
vals = action(X_complex).real
plt.contourf(X, Y, vals, levels=40, cmap='terrain', alpha=0.5)
saddles = [saddle_1, saddle_2]
for i, s in enumerate(saddles):
    starts_stable, starts_unstable = get_start_points(s, radius=0.01)
    for p_start in starts_stable:
        sol = solve_ivp(flow_stable, [0, 10], [p_start.real, p_start.imag], 
                        max_step=0.05, events=lambda t, y: np.linalg.norm(y) - 10)
        plt.plot(sol.y[0], sol.y[1], 'k-', linewidth=1.5, zorder=3)
    for p_start in starts_unstable:
        sol = solve_ivp(flow_unstable, [0, 10], [p_start.real, p_start.imag], 
                        max_step=0.05, events=lambda t, y: np.linalg.norm(y) - 10)
        plt.plot(sol.y[0], sol.y[1], 'r--', linewidth=1.5, zorder=2)
plt.plot(saddle_1.real, saddle_1.imag, 'o', color='orange', markeredgecolor='black', zorder=5, label='Saddle 1 (+)')
plt.plot(saddle_2.real, saddle_2.imag, 'o', color='black', markeredgecolor='white', zorder=5, label='Saddle 2 (-)')
plt.title(f"Lefschetz Thimbles for Airy Integral", fontsize=14)
plt.xlabel("Re(x)")
plt.ylabel("Im(x)")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.grid(alpha=0.3)
print(f"\n--- Thimble Integration for z = {z_param:.4f} ---")
target_saddle = saddle_1
starts_stable, _ = get_start_points(target_saddle, radius=0.01)
path_pieces = []
branches = []
for p_start in starts_stable:
    sol = solve_ivp(flow_stable, [0, 20], [p_start.real, p_start.imag], 
                    max_step=0.01,
                    events=lambda t, y: np.linalg.norm(y) - 8)
    
    branch_points = sol.y[0] + 1j * sol.y[1]
    branches.append(branch_points)
branches.sort(key=lambda b: b[-1].real)
left_branch = branches[0]
right_branch = branches[1]
full_path_x = np.concatenate([left_branch[::-1], [target_saddle], right_branch])
integrand = np.exp(action(full_path_x))
approx_val = (1 / (2 * np.pi)) * trapezoid(integrand, full_path_x)
true_ai, _, _, _ = airy(z_param)
print(f"Calculated Value: {approx_val:.6f}")
print(f"Scipy Reference:  {true_ai:.6f}")
error = abs(approx_val - true_ai)
print(f"Absolute Error:   {error:.2e}")
plt.plot(full_path_x.real, full_path_x.imag, 'c--', linewidth=3, alpha=0.6, label='Integration Path')
plt.legend()
plt.show()
def monte_carlo_integrate(N):
    mask = np.abs(full_path_x) < 10.0
    mid = len(full_path_x) // 2
    mask[mid] = True 
    safe_path = full_path_x[mask]
    if len(safe_path) < 2:
        return 0j, 0.0
    dz = np.diff(safe_path)
    ds = np.abs(dz)
    path_s = np.concatenate(([0], np.cumsum(ds)))
    total_length = path_s[-1]
    segment_tangents = dz / ds
    tangents = np.append(segment_tangents, segment_tangents[-1])
    s_samples = np.random.uniform(0, total_length, N)
    z_real = np.interp(s_samples, path_s, safe_path.real)
    z_imag = np.interp(s_samples, path_s, safe_path.imag)
    z_samples = z_real + 1j * z_imag
    t_real = np.interp(s_samples, path_s, tangents.real)
    t_imag = np.interp(s_samples, path_s, tangents.imag)
    t_samples = t_real + 1j * t_imag
    t_mag = np.abs(t_samples)
    t_samples = np.divide(t_samples, t_mag, where=t_mag!=0)
    S = action(z_samples)
    integrand = np.exp(S) * t_samples
    result = (total_length / N) * np.sum(integrand)
    result = result / (2 * np.pi)
    true_ai = airy(z_param)[0]
    error = abs(result - true_ai)
    return result, error
val, err = monte_carlo_integrate(1000000)
print(f"Result: {val:.6f}")
print(f"Error:  {err:.2e}")
