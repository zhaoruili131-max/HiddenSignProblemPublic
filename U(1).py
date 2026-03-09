"""
Monte Carlo reproduction of the Abelian U(1) example from

    Detmold, Kanwar, Wagman, Warrington,
    "Path integral contour deformations for noisy observables",
    Phys. Rev. D 102, 014514 (2020)

This script samples the ORIGINAL positive probability measure
    p(theta) ∝ exp(beta cos(theta))
and compares:

1) the original Wilson-loop estimator
       W_A = prod_{x in A} exp(i theta_x)

2) the deformed-observable estimator on the SAME ensemble
       X_A = exp(-(S(tilde theta)-S(theta))) * W_A(tilde theta)
with the one-parameter deformation used in the paper
       tilde theta_x = theta_x + i delta   for x in A
                      theta_x             otherwise

In the plaquette-angle representation of the 2D U(1) toy model,
the theta_x are independent, so exact sampling is available via a
von Mises distribution. That makes the code short and directly runnable.

Outputs:
- results CSV
- sigma_eff plot
- StN plot
- a short terminal summary

Example:
    python deformed_observable_u1_mc.py --beta 5.555 --delta 0.2 --samples 10000 --max-area 1000
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.special import iv
except Exception as exc:
    raise SystemExit from exc
@dataclass
class Curves:
    area: np.ndarray
    sigma_a: np.ndarray

    mean_orig: np.ndarray
    mean_deform: np.ndarray

    sigma_eff_orig: np.ndarray
    sigma_eff_deform: np.ndarray

    stn_orig: np.ndarray
    stn_deform: np.ndarray

    exact_mean: np.ndarray
    exact_stn_orig: np.ndarray
    exact_stn_deform: np.ndarray

    sigma_exact: float


def local_deformed_factor(theta: np.ndarray, beta: float, delta: float) -> np.ndarray:

    cos_tilde = np.cos(theta + 1j * delta)
    return np.exp(beta * (cos_tilde - np.cos(theta)) - delta + 1j * theta)
def compute_exact_curves(beta: float, delta: float, max_area: int, grid_size: int = 20001):
    area = np.arange(1, max_area + 1, dtype=np.int64)

    i0 = iv(0, beta)
    i1 = iv(1, beta)
    i2 = iv(2, beta)
    sigma = float(np.log(i0 / i1))
    sigma0 = float(np.log(i0 / i2))
    
    m1_orig = i1 / i0
    m2_orig = i2 / i0
    mabs2_orig = 1.0

    exact_mean = np.power(m1_orig, area)
    exact_var_orig = 0.5 * (np.power(mabs2_orig, area) + np.real(np.power(m2_orig, area))) - np.square(np.real(exact_mean))
    exact_stn_orig = np.abs(np.real(exact_mean)) / np.sqrt(np.maximum(exact_var_orig, 1e-300))
    
    theta = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    p = np.exp(beta * np.cos(theta)) / (2 * np.pi * i0)
    x = local_deformed_factor(theta, beta, delta)
    
    dtheta = theta[1] - theta[0]
    m1_def = np.sum(p * x) * dtheta
    m2_def = np.sum(p * x * x) * dtheta
    mabs2_def = np.sum(p * np.abs(x) ** 2) * dtheta

    exact_mean_def = np.power(m1_def, area)
    exact_var_def = 0.5 * (np.power(mabs2_def, area) + np.real(np.power(m2_def, area))) - np.square(np.real(exact_mean_def))
    exact_stn_def = np.abs(np.real(exact_mean_def)) / np.sqrt(np.maximum(exact_var_def, 1e-300))

    return sigma, exact_mean, exact_stn_orig, exact_stn_def
def run_mc(beta: float, delta: float, n_samples: int, max_area: int, seed: int) -> Curves:
    rng = np.random.default_rng(seed)
    area = np.arange(1, max_area + 1, dtype=np.int64)

    sigma_exact, exact_mean, exact_stn_orig, exact_stn_deform = compute_exact_curves(beta, delta, max_area)

    # Streaming accumulation: only keep the current partial products.
    w_prod = np.ones(n_samples, dtype=np.complex128)
    x_prod = np.ones(n_samples, dtype=np.complex128)

    mean_orig = np.empty(max_area, dtype=np.float64)
    mean_deform = np.empty(max_area, dtype=np.float64)
    stn_orig = np.empty(max_area, dtype=np.float64)
    stn_deform = np.empty(max_area, dtype=np.float64)

    for a in range(max_area):
        theta = rng.vonmises(mu=0.0, kappa=beta, size=n_samples)

        local_w = np.exp(1j * theta)
        local_x = local_deformed_factor(theta, beta, delta)

        w_prod *= local_w
        x_prod *= local_x

        re_w = np.real(w_prod)
        re_x = np.real(x_prod)

        mean_orig[a] = np.mean(re_w)
        mean_deform[a] = np.mean(re_x)

        var_w = np.var(re_w, ddof=1)
        var_x = np.var(re_x, ddof=1)

        stn_orig[a] = abs(mean_orig[a]) / np.sqrt(max(var_w, 1e-300))
        stn_deform[a] = abs(mean_deform[a]) / np.sqrt(max(var_x, 1e-300))

    sigma_eff_orig = np.empty(max_area, dtype=np.float64)
    sigma_eff_deform = np.empty(max_area, dtype=np.float64)

    sigma_eff_orig[0] = -np.log(max(mean_orig[0], 1e-300))
    sigma_eff_deform[0] = -np.log(max(mean_deform[0], 1e-300))

    ratio_orig = np.clip(mean_orig[1:] / np.maximum(mean_orig[:-1], 1e-300), 1e-300, None)
    ratio_def = np.clip(mean_deform[1:] / np.maximum(mean_deform[:-1], 1e-300), 1e-300, None)

    sigma_eff_orig[1:] = -np.log(ratio_orig)
    sigma_eff_deform[1:] = -np.log(ratio_def)

    return Curves(
        area=area,
        sigma_a=sigma_exact * area,
        mean_orig=mean_orig,
        mean_deform=mean_deform,
        sigma_eff_orig=sigma_eff_orig,
        sigma_eff_deform=sigma_eff_deform,
        stn_orig=stn_orig,
        stn_deform=stn_deform,
        exact_mean=exact_mean,
        exact_stn_orig=exact_stn_orig,
        exact_stn_deform=exact_stn_deform,
        sigma_exact=sigma_exact,
    )


def save_csv(curves: Curves, out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "A",
                "sigma_times_A",
                "mean_original",
                "mean_deformed",
                "sigma_eff_original",
                "sigma_eff_deformed",
                "stn_original",
                "stn_deformed",
                "exact_mean",
                "exact_stn_original",
                "exact_stn_deformed",
                "sigma_exact",
            ]
        )
        for i in range(len(curves.area)):
            writer.writerow(
                [
                    int(curves.area[i]),
                    float(curves.sigma_a[i]),
                    float(curves.mean_orig[i]),
                    float(curves.mean_deform[i]),
                    float(curves.sigma_eff_orig[i]),
                    float(curves.sigma_eff_deform[i]),
                    float(curves.stn_orig[i]),
                    float(curves.stn_deform[i]),
                    float(np.real(curves.exact_mean[i])),
                    float(curves.exact_stn_orig[i]),
                    float(curves.exact_stn_deform[i]),
                    float(curves.sigma_exact),
                ]
            )


def plot_sigma_eff(curves: Curves, out_path: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(curves.sigma_a, curves.sigma_eff_orig, ".", ms=3, label="Original MC")
    plt.plot(curves.sigma_a, curves.sigma_eff_deform, ".", ms=3, label="Deformed MC")
    plt.axhline(curves.sigma_exact, linestyle="--", label="Exact sigma")
    plt.xscale("log")
    plt.xlabel("sigma * A")
    plt.ylabel("sigma_eff(A)")
    plt.title("Effective string tension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_stn(curves: Curves, out_path: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(curves.sigma_a, curves.stn_orig, ".", ms=3, label="Original MC")
    plt.plot(curves.sigma_a, curves.stn_deform, ".", ms=3, label="Deformed MC")
    plt.plot(curves.sigma_a, curves.exact_stn_orig, linestyle="--", label="Exact original")
    plt.plot(curves.sigma_a, curves.exact_stn_deform, linestyle="--", label="Exact deformed")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("sigma * A")
    plt.ylabel("StN")
    plt.title("Signal-to-noise comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=5.555, help="Gauge coupling beta")
    parser.add_argument("--delta", type=float, default=0.2, help="Deformation parameter delta")
    parser.add_argument("--samples", type=int, default=10000, help="Number of MC samples")
    parser.add_argument("--max-area", type=int, default=1000, help="Largest loop area A to test")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")
    args, _ = parser.parse_known_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    curves = run_mc(
        beta=args.beta,
        delta=args.delta,
        n_samples=args.samples,
        max_area=args.max_area,
        seed=args.seed,
    )

    csv_path = outdir / "u1_deformed_observable_results.csv"
    sigma_plot = outdir / "u1_sigma_eff.png"
    stn_plot = outdir / "u1_stn.png"

    save_csv(curves, csv_path)
    plot_sigma_eff(curves, sigma_plot)
    plot_stn(curves, stn_plot)

    target_area = int(round(100.0 / curves.sigma_exact))
    if 1 <= target_area <= args.max_area:
        idx = target_area - 1
        exact_gain = curves.exact_stn_deform[idx] / max(curves.exact_stn_orig[idx], 1e-300)
    else:
        exact_gain = float("nan")

    print(f"beta              = {args.beta}")
    print(f"delta             = {args.delta}")
    print(f"samples           = {args.samples}")
    print(f"max_area          = {args.max_area}")
    print(f"exact sigma       = {curves.sigma_exact:.8f}")
    if np.isfinite(exact_gain):
        print(f"exact StN gain near A = 100/sigma: {exact_gain:.3e}")
    print(f"wrote             = {csv_path}")
    print(f"wrote             = {sigma_plot}")
    print(f"wrote             = {stn_plot}")


if __name__ == "__main__":
    main()
