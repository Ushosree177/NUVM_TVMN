"""
TVMN Step-by-Step Mathematical Derivation

Triangular Variance Mixed Normal model:

    V ~ Triangular(a, m, b), 0 < a < m < b
    X | V = v ~ Normal(0, v)

This file is intentionally written like a mathematical notebook in Python.
It prints the derivation steps and also defines reusable numerical functions:

    tvmn_pdf
    tvmn_cdf
    tvmn_survival
    tvmn_hazard
    tvmn_reverse_hazard
    tvmn_cumulative_hazard
    tvmn_moments

No simulation, estimation, or real-data analysis is included here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy import integrate, stats
from scipy.special import erfc


@dataclass(frozen=True)
class TVMNParameters:
    """Parameters of V ~ Triangular(a, m, b)."""

    a: float
    m: float
    b: float

    def validate(self) -> None:
        if not (self.a > 0 and self.a < self.m < self.b):
            raise ValueError("TVMN requires 0 < a < m < b.")


def triangular_variance_pdf(v, params: TVMNParameters):
    """
    Density of V ~ Triangular(a, m, b).

    g(v) =
        2(v-a) / ((b-a)(m-a)),  a <= v <= m
        2(b-v) / ((b-a)(b-m)),  m <= v <= b
        0,                       otherwise
    """
    params.validate()
    a, m, b = params.a, params.m, params.b
    v = np.asarray(v, dtype=float)
    scalar_input = v.ndim == 0
    v = np.atleast_1d(v)

    density = np.zeros_like(v, dtype=float)
    left = (a <= v) & (v <= m)
    right = (m < v) & (v <= b)
    density[left] = 2.0 * (v[left] - a) / ((b - a) * (m - a))
    density[right] = 2.0 * (b - v[right]) / ((b - a) * (b - m))

    return float(density[0]) if scalar_input else density


def _A_antiderivative(v, lam):
    """
    A(v) = integral v^(-1/2) exp(-lambda / v) dv.

    A(v) =
        2 sqrt(v) exp(-lambda/v)
        - 2 sqrt(pi lambda) erfc(sqrt(lambda/v))
    """
    v = np.asarray(v, dtype=float)
    lam = np.asarray(lam, dtype=float)
    return (
        2.0 * np.sqrt(v) * np.exp(-lam / v)
        - 2.0 * np.sqrt(np.pi * lam) * erfc(np.sqrt(lam / v))
    )


def _B_antiderivative(v, lam):
    """
    B(v) = integral v^(1/2) exp(-lambda / v) dv.

    B(v) =
        (2/3)v^(3/2) exp(-lambda/v)
        - (4/3)lambda sqrt(v) exp(-lambda/v)
        + (4/3)sqrt(pi)lambda^(3/2) erfc(sqrt(lambda/v))
    """
    v = np.asarray(v, dtype=float)
    lam = np.asarray(lam, dtype=float)
    return (
        (2.0 / 3.0) * v ** 1.5 * np.exp(-lam / v)
        - (4.0 / 3.0) * lam * np.sqrt(v) * np.exp(-lam / v)
        + (4.0 / 3.0) * np.sqrt(np.pi) * lam ** 1.5 * erfc(np.sqrt(lam / v))
    )


def tvmn_pdf(x, a, m, b):
    """
    Closed-form TVMN marginal PDF.

    f(x) = integral_a^b phi(x; 0, v) g(v) dv

    where phi(x; 0, v) is the N(0, v) density.
    """
    params = TVMNParameters(a, m, b)
    params.validate()

    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    lam = 0.5 * x * x

    A_a = _A_antiderivative(a, lam)
    A_m = _A_antiderivative(m, lam)
    A_b = _A_antiderivative(b, lam)

    B_a = _B_antiderivative(a, lam)
    B_m = _B_antiderivative(m, lam)
    B_b = _B_antiderivative(b, lam)

    left_part = (B_m - B_a - a * (A_m - A_a)) / (m - a)
    right_part = (b * (A_b - A_m) - (B_b - B_m)) / (b - m)
    density = 2.0 * (left_part + right_part) / ((b - a) * np.sqrt(2.0 * np.pi))
    density = np.maximum(density, 0.0)

    return float(density[0]) if scalar_input else density


def tvmn_cdf(x, a, m, b, quadrature_points=100):
    """
    TVMN CDF by Gauss-Legendre quadrature.

    F(x) = integral_a^b Phi(x / sqrt(v)) g(v) dv

    The integral is split at m because the triangular density is piecewise.
    """
    params = TVMNParameters(a, m, b)
    params.validate()

    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)

    def integrate_on_interval(lower, upper, triangular_side):
        v = lower + 0.5 * (nodes + 1.0) * (upper - lower)
        w = 0.5 * (upper - lower) * weights

        if triangular_side == "left":
            gv = 2.0 * (v - a) / ((b - a) * (m - a))
        else:
            gv = 2.0 * (b - v) / ((b - a) * (b - m))

        conditional_cdf = stats.norm.cdf(x[None, :] / np.sqrt(v[:, None]))
        return np.sum(w[:, None] * gv[:, None] * conditional_cdf, axis=0)

    cdf = integrate_on_interval(a, m, "left") + integrate_on_interval(m, b, "right")
    cdf = np.clip(cdf, 0.0, 1.0)

    return float(cdf[0]) if scalar_input else cdf


def tvmn_survival(x, a, m, b):
    """Survival function S(x) = 1 - F(x)."""
    return 1.0 - tvmn_cdf(x, a, m, b)


def tvmn_hazard(x, a, m, b):
    """Hazard function h(x) = f(x) / S(x)."""
    return tvmn_pdf(x, a, m, b) / np.maximum(tvmn_survival(x, a, m, b), 1e-300)


def tvmn_reverse_hazard(x, a, m, b):
    """Reverse hazard function r(x) = f(x) / F(x)."""
    return tvmn_pdf(x, a, m, b) / np.maximum(tvmn_cdf(x, a, m, b), 1e-300)


def tvmn_cumulative_hazard(x, a, m, b):
    """Cumulative hazard function H(x) = -log(S(x))."""
    return -np.log(np.maximum(tvmn_survival(x, a, m, b), 1e-300))


def tvmn_moments(a, m, b):
    """
    Core moments using conditional expectation.

    E[X] = E[E[X | V]] = 0
    Var(X) = E[V] = (a + m + b) / 3
    E[X^4] = E[E[X^4 | V]] = 3E[V^2]
    """
    params = TVMNParameters(a, m, b)
    params.validate()

    ev = (a + m + b) / 3.0
    ev2 = (a * a + m * m + b * b + a * m + a * b + m * b) / 6.0
    fourth_moment = 3.0 * ev2
    variance = ev
    raw_kurtosis = fourth_moment / (variance * variance)
    excess_kurtosis = raw_kurtosis - 3.0

    return {
        "E[X]": 0.0,
        "E[X^2]": ev,
        "Var(X)": variance,
        "E[V]": ev,
        "E[V^2]": ev2,
        "E[X^3]": 0.0,
        "E[X^4]": fourth_moment,
        "Skewness": 0.0,
        "Kurtosis": raw_kurtosis,
        "Excess kurtosis": excess_kurtosis,
    }


def verify_pdf_numerically(a, m, b):
    """Numerical sanity check of integral f(x) dx = 1."""
    result, error = integrate.quad(lambda x: tvmn_pdf(x, a, m, b), -np.inf, np.inf, epsabs=1e-10)
    return result, error


def print_derivation(a, m, b):
    """Print the paper-style mathematical steps."""
    print("\nTVMN: Triangular Variance Mixed Normal Distribution")
    print("=" * 70)

    print("\nStep 1. Define the latent variance")
    print("V ~ Triangular(a, m, b), with 0 < a < m < b.")
    print(f"Here: a={a}, m={m}, b={b}")

    print("\nThe triangular variance density is")
    print("g(v) = 2(v-a) / ((b-a)(m-a)),  a <= v <= m")
    print("g(v) = 2(b-v) / ((b-a)(b-m)),  m <= v <= b")
    print("g(v) = 0, otherwise")

    print("\nStep 2. Define the conditional model")
    print("X | V=v ~ Normal(0, v)")
    print("f(x | v) = 1 / sqrt(2*pi*v) * exp(-x^2/(2v))")

    print("\nStep 3. Marginal PDF")
    print("f(x) = integral_a^b f(x | v) g(v) dv")
    print("Split at m because g(v) is piecewise:")
    print("f(x) = 2 / ((b-a)sqrt(2*pi)) * [")
    print("       1/(m-a) integral_a^m (v-a)v^(-1/2) exp(-x^2/(2v)) dv")
    print("     + 1/(b-m) integral_m^b (b-v)v^(-1/2) exp(-x^2/(2v)) dv")
    print("]")

    print("\nLet lambda = x^2 / 2 and define")
    print("A(v) = integral v^(-1/2) exp(-lambda/v) dv")
    print("B(v) = integral v^(1/2) exp(-lambda/v) dv")

    print("\nThen")
    print("A(v) = 2sqrt(v)exp(-lambda/v)")
    print("       - 2sqrt(pi*lambda)erfc(sqrt(lambda/v))")
    print("B(v) = (2/3)v^(3/2)exp(-lambda/v)")
    print("       - (4/3)lambda sqrt(v)exp(-lambda/v)")
    print("       + (4/3)sqrt(pi)lambda^(3/2)erfc(sqrt(lambda/v))")

    print("\nTherefore the TVMN PDF is")
    print("f(x) = 2 / ((b-a)sqrt(2*pi)) * {")
    print("       [B(m)-B(a)-a(A(m)-A(a))] / (m-a)")
    print("     + [b(A(b)-A(m))-(B(b)-B(m))] / (b-m)")
    print("}")

    print("\nTheorem 1. PDF validity")
    print("We verify that integral_{-infinity}^{infinity} f(x) dx = 1.")
    print("Using f(x) = integral_a^b f(x|v)g(v)dv,")
    print("integral f(x) dx")
    print("= integral_{-infinity}^{infinity} integral_a^b f(x|v)g(v) dv dx")
    print("= integral_a^b [integral_{-infinity}^{infinity} f(x|v) dx] g(v) dv")
    print("= integral_a^b 1 * g(v) dv")
    print("= 1.")
    print("Thus f(x) is a valid probability density.")

    print("\nSection 3. CDF derivation")
    print("F(x) = P(X <= x)")
    print("     = integral_a^b P(X <= x | V=v) g(v) dv")
    print("Since X | V=v ~ Normal(0, v),")
    print("P(X <= x | V=v) = Phi(x / sqrt(v)).")
    print("Therefore")
    print("F(x) = integral_a^b Phi(x / sqrt(v)) g(v) dv")
    print("Split at m:")
    print("F(x) = integral_a^m Phi(x/sqrt(v)) * 2(v-a)/((b-a)(m-a)) dv")
    print("     + integral_m^b Phi(x/sqrt(v)) * 2(b-v)/((b-a)(b-m)) dv")

    print("\nStandard reliability functions")
    print("Survival function:          S(x) = 1 - F(x)")
    print("Hazard function:            h(x) = f(x) / S(x)")
    print("Reverse hazard function:    r(x) = f(x) / F(x)")
    print("Cumulative hazard function: H(x) = -log(S(x))")

    print("\nSection 4. Moments by conditional expectation")
    print("E[X] = E[E[X | V]] = E[0] = 0")
    print("E[X^2] = E[E[X^2 | V]] = E[V]")
    print("For V ~ Triangular(a,m,b), E[V] = (a+m+b)/3.")
    print("Therefore Var(X) = (a+m+b)/3.")
    print("E[X^3] = 0 by symmetry.")
    print("E[X^4] = E[E[X^4 | V]] = E[3V^2] = 3E[V^2].")
    print("For triangular V,")
    print("E[V^2] = (a^2 + m^2 + b^2 + am + ab + mb) / 6.")
    print("Therefore")
    print("E[X^4] = (a^2 + m^2 + b^2 + am + ab + mb) / 2.")
    print("Kurtosis = E[X^4] / Var(X)^2.")

    print("\nNumerical check for the selected parameters")
    pdf_integral, pdf_error = verify_pdf_numerically(a, m, b)
    print(f"Numerical integral of f(x) over R = {pdf_integral:.12f}")
    print(f"Estimated integration error        = {pdf_error:.2e}")

    print("\nSample function values")
    xs = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print("x values:       ", xs)
    print("PDF f(x):       ", np.round(tvmn_pdf(xs, a, m, b), 8))
    print("CDF F(x):       ", np.round(tvmn_cdf(xs, a, m, b), 8))
    print("Survival S(x):  ", np.round(tvmn_survival(xs, a, m, b), 8))
    print("Hazard h(x):    ", np.round(tvmn_hazard(xs, a, m, b), 8))

    print("\nMoment values")
    for name, value in tvmn_moments(a, m, b).items():
        print(f"{name:18s} = {value:.12f}")


def main():
    parser = argparse.ArgumentParser(description="Print the TVMN mathematical derivation.")
    parser.add_argument("--a", type=float, default=0.5, help="Minimum variance")
    parser.add_argument("--m", type=float, default=1.0, help="Most likely variance")
    parser.add_argument("--b", type=float, default=2.0, help="Maximum variance")
    args = parser.parse_args()

    print_derivation(args.a, args.m, args.b)


if __name__ == "__main__":
    main()
