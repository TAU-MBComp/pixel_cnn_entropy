"""This module contains the exact solution for the entropy of the 2D Ising model, based on A. E. Ferdinand and M. E. Fisher, Bounded and Inhomogeneous Ising Models. I. Specific-Heat Anomaly of a Finite Lattice, Phys. Rev. 185, 832 (1969)"""
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpmath import mp
import gmpy2 as g2


def c(l, K, n):
    c = (g2.cosh(g2.mpfr(2.0) * K) * g2.coth(g2.mpfr(2.0) * K) -
         g2.cos(g2.mpfr(l) * np.pi / g2.mpfr(n)))
    return c


def gamma(l, K, n):
    if l == 0:
        gamma_0 = (2.0 * K + g2.log(g2.tanh(K)))
        return gamma_0
    else:
        gamma = g2.log(c(l, K, n) + (c(l, K, n)**2 - 1.0)**0.5)
        return gamma


def Z_i(i, K, n):
    if i == 0:
        Z_1 = []
        for r in range(n):
            z = (2.0 * g2.cosh(0.5 * n * gamma((2.0 * r + 1.0), K, n)))
            Z_1.append(2.0 * g2.cosh(0.5 * n * gamma((2.0 * r + 1.0), K, n)))
        return np.product(Z_1)

    elif i == 1:
        Z_2 = []
        for r in range(n):
            Z_2.append(2.0 * g2.sinh(0.5 * n * gamma((2.0 * r + 1.0), K, n)))
        return np.product(Z_2)

    elif i == 2:
        Z_3 = []
        for r in range(n):
            Z_3.append(2 * g2.cosh(0.5 * n * gamma((2.0 * r), K, n)))
        return np.product(Z_3)

    elif i == 3:
        Z_4 = []
        for r in range(n):
            Z_4.append(2.0 * g2.sinh(0.5 * n * gamma((2.0 * r), K, n)))
        return np.product(Z_4)
    else:
        print("error")


def Z(K, n):
    Z_partial = 0
    for i in range(4):
        Z_partial += Z_i(i, K, n)

    partial_correction = 0.5 * (2.0 * g2.sinh(2.0 * K))**(0.5 * n**2)
    return partial_correction * Z_partial


def d_log_Z(K, d_K, n):
    return (g2.log(Z(K + d_K, n)) - g2.log(Z(K - d_K, n))) / (2 * d_K)


def d2_log_Z(K, d_K, n):
    return (g2.log(Z(K + d_K, n)) - 2.0 * g2.log(Z(K, n)) +
            g2.log(Z(K - d_K, n))) / (d_K**2)


def U(T, d_T, n):
    return -d_log_Z(1.0 / T, d_T, n)


def F(T, n):
    return (-g2.log(Z(1.0 / T, n)) * T)


def S(T, n, d_T):
    return (U(T, d_T, n) - F(T, n)) / (T * n**2)


def C(K, d_K, n):
    return (K**2) * d2_log_Z(K, d_K, n)


def entropy(T, n, d_T=1e-12):
    d_T = g2.mpfr(d_T)
    return S(T, n, d_T)
