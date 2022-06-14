\newcommand{\var}{\mathrm{Var}}

# Example usage and detailed explanation

We will explain the inner workings of `SparseIR.jl` by means of an example use case, adapted from the `sparse-ir` paper.
<!-- TODO: link to paper once it's released -->

## Problem statement

> Let us perform self-consistent second-order perturbation theory for the single impurity Anderson model at finite temperature.
> Its Hamiltonian is given by
> $$
>     H = U c^\dagger_\uparrow c^\dagger_\downarrow c_\downarrow c_\uparrow + \sum_{p\sigma} \big(V_{p\sigma}  f_{p\sigma}^\dagger c_\sigma + V_{p\sigma}^* c_\sigma^\dagger c_\sigma^\dagger\big) + \sum_{p\sigma} \epsilon_{p} f_{p\sigma}^\dagger f_{p\sigma}
> $$
> where $U$ is the electron interaction strength, $c_\sigma$ annihilates an electron on the impurity, $f_{p\sigma}$ annihilates an electron in the bath, $\dagger$ denotes the Hermitian conjugate, $p\in\mathbb R$ is bath momentum, and $\sigma\in\{\uparrow, \downarrow\}$ is spin.
> The hybridization strength $V_{p\sigma}$ and bath energies $\epsilon_p$ are chosen such that the non-interacting density of states is semi-elliptic with a half-bandwidth of one, $\rho_0(\omega) = \frac2\pi\sqrt{1-\omega^2}$, $U=1.2$, $\beta=10$, and the system is assumed to be half-filled.

## Treatment

We first import `SparseIR` and construct an appropriate basis ($\omega_\mathrm{max} = 8$ should be more than enough for this example):
```julia-repl
julia> using SparseIR

julia> basis = FiniteTempBasis(fermion, 10, 8)
FiniteTempBasis{LogisticKernel, Float64}(fermion, 10.0, 8.0)
```
There's quite a lot happening behind the scenes in this first innocuous-looking statement, so let's break it down:
Because we did not specify otherwise, the constructor chose the analytic continuation kernel for fermions, `LogisticKernel(Λ=80.0)`, defined by
$$
    K(x, y) = \frac{e^{-Λ y (x + 1) / 2}}{1 + e^{-Λ y}},
$$
for us.

Central is the _singular value expansion_'s (SVE) computation, which is handled by the function `compute_sve`:

It first constructs 