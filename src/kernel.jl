@doc raw"""
    LogisticKernel <: AbstractKernel

Fermionic/bosonic analytical continuation kernel.

In dimensionless variables ``x = 2 τ/β - 1``, ``y = β ω/Λ``,
the integral kernel is a function on ``[-1, 1] × [-1, 1]``:
```math
    K(x, y) = \frac{e^{-Λ y (x + 1) / 2}}{1 + e^{-Λ y}}
```
LogisticKernel is a fermionic analytic continuation kernel.
Nevertheless, one can model the ``τ`` dependence of
a bosonic correlation function as follows:
```math
    ∫ \frac{e^{-Λ y (x + 1) / 2}}{1 - e^{-Λ y}} ρ(y) dy = ∫ K(x, y) ρ'(y) dy,
```
with
```math
    ρ'(y) = w(y) ρ(y),
```
where the weight function is given by
```math
    w(y) = \frac{1}{\tanh(Λ y/2)}.
```
"""
mutable struct LogisticKernel <: AbstractKernel
    ptr::Ptr{spir_kernel}
    Λ::Float64

    function LogisticKernel(Λ::Real)
        Λ ≥ 0 || throw(DomainError(Λ, "Kernel cutoff Λ must be non-negative"))
        status = Ref{Cint}(-100)
        ptr = spir_logistic_kernel_new(Float64(Λ), status)
        status[] == 0 || error("Failed to create logistic kernel")
        kernel = new(ptr, Float64(Λ))
        finalizer(k -> spir_kernel_release(k.ptr), kernel)
        return kernel
    end
end

@doc raw"""
    RegularizedBoseKernel <: AbstractKernel

Regularized bosonic analytical continuation kernel.

In dimensionless variables ``x = 2 τ/β - 1``, ``y = β ω/Λ``, the fermionic
integral kernel is a function on ``[-1, 1] × [-1, 1]``:
```math
    K(x, y) = y \frac{e^{-Λ y (x + 1) / 2}}{e^{-Λ y} - 1}
```
Care has to be taken in evaluating this expression around ``y = 0``.
"""
mutable struct RegularizedBoseKernel <: AbstractKernel
    ptr::Ptr{spir_kernel}
    Λ::Float64

    function RegularizedBoseKernel(Λ::Real)
        Λ ≥ 0 || throw(DomainError(Λ, "Kernel cutoff Λ must be non-negative"))
        status = Ref{Cint}(-100)
        ptr = spir_reg_bose_kernel_new(Float64(Λ), status)
        status[] != 0 && error("Failed to create regularized Bose kernel")
        kernel = new(ptr, Float64(Λ))
        finalizer(k -> spir_kernel_release(k.ptr), kernel)
        return kernel
    end
end

Λ(kernel::LogisticKernel) = kernel.Λ
Λ(kernel::RegularizedBoseKernel) = kernel.Λ
