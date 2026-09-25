# The C library rejects Λ = 0 as well, so a non-positive or non-finite cutoff
# is reported here, with the value, before the call.
function _check_cutoff(Λ::Real)
    isfinite(Λ) && Λ > 0 ||
        throw(DomainError(Λ, "kernel cutoff Λ must be positive and finite"))
    return nothing
end

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
        _check_cutoff(Λ)
        status = Ref{Cint}(-100)
        ptr = spir_logistic_kernel_new(Float64(Λ), status)
        _check_status(status[], "spir_logistic_kernel_new")
        _check_handle(ptr, "spir_logistic_kernel_new")
        kernel = new(ptr, Float64(Λ))
        finalizer(k -> spir_kernel_release(k.ptr), kernel)
        return kernel
    end
end

@doc raw"""
    RegularizedBoseKernel <: AbstractKernel

Regularized bosonic analytical continuation kernel.

!!! warning "Deprecated"

    Use [`LogisticKernel`](@ref), the default kernel for both statistics.
    `RegularizedBoseKernel` will be removed in a future release. For
    `ωmax ≠ 1`, libsparseir releases without the fix of
    [SpM-lab/sparse-ir-rs#273](https://github.com/SpM-lab/sparse-ir-rs/issues/273)
    scale the singular values of its bases by ``ω_\mathrm{max}^{-1}`` instead
    of ``ω_\mathrm{max}^{+1}``.

In dimensionless variables ``x = 2 τ/β - 1``, ``y = β ω/Λ``, the bosonic
integral kernel is a function on ``[-1, 1] × [-1, 1]``:
```math
    K(x, y) = y \frac{e^{-Λ y (x + 1) / 2}}{1 - e^{-Λ y}}
```
In physical units it is ``K(τ, ω) = ω_\mathrm{max} K(x, y) = ω e^{-τω} / (1 - e^{-βω})``,
which acts on ``ρ(ω)/ω`` (N. Chikano et al., Computer Physics Communications
240, 181 (2019), Eqs. (1)-(3)). Care has to be taken in evaluating this expression
around ``y = 0``.
"""
mutable struct RegularizedBoseKernel <: AbstractKernel
    ptr::Ptr{spir_kernel}
    Λ::Float64

    function RegularizedBoseKernel(Λ::Real)
        Base.depwarn("RegularizedBoseKernel is deprecated and will be removed in a \
                      future release; use LogisticKernel, the default kernel for both \
                      statistics (https://github.com/SpM-lab/sparse-ir-rs/issues/273)",
            :RegularizedBoseKernel)
        _check_cutoff(Λ)
        status = Ref{Cint}(-100)
        ptr = spir_reg_bose_kernel_new(Float64(Λ), status)
        _check_status(status[], "spir_reg_bose_kernel_new")
        _check_handle(ptr, "spir_reg_bose_kernel_new")
        kernel = new(ptr, Float64(Λ))
        finalizer(k -> spir_kernel_release(k.ptr), kernel)
        return kernel
    end
end

Λ(kernel::LogisticKernel) = kernel.Λ
Λ(kernel::RegularizedBoseKernel) = kernel.Λ

iscentrosymmetric(::LogisticKernel) = true
iscentrosymmetric(::RegularizedBoseKernel) = true
