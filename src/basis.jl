"""
    FiniteTempBasis <: AbstractBasis

Intermediate representation (IR) basis for given temperature.

For a continuation kernel `K` from real frequencies, `ω ∈ [-ωmax, ωmax]`, to
imaginary time, `τ ∈ [0, β]`, this type stores the truncated singular
value expansion or IR basis:

    K(τ, ω) ≈ sum(U_l(τ) * S_l * V_l(ω) for l in 0:L-1),

where `L = length(basis)`, `S_0 ≥ S_1 ≥ … > 0`, and Julia's `basis.u[l+1]`,
`basis.s[l+1]` and `basis.v[l+1]` are `U_l`, `S_l` and `V_l`. The `U_l` are
orthonormal on `[0, β]`, the `V_l` on `[-ωmax, ωmax]`, and their sign is fixed
by `U_l(β⁻) > 0`. The basis keeps the functions with `S_l/S_0 ≥ ε`.

This basis is inferred from a reduced form by appropriate scaling of
the variables: with `x = 2τ/β - 1` and `y = ω/ωmax`, the dimensionless SVE
`K(x, y) = Σ_l s_l u_l(x) v_l(y)` (see [`SVEResult`](@ref)) of the default
[`LogisticKernel`](@ref) gives

    U_l(τ) = √(2/β) u_l(x),    V_l(ω) = √(1/ωmax) v_l(y),    S_l = √(β ωmax/2) s_l.

With the `LogisticKernel`, fermions and bosons share `U_l`, `S_l` and `V_l`;
only `uhat` differs. A Green's function is expanded as

    G(τ) ≈ Σ_l G_l U_l(τ),    G(iν) ≈ Σ_l G_l Û_l(iν),
    G_l = -S_l ρ_l,           ρ_l = ∫ dω ρ(ω) V_l(ω),

where `ρ(ω) = A(ω)` for fermions and `ρ(ω) = A(ω)/tanh(βω/2)` for bosons, and
`A(ω)` is the spectral function (see [`LogisticKernel`](@ref)).

# Fields

  - `u::PiecewiseLegendrePolyVector`:
    Set of IR basis functions `U_l(τ)` on the imaginary time (`tau`) axis.
    These functions are stored as piecewise Legendre polynomials.

    To obtain the value of all basis functions at a point or an array of
    points `τ`, you can call the function `u(τ)`. `u[l+1]` is the single
    basis function `U_l`, and `u[range]` a subset.

    `u` accepts `τ ∈ [-β, β]` and throws `DomainError` outside. For `τ < 0` it
    returns `u(τ) = (-1)^ζ u(τ + β)`, where the parity `ζ` is 1 for fermions and
    0 for bosons. The endpoints are read as one-sided limits: `+0.0` is `0⁺`,
    `β` is `β⁻`, `-0.0` is `0⁻` (the value `(-1)^ζ U_l(β⁻)`) and `-β` is `(-β)⁺`
    (the value `(-1)^ζ U_l(0⁺)`).

  - `uhat::PiecewiseLegendreFTVector`:
    Set of IR basis functions `Û_l(iν)` on the Matsubara frequency axis, the
    Fourier transforms

        Û_l(iν) = ∫₀^β dτ exp(iντ) U_l(τ),    ν = nπ/β.

    To obtain the value of all basis functions at a Matsubara frequency
    or an array of frequencies, you can call the function `uhat(n)`.
    Note that we expect reduced frequencies `n`, which are simply even/odd
    numbers for bosonic/fermionic objects, or [`MatsubaraFreq`](@ref)s.
    `uhat[l+1]` is the single function `Û_l`, and `uhat[range]` a subset.
    For fermions, `Û_l(iν)` is purely imaginary for even `l` and real for odd
    `l`; for bosons it is the other way round.
  - `s`: Vector of singular values `S_l` of the continuation kernel
  - `v::PiecewiseLegendrePolyVector`:
    Set of IR basis functions `V_l(ω)` on the real frequency axis,
    `ω ∈ [-ωmax, ωmax]`. These functions are stored as piecewise Legendre
    polynomials.

    To obtain the value of all basis functions at a point or an array of
    points `ω`, you can call the function `v(ω)`. `v[l+1]` is the single
    basis function `V_l`, and `v[range]` a subset.
"""
mutable struct FiniteTempBasis{S,K} <: AbstractBasis{S}
    ptr::Ptr{spir_basis}
    kernel::K
    sve_result::SVEResult{K}
    beta::Float64
    wmax::Float64
    epsilon::Float64
    s::Vector{Float64}
    u::PiecewiseLegendrePolyVector
    v::PiecewiseLegendrePolyVector
    uhat::PiecewiseLegendreFTVector
    function FiniteTempBasis{S}(kernel::K, sve_result::SVEResult{K}, β::Real, ωmax::Real,
            ε::Real, max_size::Int) where {S<:Statistics,K<:AbstractKernel}
        _check_basis_parameters(β, ωmax, ε, max_size)
        # Validate kernel/statistics compatibility
        if isa(kernel, RegularizedBoseKernel) && S === Fermionic
            throw(ArgumentError("RegularizedBoseKernel is incompatible with Fermionic statistics"))
        end
        # The C library builds a basis from a kernel or SVE with another cutoff
        # without complaint, and the result is a wrong basis.
        isapprox(Λ(kernel), β * ωmax; rtol=1e-12) ||
            throw(ArgumentError("kernel cutoff Λ = $(Λ(kernel)) does not match β ωmax = $(β * ωmax)"))
        isapprox(Λ(sve_result.kernel), Λ(kernel); rtol=1e-12) ||
            throw(ArgumentError("sve_result was computed for Λ = $(Λ(sve_result.kernel)), \
                                 but the kernel has Λ = $(Λ(kernel))"))

        # Create basis
        status = Ref{Int32}(-100)
        basis = SparseIR.spir_basis_new(
            _statistics_to_c(S), β, ωmax, ε,
            kernel.ptr, sve_result.ptr, max_size, status)
        _check_status(status[], "spir_basis_new")
        _check_handle(basis, "spir_basis_new")

        u = v = uhat = Ptr{spir_funcs}(C_NULL)
        s = Float64[]
        try
            basis_size = Ref{Int32}(0)
            _check_status(spir_basis_get_size(basis, basis_size), "spir_basis_get_size")
            s = Vector{Float64}(undef, Int(basis_size[]))
            _check_status(spir_basis_get_svals(basis, s), "spir_basis_get_svals")
            u_status = Ref{Int32}(-100)
            u = spir_basis_get_u(basis, u_status)
            _check_status(u_status[], "spir_basis_get_u")
            v_status = Ref{Int32}(-100)
            v = spir_basis_get_v(basis, v_status)
            _check_status(v_status[], "spir_basis_get_v")
            uhat_status = Ref{Int32}(-100)
            uhat = spir_basis_get_uhat(basis, uhat_status)
            _check_status(uhat_status[], "spir_basis_get_uhat")
        catch
            # Nothing owns the handles yet: release them before rethrowing.
            for f in (u, v, uhat)
                f == C_NULL || spir_funcs_release(f)
            end
            spir_basis_release(basis)
            rethrow()
        end
        result = new{S,K}(
            basis, kernel, sve_result, Float64(β), Float64(ωmax), Float64(ε),
            s,
            PiecewiseLegendrePolyVector(u, -β, β, β, (0.0, β)),  # u uses [0, β] as default overlap range
            PiecewiseLegendrePolyVector(v, -ωmax, ωmax, 0.0),     # v uses default range (xmin, xmax)
            PiecewiseLegendreFTVector(uhat, zeta(S()))
        )
        finalizer(b -> spir_basis_release(b.ptr), result)
        return result
    end
end

"""
    FiniteTempBasis{S}(β, ωmax, ε; kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel, ε), max_size=-1)

Construct a finite temperature basis suitable for the given `S` (`Fermionic`
or `Bosonic`), inverse temperature `β` and frequency cutoff `ωmax`.

# Arguments

  - `β`: Inverse temperature (must be positive)
  - `ωmax`: Frequency cutoff (must be positive). The spectral function must
    vanish outside `[-ωmax, ωmax]`.
  - `ε`: This parameter controls the number of basis functions. Only the singular values with `S_l/S_0 ≥ ε` are kept.
    Typical values are 1e-6 to 1e-12 depending on the desired accuracy for your calculations. If ε is smaller than 1e-8, the library will automatically use higher (double-double) precision for the singular value expansion, resulting in longer computation time for basis generation.
  - `kernel`: The kernel; its cutoff must be `Λ = β * ωmax` (otherwise `ArgumentError`).
  - `sve_result`: The SVE of `kernel`, or of a kernel of the same type and `Λ`.
  - `max_size`: Maximum number of basis functions, `-1` for no limit.

The number of basis functions grows logarithmically as log(1/ε) log (β * ωmax).
"""
function FiniteTempBasis{S}(β::Real, ωmax::Real, ε::Real; kernel=nothing,
        sve_result=nothing, max_size=-1) where {S<:Statistics}
    # Validate before the defaults are built: LogisticKernel(β * ωmax) and
    # SVEResult(kernel, ε) would otherwise report a bad β, ωmax or ε in terms
    # of the kernel cutoff or as a C-level failure.
    _check_basis_parameters(β, ωmax, ε, max_size)
    kernel === nothing && (kernel = LogisticKernel(β * ωmax))
    sve_result === nothing && (sve_result = SVEResult(kernel, ε))
    _check_sve_kernel(sve_result, kernel)
    FiniteTempBasis{S}(kernel, sve_result, Float64(β), Float64(ωmax), Float64(ε), max_size)
end

function _check_sve_kernel(sve_result::SVEResult, kernel::AbstractKernel)
    sve_result.kernel isa typeof(kernel) || throw(ArgumentError(
        "sve_result was computed for a $(nameof(typeof(sve_result.kernel))), \
         but the basis uses a $(nameof(typeof(kernel)))"))
    return nothing
end

"""
    basis[1:n]

Truncate the basis to its `n` most significant singular values and functions.
The truncated basis shares the kernel and the SVE of `basis`; only ranges
`1:n` with `1 ≤ n ≤ length(basis)` are supported.
"""
function Base.getindex(basis::FiniteTempBasis{S}, range::AbstractRange) where {S}
    step(range) == 1 ||
        throw(ArgumentError("basis truncation needs a unit range, got $range"))
    stop = range_to_length(range)
    1 ≤ stop ≤ length(basis) || throw(BoundsError(basis, range))
    return FiniteTempBasis{S}(basis.beta, basis.wmax, basis.epsilon;
        kernel=basis.kernel, sve_result=basis.sve_result, max_size=stop)
end

function _check_basis_parameters(β::Real, ωmax::Real, ε::Real, max_size::Integer)
    isfinite(β) && β > 0 ||
        throw(DomainError(β, "inverse temperature β must be positive and finite"))
    isfinite(ωmax) && ωmax > 0 ||
        throw(DomainError(ωmax, "frequency cutoff ωmax must be positive and finite"))
    isfinite(ε) && ε > 0 || throw(DomainError(ε, "accuracy ε must be positive and finite"))
    max_size == -1 || max_size ≥ 1 ||
        throw(DomainError(max_size, "max_size must be -1 (no limit) or positive"))
    return nothing
end

"""
    FiniteTempBasis(stat::Statistics, β, ωmax, ε; kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel, ε), max_size=-1)

Convenience constructor, the same as
`FiniteTempBasis{typeof(stat)}(β, ωmax, ε; kernel, sve_result, max_size)`.

Construct a finite temperature basis for the given statistics, inverse
temperature and frequency cutoff.

# Arguments

  - `stat`: Statistics instance (`Fermionic()` or `Bosonic()`)
  - `β`: Inverse temperature (must be positive)
  - `ωmax`: Frequency cutoff (must be positive)
  - `ε`: Accuracy target for the basis. This parameter controls the number of basis functions. Only the singular values with `S_l/S_0 ≥ ε` are kept.
    Typical values are 1e-6 to 1e-12 depending on the desired accuracy for your calculations. If ε is smaller than 1e-8, the library will automatically use higher (double-double) precision for the singular value expansion, resulting in longer computation time for basis generation.

The number of basis functions grows logarithmically as log(1/ε) log (β * ωmax).
"""
function FiniteTempBasis(
        stat::S, β::Real, ωmax::Real, ε::Real; kernel=nothing,
        sve_result=nothing, max_size=-1) where {S<:Statistics}
    FiniteTempBasis{typeof(stat)}(β, ωmax, ε; kernel, sve_result, max_size)
end

function default_tau_sampling_points(basis::FiniteTempBasis; use_positive_taus::Bool=true)
    n_points = Ref{Int32}(-1)
    ret = spir_basis_get_n_default_taus(basis.ptr, n_points)
    _check_status(ret, "spir_basis_get_n_default_taus")
    points_array = Vector{Float64}(undef, n_points[])
    ret = spir_basis_get_default_taus(basis.ptr, points_array)
    _check_status(ret, "spir_basis_get_default_taus")

    if use_positive_taus
        points_array = mod.(points_array, β(basis))
        sort!(points_array)
    end

    return points_array
end

function default_matsubara_sampling_points(basis::FiniteTempBasis; positive_only=false)
    n_points = Ref{Int32}(0)
    ret = spir_basis_get_n_default_matsus(basis.ptr, positive_only, n_points)
    _check_status(ret, "spir_basis_get_n_default_matsus")
    n_points[] > 0 ||
        throw(SparseIRError("spir_basis_get_n_default_matsus returned no default points"))

    points_array = Vector{Int64}(undef, n_points[])
    ret = spir_basis_get_default_matsus(basis.ptr, positive_only, points_array)
    _check_status(ret, "spir_basis_get_default_matsus")
    return points_array
end

function default_omega_sampling_points(basis::FiniteTempBasis)
    n_points = Ref{Int32}(-1)
    ret = spir_basis_get_n_default_ws(basis.ptr, n_points)
    _check_status(ret, "spir_basis_get_n_default_ws")
    points_array = Vector{Float64}(undef, n_points[])
    ret = spir_basis_get_default_ws(basis.ptr, points_array)
    _check_status(ret, "spir_basis_get_default_ws")
    return points_array
end

# Basis function type
mutable struct BasisFunction
    ptr::Ptr{spir_funcs}
    basis::FiniteTempBasis  # Keep reference to prevent GC
end

# Property accessors
β(basis::FiniteTempBasis) = basis.beta
ωmax(basis::FiniteTempBasis) = basis.wmax
Λ(basis::FiniteTempBasis) = basis.beta * basis.wmax

function accuracy(basis::FiniteTempBasis)
    s_full = basis.sve_result.s
    n = length(basis.s)
    return length(s_full) > n ? s_full[n + 1] / first(s_full) : last(s_full) / first(s_full)
end

function (f::BasisFunction)(freq::MatsubaraFreq)
    return f(freq.n)
end

"""
    rescale(basis::FiniteTempBasis, new_beta)

Return a basis for different temperature.

Creates a new basis with the same accuracy ``ε`` but different temperature.
The new kernel is constructed with the same cutoff parameter ``Λ = β * ωmax``,
which implies a different frequency cutoff `ωmax = Λ / new_beta` since ``Λ``
stays constant.

# Arguments

  - `basis`: The original basis to rescale
  - `new_beta`: New inverse temperature

# Returns

A new `FiniteTempBasis` with the same statistics type and accuracy but different temperature.
"""
function rescale(basis::FiniteTempBasis{S}, new_beta::Real) where {S}
    new_beta > 0 || throw(DomainError(new_beta, "new_beta must be positive"))
    # Λ = β * ωmax is held fixed, so the kernel — and hence the SVE result —
    # can be reused unchanged and only ωmax moves.
    new_wmax = Λ(basis) / new_beta
    return FiniteTempBasis{S}(basis.kernel, basis.sve_result, Float64(new_beta),
        new_wmax, basis.epsilon, length(basis))
end

# Additional utility functions
significance(basis::FiniteTempBasis) = basis.s ./ first(basis.s)

s(basis::FiniteTempBasis) = basis.s
u(basis::FiniteTempBasis) = basis.u
v(basis::FiniteTempBasis) = basis.v
uhat(basis::FiniteTempBasis) = basis.uhat

function range_to_length(range::AbstractRange)
    isone(first(range)) ||
        throw(ArgumentError("basis truncation must start at 1, got the range $range"))
    return last(range)
end

"""
    finite_temp_bases(β::Real, ωmax::Real, ε;
                      kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel, ε))

Construct `FiniteTempBasis` objects for fermion and bosons using the same
`LogisticKernel` instance and SVE. The two bases share `U_l`, `S_l` and
`V_l`; the bosonic IR coefficients are those of `ρ(ω) = A(ω)/tanh(βω/2)`
(see [`FiniteTempBasis`](@ref)).

# Arguments

  - `β`: Inverse temperature (must be positive)
  - `ωmax`: Frequency cutoff (must be positive)
  - `ε`: This parameter controls the number of basis functions. Only the singular values with `S_l/S_0 ≥ ε` are kept.
    Typical values are 1e-6 to 1e-12 depending on the desired accuracy for your calculations. If ε is smaller than 1e-8, the library will automatically use higher (double-double) precision for the singular value expansion, resulting in longer computation time for basis generation.

The number of basis functions grows logarithmically as log(1/ε) log (β * ωmax).
"""
function finite_temp_bases(β::Real, ωmax::Real, ε::Real;
        kernel=nothing, sve_result=nothing)
    _check_basis_parameters(β, ωmax, ε, -1)
    kernel === nothing && (kernel = LogisticKernel(β * ωmax))
    sve_result === nothing && (sve_result = SVEResult(kernel, ε))
    _check_sve_kernel(sve_result, kernel)
    basis_f = FiniteTempBasis{Fermionic}(β, ωmax, ε; sve_result, kernel)
    basis_b = FiniteTempBasis{Bosonic}(β, ωmax, ε; sve_result, kernel)
    return basis_f, basis_b
end
