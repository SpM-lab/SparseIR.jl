"""
    FiniteTempBasis <: AbstractBasis

Intermediate representation (IR) basis for given temperature.

For a continuation kernel `K` from real frequencies, `ω ∈ [-ωmax, ωmax]`, to
imaginary time, `τ ∈ [0, β]`, this type stores the truncated singular
value expansion or IR basis:

    K(τ, ω) ≈ sum(u[l](τ) * s[l] * v[l](ω) for l in 1:L)

This basis is inferred from a reduced form by appropriate scaling of
the variables.

# Fields

  - `u::PiecewiseLegendrePolyVector`:
    Set of IR basis functions on the imaginary time (`tau`) axis.
    These functions are stored as piecewise Legendre polynomials.

    To obtain the value of all basis functions at a point or a array of
    points `x`, you can call the function `u(x)`. To obtain a single
    basis function, a slice or a subset `l`, you can use `u[l]`.

  - `uhat::PiecewiseLegendreFTVector`:
    Set of IR basis functions on the Matsubara frequency (`wn`) axis.
    These objects are stored as a set of Bessel functions.

    To obtain the value of all basis functions at a Matsubara frequency
    or a array of points `wn`, you can call the function `uhat(wn)`.
    Note that we expect reduced frequencies, which are simply even/odd
    numbers for bosonic/fermionic objects. To obtain a single basis
    function, a slice or a subset `l`, you can use `uhat[l]`.
  - `s`: Vector of singular values of the continuation kernel
  - `v::PiecewiseLegendrePolyVector`:
    Set of IR basis functions on the real frequency (`w`) axis.
    These functions are stored as piecewise Legendre polynomials.

    To obtain the value of all basis functions at a point or a array of
    points `w`, you can call the function `v(w)`. To obtain a single
    basis function, a slice or a subset `l`, you can use `v[l]`.
"""
mutable struct FiniteTempBasis{S, K} <: AbstractBasis{S}
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
	function FiniteTempBasis{S}(kernel::K, sve_result::SVEResult{K}, β::Real, ωmax::Real, ε::Real, max_size::Int) where {S<:Statistics, K<:AbstractKernel}
	    # Validate kernel/statistics compatibility
	    if isa(kernel, RegularizedBoseKernel) && S === Fermionic
	        throw(ArgumentError("RegularizedBoseKernel is incompatible with Fermionic statistics"))
	    end

	    # Create basis
	    status = Ref{Int32}(-100)
	    basis = SparseIR.spir_basis_new(_statistics_to_c(S), β, ωmax, kernel.ptr, sve_result.ptr, max_size, status)
	    status[] == SparseIR.SPIR_COMPUTATION_SUCCESS || error("Failed to create FiniteTempBasis $S $K $β $ωmax $ε $max_size $status[]")

        basis_size = Ref{Int32}(0)
        spir_basis_get_size(basis, basis_size) == SPIR_COMPUTATION_SUCCESS || error("Failed to get basis size")
        s = Vector{Float64}(undef, Int(basis_size[]))
        spir_basis_get_svals(basis, s) == SPIR_COMPUTATION_SUCCESS || error("Failed to get singular values")
        u_status = Ref{Int32}(-100)
        u = spir_basis_get_u(basis, u_status)
        u_status[] == SPIR_COMPUTATION_SUCCESS || error("Failed to get basis functions u $u_status[]")
        v_status = Ref{Int32}(-100)
        v = spir_basis_get_v(basis, v_status)
        v_status[] == SPIR_COMPUTATION_SUCCESS || error("Failed to get basis functions v $v_status[]")
        uhat_status = Ref{Int32}(-100)
        uhat = spir_basis_get_uhat(basis, uhat_status)
        uhat_status[] == SPIR_COMPUTATION_SUCCESS || error("Failed to get basis functions uhat $uhat_status[]")
	    result = new{S, K}(
            basis, kernel, sve_result, Float64(β), Float64(ωmax), Float64(ε),
            s,
            PiecewiseLegendrePolyVector(u, 0.0, β),
            PiecewiseLegendrePolyVector(v, -ωmax, ωmax),
            PiecewiseLegendreFTVector(uhat)
        )
	    finalizer(b -> spir_basis_release(b.ptr), result)
	    return result
	end
end

"""
    FiniteTempBasis{S}(β, ωmax, ε; kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel, ε), max_size=-1)

Construct a finite temperature basis suitable for the given `S` (`Fermionic`
or `Bosonic`) and cutoffs `β` and `ωmax`.

# Arguments

- `β`: Inverse temperature (must be positive)
- `ωmax`: Frequency cutoff (must be non-negative)
- `ε`: This parameter controls the number of basis functions. Only singular values ≥ ε * s[1] are kept.
  Typical values are 1e-6 to 1e-12 depending on the desired accuracy for your calculations. If ε is smaller than the square root of double precision machine epsilon (≈ 1.49e-8), the library will automatically use higher precision for the singular value decomposition, resulting in longer computation time for basis generation.

The number of basis functions grows logarithmically as log(1/ε) log (β * ωmax).
"""
function FiniteTempBasis{S}(β::Real, ωmax::Real, ε::Real; kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel, ε), max_size=-1) where {S<:Statistics}
	FiniteTempBasis{S}(kernel, sve_result, Float64(β), Float64(ωmax), Float64(ε), max_size)
end

"""
    FiniteTempBasis(stat::Statistics, β, ωmax, ε; kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel, ε), max_size=-1)

Convenience constructor that matches SparseIR.jl signature.

Construct a finite temperature basis for the given statistics type and cutoffs.

# Arguments

- `stat`: Statistics type (`Fermionic()` or `Bosonic()`)
- `β`: Inverse temperature (must be positive)
- `ωmax`: Frequency cutoff (must be non-negative)
- `ε`: Accuracy target for the basis. This parameter controls the number of basis functions. Only singular values ≥ ε * s[1] are kept.
  Typical values are 1e-6 to 1e-12 depending on the desired accuracy for your calculations. If ε is smaller than the square root of double precision machine epsilon (≈ 1.49e-8), the library will automatically use higher precision for the singular value decomposition, resulting in longer computation time for basis generation.

The number of basis functions grows logarithmically as log(1/ε) log (β * ωmax).
"""
function FiniteTempBasis(stat::S, β::Real, ωmax::Real, ε::Real; kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel, ε), max_size=-1) where {S<:Statistics}
    FiniteTempBasis{typeof(stat)}(β, ωmax, ε; kernel, sve_result, max_size)
end

function default_tau_sampling_points(basis::FiniteTempBasis)
    n_points = Ref{Int32}(-1)
    ret = spir_basis_get_n_default_taus(basis.ptr, n_points)
    ret == SPIR_COMPUTATION_SUCCESS || error("Failed to get number of default tau points")
    points_array = Vector{Float64}(undef, n_points[])
    ret = spir_basis_get_default_taus(basis.ptr, points_array)
    return points_array
end

function default_matsubara_sampling_points(basis::FiniteTempBasis; positive_only=false)
    n_points = Ref{Int32}(0)
    ret = spir_basis_get_n_default_matsus(basis.ptr, positive_only, n_points)
    ret == SPIR_COMPUTATION_SUCCESS || error("Failed to get number of default matsubara points")
	n_points[] > 0 || error("No default matsubara points found")

    points_array = Vector{Int64}(undef, n_points[])
    ret = spir_basis_get_default_matsus(basis.ptr, positive_only, points_array)
    ret == SPIR_COMPUTATION_SUCCESS || error("Failed to get default matsubara points")
    return points_array
end

function default_omega_sampling_points(basis::FiniteTempBasis)
    n_points = Ref{Int32}(-1)
    ret = spir_basis_get_n_default_ws(basis.ptr, n_points)
    ret == SPIR_COMPUTATION_SUCCESS || error("Failed to get number of default omega points")
    points_array = Vector{Float64}(undef, n_points[])
    ret = spir_basis_get_default_ws(basis.ptr, points_array)
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

# For now, accuracy is approximated by epsilon
# In reality, it would be computed from the singular values
accuracy(basis::FiniteTempBasis) = basis.epsilon

function (f::BasisFunction)(freq::MatsubaraFreq)
    return f(freq.n)
end

"""
    rescale(basis::FiniteTempBasis, new_beta)

Return a basis for different temperature.

Creates a new basis with the same accuracy ``ε`` but different temperature.
The new kernel is constructed with the same cutoff parameter ``Λ = β * ωmax``,
which implies a different UV cutoff ``ωmax`` since ``Λ`` stays constant.

# Arguments
- `basis`: The original basis to rescale
- `new_beta`: New inverse temperature

# Returns
A new `FiniteTempBasis` with the same statistics type and accuracy but different temperature.
"""
function rescale(basis::FiniteTempBasis{S}, new_beta::Real) where {S}
    # Rescale basis to new temperature
    new_lambda = Λ(basis) * new_beta / β(basis)
    kernel = LogisticKernel(new_lambda)
    return FiniteTempBasis{S}(kernel, new_beta, ωmax(basis), accuracy(basis))
end

# Additional utility functions
significance(basis::FiniteTempBasis) = basis.s ./ first(basis.s)

function range_to_length(range::UnitRange)
    isone(first(range)) || error("Range must start at 1.")
    return last(range)
end


"""
    finite_temp_bases(β::Real, ωmax::Real, ε;
                      kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel, ε))

Construct `FiniteTempBasis` objects for fermion and bosons using the same
`LogisticKernel` instance.

# Arguments

- `β`: Inverse temperature (must be positive)
- `ωmax`: Frequency cutoff (must be non-negative)
- `ε`: This parameter controls the number of basis functions. Only singular values ≥ ε * s[1] are kept.
  Typical values are 1e-6 to 1e-12 depending on the desired accuracy for your calculations. If ε is smaller than the square root of double precision machine epsilon (≈ 1.49e-8), the library will automatically use higher precision for the singular value decomposition, resulting in longer computation time for basis generation.

The number of basis functions grows logarithmically as log(1/ε) log (β * ωmax).
"""
function finite_temp_bases(β::Real, ωmax::Real, ε::Real;
        kernel=LogisticKernel(β * ωmax),
        sve_result=SVEResult(kernel, ε))
    basis_f = FiniteTempBasis{Fermionic}(β, ωmax, ε; sve_result, kernel)
    basis_b = FiniteTempBasis{Bosonic}(β, ωmax, ε; sve_result, kernel)
    return basis_f, basis_b
end
