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

function FiniteTempBasis{S}(β::Real, ωmax::Real, ε::Real; kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel, ε), max_size=-1) where {S<:Statistics}
	FiniteTempBasis{S}(kernel, sve_result, Float64(β), Float64(ωmax), Float64(ε), max_size)
end

# Convenience constructor - matches SparseIR.jl signature
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
    finite_temp_bases(β::Real, ωmax::Real, ε=nothing;
                      kernel=LogisticKernel(β * ωmax), sve_result=SVEResult(kernel; ε))

Construct `FiniteTempBasis` objects for fermion and bosons using the same
`LogisticKernel` instance.
"""
function finite_temp_bases(β::Real, ωmax::Real, ε::Real;
        kernel=LogisticKernel(β * ωmax),
        sve_result=SVEResult(kernel, ε))
    basis_f = FiniteTempBasis{Fermionic}(β, ωmax, ε; sve_result, kernel)
    basis_b = FiniteTempBasis{Bosonic}(β, ωmax, ε; sve_result, kernel)
    return basis_f, basis_b
end
