
"""
TauSampling{T,B} <: AbstractSampling

Sparse sampling in imaginary time using the C API.

Allows transformation between IR basis coefficients and sampling points in imaginary time.
"""
mutable struct TauSampling{T<:Real,B<:AbstractBasis} <: AbstractSampling{T,Float64,Nothing}
    ptr::Ptr{spir_sampling}
    sampling_points::Vector{T}
    basis::B

    function TauSampling{T,B}(ptr::Ptr{spir_sampling}, sampling_points::Vector{T}, basis::B) where {T<:Real,B<:AbstractBasis}
        obj = new{T,B}(ptr, sampling_points, basis)
        finalizer(s->spir_sampling_release(s.ptr), obj)
        return obj
    end
end

const TauSampling64F = TauSampling{Float64, FiniteTempBasis{Fermionic, LogisticKernel}}
const TauSampling64B = TauSampling{Float64, FiniteTempBasis{Bosonic, LogisticKernel}}

"""
MatsubaraSampling{T,B} <: AbstractSampling

Sparse sampling in Matsubara frequencies using the C API.

Allows transformation between IR basis coefficients and sampling points in Matsubara frequencies.
"""
mutable struct MatsubaraSampling{T<:MatsubaraFreq,B<:AbstractBasis} <: AbstractSampling{T,ComplexF64,Nothing}
    ptr::Ptr{spir_sampling}
    sampling_points::Vector{T}
    positive_only::Bool
    basis::B

    function MatsubaraSampling{T,B}(ptr::Ptr{spir_sampling}, sampling_points::Vector{T}, positive_only::Bool, basis::B) where {T<:MatsubaraFreq,B<:AbstractBasis}
        obj = new{T,B}(ptr, sampling_points, positive_only, basis)
        finalizer(s->spir_sampling_release(s.ptr), obj)
        return obj
    end
end

const MatsubaraSampling64F = MatsubaraSampling{FermionicFreq, FiniteTempBasis{Fermionic, LogisticKernel}}
const MatsubaraSampling64B = MatsubaraSampling{BosonicFreq, FiniteTempBasis{Bosonic, LogisticKernel}}

# Convenience constructors

"""
    TauSampling(basis::AbstractBasis; sampling_points=nothing, use_positive_taus=false)

Construct a `TauSampling` object from a basis. If `sampling_points` is not provided,
the default tau sampling points from the basis are used.

If `use_positive_taus=false` (default), the sampling points are in the range [-β/2, β/2].

If `use_positive_taus=true`, the sampling points are folded to the positive tau domain [0, β).
This was the default behavior in SparseIR.jl of versions 1.x.x.
"""
function TauSampling(basis::AbstractBasis; sampling_points=nothing, use_positive_taus=false)
    @show use_positive_taus
    if sampling_points === nothing
        points = default_tau_sampling_points(basis)
        if use_positive_taus
            points = mod.(points, β(basis))
            sort!(points)
        end
        sampling_points = points
    end

    # Create sampling object with C_API
    status = Ref{Int32}(-100)
    if !_is_column_major_contiguous(sampling_points)
        error("Sampling points must be contiguous")
    end
    ptr = C_API.spir_tau_sampling_new(_get_ptr(basis), length(sampling_points), sampling_points, status)
    status[] == C_API.SPIR_COMPUTATION_SUCCESS || error("Failed to create tau sampling: status=$(status[])")
    ptr != C_API.C_NULL || error("Failed to create tau sampling: null pointer returned")

    return TauSampling{Float64,typeof(basis)}(ptr, sampling_points, basis)
end

"""
    MatsubaraSampling(basis::AbstractBasis; positive_only=false, sampling_points=nothing, factorize=true)

Construct a `MatsubaraSampling` object from a basis. If `sampling_points` is not provided,
the default Matsubara sampling points from the basis are used.

If `positive_only=true`, assumes functions are symmetric in Matsubara frequency.
"""
function MatsubaraSampling(basis::AbstractBasis; positive_only=false, sampling_points=nothing)
    if sampling_points === nothing
        # Get default Matsubara sampling points from basis
        status = Ref{Int32}(-100)
        n_points = Ref{Int32}(-1)

        ret = C_API.spir_basis_get_n_default_matsus(_get_ptr(basis), positive_only, n_points)
        ret == C_API.SPIR_COMPUTATION_SUCCESS || error("Failed to get number of default Matsubara points")

        points_array = Vector{Int64}(undef, n_points[])
        ret = C_API.spir_basis_get_default_matsus(_get_ptr(basis), positive_only, points_array)
        ret == C_API.SPIR_COMPUTATION_SUCCESS || error("Failed to get default Matsubara points")

        # Convert to MatsubaraFreq objects based on statistics
        if statistics(basis) isa Fermionic
            sampling_points = [FermionicFreq(n) for n in points_array]
        else
            sampling_points = [BosonicFreq(n) for n in points_array]
        end
    else
        # Convert input to appropriate MatsubaraFreq type
        if statistics(basis) isa Fermionic
            sampling_points = [p isa FermionicFreq ? p : FermionicFreq(Int(p)) for p in sampling_points]
        else
            sampling_points = [p isa BosonicFreq ? p : BosonicFreq(Int(p)) for p in sampling_points]
        end
    end

    # Extract indices for C API
    indices = [Int64(Int(p)) for p in sampling_points]

    status = Ref{Int32}(-100)
    ptr = C_API.spir_matsu_sampling_new(_get_ptr(basis), positive_only, length(indices), indices, status)
    status[] == C_API.SPIR_COMPUTATION_SUCCESS || error("Failed to create Matsubara sampling: status=$(status[])")
    ptr != C_NULL || error("Failed to create Matsubara sampling: null pointer returned")

    return MatsubaraSampling{eltype(sampling_points),typeof(basis)}(ptr, sampling_points, positive_only, basis)
end

# Common interface functions

"""
    eval_matrix(T, basis, x)

Return evaluation matrix from coefficients to sampling points. `T <: AbstractSampling`.
"""
function eval_matrix end
eval_matrix(::Type{TauSampling}, basis, x)       = permutedims(basis.u(x))
eval_matrix(::Type{MatsubaraSampling}, basis, x) = permutedims(basis.uhat(x))

"""
    npoints(sampling::AbstractSampling)

Get the number of sampling points.
"""
function npoints(sampling::Union{TauSampling,MatsubaraSampling})
    n_points = Ref{Int32}(-1)
    ret = C_API.spir_sampling_get_npoints(sampling.ptr, n_points)
    ret == C_API.SPIR_COMPUTATION_SUCCESS || error("Failed to get number of sampling points")
    return Int(n_points[])
end

# Evaluation and fitting functions

"""
    evaluate(sampling::AbstractSampling, al::Array; dim=1)

Evaluate basis coefficients at the sampling points using the C API.

For multidimensional arrays, `dim` specifies which dimension corresponds to the basis coefficients.
"""
function evaluate(sampling::Union{TauSampling,MatsubaraSampling}, al::Array{T,N}; dim=1) where {T,N}
    # Determine output dimensions
    output_dims = collect(size(al))
    output_dims[dim] = npoints(sampling)

    # Determine output type based on sampling type
    if sampling isa TauSampling
        # For complex input, TauSampling should produce complex output
        output_type = T
        output = Array{output_type,N}(undef, output_dims...)
        evaluate!(output, sampling, al; dim=dim)
    else # MatsubaraSampling
        output_type = T <: Real ? ComplexF64 : promote_type(ComplexF64, T)
        output = Array{output_type,N}(undef, output_dims...)
        evaluate!(output, sampling, al; dim=dim)
    end

    return output
end

"""
    evaluate!(output::Array, sampling::AbstractSampling, al::Array; dim=1)

In-place version of [`evaluate`](@ref). Write results to the pre-allocated `output` array.
"""
function evaluate!(output::Array{Tout,N}, sampling::TauSampling, al::Array{Tin,N}; dim=1) where {Tout,Tin,N}
    # Check dimensions
    expected_dims = collect(size(al))
    expected_dims[dim] = npoints(sampling)
    size(output) == tuple(expected_dims...) || throw(DimensionMismatch("Output array has wrong dimensions"))

    # Prepare arguments for C API
    ndim = N
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR

    if !_is_column_major_contiguous(al)
        error("Input array must be contiguous")
    end
    if !_is_column_major_contiguous(output)
        error("Output array must be contiguous")
    end

    # Call appropriate C function based on input/output types
    if Tin <: Real && Tout <: Real
        ret = C_API.spir_sampling_eval_dd(sampling.ptr, order, ndim, input_dims, target_dim, al, output)
    elseif Tin <: Complex && Tout <: Complex
        ret = C_API.spir_sampling_eval_zz(sampling.ptr, order, ndim, input_dims, target_dim, al, output)
    else
        error("Type combination not yet supported for TauSampling: input=$Tin, output=$Tout")
    end

    ret in [
        C_API.SPIR_INPUT_DIMENSION_MISMATCH,
        C_API.SPIR_OUTPUT_DIMENSION_MISMATCH,
        C_API.SPIR_INVALID_DIMENSION,
    ] && throw(DimensionMismatch("Failed to evaluate sampling: status=$ret"))
    return output
end

function evaluate!(output::Array{Tout,N}, sampling::MatsubaraSampling, al::Array{Tin,N}; dim=1) where {Tout,Tin,N}
    # Check dimensions
    expected_dims = collect(size(al))
    expected_dims[dim] = npoints(sampling)
    size(output) == tuple(expected_dims...) || throw(DimensionMismatch("Output array has wrong dimensions"))

    # Prepare arguments for C API
    ndim = N
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR

    if !_is_column_major_contiguous(al)
        error("Input array must be contiguous")
    end
    if !_is_column_major_contiguous(output)
        error("Output array must be contiguous")
    end

    # Call appropriate C function based on input/output types
    if Tin <: Real && Tout <: Complex
        ret = C_API.spir_sampling_eval_dz(sampling.ptr, order, ndim, input_dims, target_dim, al, output)
    elseif Tin <: Complex && Tout <: Complex
        ret = C_API.spir_sampling_eval_zz(sampling.ptr, order, ndim, input_dims, target_dim, al, output)
    else
        error("Type combination not supported for MatsubaraSampling: input=$Tin, output=$Tout")
    end

    ret in [
        C_API.SPIR_INPUT_DIMENSION_MISMATCH,
        C_API.SPIR_OUTPUT_DIMENSION_MISMATCH,
        C_API.SPIR_INVALID_DIMENSION,
    ] && throw(DimensionMismatch("Failed to evaluate sampling: status=$ret"))
    return output
end

"""
    fit(sampling::AbstractSampling, al::Array; dim=1)

Fit basis coefficients from values at sampling points using the C API.

For multidimensional arrays, `dim` specifies which dimension corresponds to the sampling points.
"""
function fit(sampling::Union{TauSampling,MatsubaraSampling}, al::Array{T,N}; dim=1) where {T,N}
    # Determine output dimensions
    output_dims = collect(size(al))
    output_dims[dim] = length(sampling.basis)

    # Determine output type - typically real for coefficients
    if sampling isa TauSampling
        # For complex input, we need complex output
        output_type = T
    else # MatsubaraSampling
        # For Matsubara sampling, we need to be careful about type matching
        # The C API might expect complex output even for real input
        output_type = T <: Complex ? T : ComplexF64
    end

    output = Array{output_type,N}(undef, output_dims...)
    fit!(output, sampling, al; dim=dim)

    # For MatsubaraSampling, if we want real coefficients, extract real part
    if sampling isa MatsubaraSampling && T <: Complex && output_type <: Complex
        # The fitted coefficients should be real for physical reasons
        # Extract real part and return as real array
        real_output = Array{real(output_type),N}(undef, output_dims...)
        real_output .= real.(output)
        return real_output
    end

    return output
end

"""
    fit!(output::Array, sampling::AbstractSampling, al::Array; dim=1)

In-place version of [`fit`](@ref). Write results to the pre-allocated `output` array.
"""
function fit!(output::Array{Tout,N}, sampling::TauSampling, al::Array{Tin,N}; dim=1) where {Tout,Tin,N}
    # Check dimensions
    expected_dims = collect(size(al))
    expected_dims[dim] = length(sampling.basis)
    size(output) == tuple(expected_dims...) || throw(DimensionMismatch("Output array has wrong dimensions"))

    # Prepare arguments for C API
    ndim = N
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR

    if !_is_column_major_contiguous(al)
        error("Input array must be contiguous")
    end
    if !_is_column_major_contiguous(output)
        error("Output array must be contiguous")
    end

    # Call appropriate C function
    if Tin <: Real && Tout <: Real
        ret = C_API.spir_sampling_fit_dd(sampling.ptr, order, ndim, input_dims, target_dim, al, output)
    elseif Tin <: Complex && Tout <: Complex
        ret = C_API.spir_sampling_fit_zz(sampling.ptr, order, ndim, input_dims, target_dim, al, output)
    else
        ArgumentError("Type combination not yet supported for TauSampling fit: input=$Tin, output=$Tout")
    end

    ret in [
        C_API.SPIR_INPUT_DIMENSION_MISMATCH,
        C_API.SPIR_OUTPUT_DIMENSION_MISMATCH,
        C_API.SPIR_INVALID_DIMENSION,
    ] && throw(DimensionMismatch("Failed to fit sampling: status=$ret"))
    return output
end

function fit!(output::Array{Tout,N}, sampling::MatsubaraSampling, al::Array{Tin,N}; dim=1) where {Tout,Tin,N}
    # Check dimensions
    expected_dims = collect(size(al))
    expected_dims[dim] = length(sampling.basis)
    size(output) == tuple(expected_dims...) || throw(DimensionMismatch("Output array has wrong dimensions"))

    # Prepare arguments for C API
    ndim = N
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR

    if !_is_column_major_contiguous(al)
        error("Input array must be contiguous")
    end
    if !_is_column_major_contiguous(output)
        error("Output array must be contiguous")
    end

    # Call appropriate C function based on input/output types
    if Tin <: Complex && Tout <: Complex
        # Use complex-to-complex API and then extract real part if needed
        ret = C_API.spir_sampling_fit_zz(sampling.ptr, order, ndim, input_dims, target_dim, al, output)
    elseif Tin <: Complex && Tout <: Real
        # Create temporary complex output, then extract real part
        temp_output = Array{ComplexF64,N}(undef, size(output)...)
        ret = C_API.spir_sampling_fit_zz(sampling.ptr, order, ndim, input_dims, target_dim, al, temp_output)
        ret == C_API.SPIR_COMPUTATION_SUCCESS || error("Failed to fit sampling: status=$ret")
        output .= real.(temp_output)
        return output
    else
        error("Type combination not supported for MatsubaraSampling fit: input=$Tin, output=$Tout")
    end

    ret in [
        C_API.SPIR_INPUT_DIMENSION_MISMATCH,
        C_API.SPIR_OUTPUT_DIMENSION_MISMATCH,
        C_API.SPIR_INVALID_DIMENSION,
    ] && throw(DimensionMismatch("Failed to fit sampling: status=$ret"))
    return output
end

# Convenience property accessors (similar to SparseIR.jl)
Base.getproperty(s::TauSampling, p::Symbol) = p === :tau ? sampling_points(s) : getfield(s, p)
Base.getproperty(s::MatsubaraSampling, p::Symbol) = p === :ωn ? sampling_points(s) : getfield(s, p)
