
"""
TauSampling{T,B} <: AbstractSampling

Sparse sampling in imaginary time using the C API.

Allows transformation between IR basis coefficients and sampling points in imaginary time.
"""
mutable struct TauSampling{T<:Real,B<:AbstractBasis} <: AbstractSampling{T,Float64,Nothing}
    ptr::Ptr{spir_sampling}
    sampling_points::Vector{T}
    basis::B

    function TauSampling{T,B}(ptr::Ptr{spir_sampling}, sampling_points::Vector{T},
            basis::B) where {T<:Real,B<:AbstractBasis}
        obj = new{T,B}(ptr, sampling_points, basis)
        finalizer(s -> spir_sampling_release(s.ptr), obj)
        return obj
    end
end

const TauSampling64F = TauSampling{Float64,FiniteTempBasis{Fermionic,LogisticKernel}}
const TauSampling64B = TauSampling{Float64,FiniteTempBasis{Bosonic,LogisticKernel}}

"""
MatsubaraSampling{T,B} <: AbstractSampling

Sparse sampling in Matsubara frequencies using the C API.

Allows transformation between IR basis coefficients and sampling points in Matsubara frequencies.
"""
mutable struct MatsubaraSampling{T<:MatsubaraFreq,B<:AbstractBasis} <:
               AbstractSampling{T,ComplexF64,Nothing}
    ptr::Ptr{spir_sampling}
    sampling_points::Vector{T}
    positive_only::Bool
    basis::B

    function MatsubaraSampling{T,B}(ptr::Ptr{spir_sampling}, sampling_points::Vector{T},
            positive_only::Bool, basis::B) where {T<:MatsubaraFreq,B<:AbstractBasis}
        obj = new{T,B}(ptr, sampling_points, positive_only, basis)
        finalizer(s -> spir_sampling_release(s.ptr), obj)
        return obj
    end
end

const MatsubaraSampling64F = MatsubaraSampling{
    FermionicFreq,FiniteTempBasis{Fermionic,LogisticKernel}}
const MatsubaraSampling64B = MatsubaraSampling{
    BosonicFreq,FiniteTempBasis{Bosonic,LogisticKernel}}

# Convenience constructors

"""
    TauSampling(basis::AbstractBasis; sampling_points=nothing, use_positive_taus=true)

Construct a `TauSampling` object from a basis. If `sampling_points` is not provided,
the default tau sampling points from the basis are used.

If `use_positive_taus=true`, the sampling points are folded to the positive tau domain [0, β) [default].

If `use_positive_taus=false`, the sampling points are in the range [-β/2, β/2].

`sampling_points`, when given, may be any real-valued `AbstractVector`
(including `Vector{Int}` and `Vector{Float32}`); it is converted to
`Vector{Float64}` — the element type the C API reads — before the pointer is
taken, so a narrower element type is never reinterpreted as `Float64`. The
points must be non-empty, finite, pairwise distinct and inside `[-β, β]`;
otherwise an `ArgumentError` (or `DomainError` for the range) is thrown before
any call into `libsparseir`.
"""
function TauSampling(basis::AbstractBasis; sampling_points=nothing, use_positive_taus=true)
    if sampling_points === nothing
        sampling_points = default_tau_sampling_points(basis; use_positive_taus=use_positive_taus)
    end
    sampling_points isa AbstractVector{<:Real} || throw(ArgumentError(
        "sampling_points must be a real-valued vector, got $(typeof(sampling_points))"))

    # Validate and normalize BEFORE any ccall: the C entry point reads a
    # Ptr{Cdouble}, so the pointer must come from a Vector{Float64} we own.
    points = convert(Vector{Float64}, sampling_points)
    isempty(points) && throw(ArgumentError("sampling_points must not be empty"))
    _check_all_finite(points, "sampling_points")
    _check_unique(points, "sampling_points")
    βb = β(basis)
    for (i, τ) in enumerate(points)
        -βb ≤ τ ≤ βb || throw(DomainError(τ,
            "sampling_points[$i] must lie in [-β, β] = [$(-βb), $βb]"))
    end

    # Create sampling object with C_API
    status = Ref{Int32}(-100)
    ptr = GC.@preserve points C_API.spir_tau_sampling_new(
        _get_ptr(basis), length(points), pointer(points), status)
    _check_status(status[], "spir_tau_sampling_new")
    _check_handle(ptr, "spir_tau_sampling_new")

    return TauSampling{Float64,typeof(basis)}(ptr, points, basis)
end

"""
    MatsubaraSampling(basis::AbstractBasis; positive_only=false, sampling_points=nothing, factorize=true)

Construct a `MatsubaraSampling` object from a basis. If `sampling_points` is not provided,
the default Matsubara sampling points from the basis are used.

`positive_only = true` asserts that the caller's data satisfies the symmetry
`g(-iω) = conj(g(iω))`, i.e. that the underlying quantity is real in imaginary
time; the sampling object then holds only the non-negative frequencies. It is a
statement about the data, not a display option, and its default is `false` (the
general case). The assertion is **not** checked and cannot be checked from the
sampled values alone — see the warning in [`fit`](@ref) — so data violating it
is fitted to silently meaningless coefficients.

The sampling points must be non-empty and pairwise distinct; otherwise an
`ArgumentError` is thrown before any call into `libsparseir`.
"""
function MatsubaraSampling(
        basis::AbstractBasis; positive_only=false, sampling_points=nothing)
    if sampling_points === nothing
        # Get default Matsubara sampling points from basis
        status = Ref{Int32}(-100)
        n_points = Ref{Int32}(-1)
        basis_ptr = _get_ptr(basis)
        ret = C_API.spir_basis_get_n_default_matsus(
            basis_ptr, positive_only, n_points)
        _check_status(ret, "spir_basis_get_n_default_matsus")

        points_array = Vector{Int64}(undef, n_points[])
        ret = C_API.spir_basis_get_default_matsus(
            basis_ptr, positive_only, points_array)
        _check_status(ret, "spir_basis_get_default_matsus")

        # Convert to MatsubaraFreq objects based on statistics
        if statistics(basis) isa Fermionic
            sampling_points = [FermionicFreq(n) for n in points_array]
        else
            sampling_points = [BosonicFreq(n) for n in points_array]
        end
    else
        # Convert input to appropriate MatsubaraFreq type
        if statistics(basis) isa Fermionic
            sampling_points = [p isa FermionicFreq ? p : FermionicFreq(Int(p))
                               for p in sampling_points]
        else
            sampling_points = [p isa BosonicFreq ? p : BosonicFreq(Int(p))
                               for p in sampling_points]
        end
    end

    # Extract indices for the C API; the entry point reads a Ptr{Int64}, so the
    # pointer is taken from this Vector{Int64}.
    indices = Int64[Int64(Int(p)) for p in sampling_points]

    # Safety checks, all before the ccall
    isempty(indices) && throw(ArgumentError("sampling_points must not be empty"))
    _check_unique(indices, "sampling_points")

    status = Ref{Int32}(-100)
    ptr = GC.@preserve indices C_API.spir_matsu_sampling_new(
        _get_ptr(basis), positive_only, length(indices), pointer(indices), status)
    _check_status(status[], "spir_matsu_sampling_new")
    _check_handle(ptr, "spir_matsu_sampling_new")

    return MatsubaraSampling{eltype(sampling_points),typeof(basis)}(
        ptr, sampling_points, positive_only, basis)
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
    _check_status(ret, "spir_sampling_get_npoints")
    return Int(n_points[])
end

# Evaluation and fitting functions

"""
    evaluate(sampling::AbstractSampling, al::Array; dim=1)

Evaluate basis coefficients at the sampling points using the C API.

For multidimensional arrays, `dim` specifies which dimension corresponds to the basis coefficients.
"""
function evaluate(
        sampling::Union{TauSampling,MatsubaraSampling}, al::Array{
            T,N}; dim=1) where {T,N}
    # Determine output dimensions
    if dim < 1 || dim > N
        throw(ArgumentError("dim $(dim) is invalid: must be in 1:$N"))
    end
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
function evaluate!(
        output::Array{Tout,N}, sampling::TauSampling, al::Array{
            Tin,N}; dim=1) where {Tout,Tin,N}
    # Check dimensions
    if dim < 1 || dim > N
        throw(ArgumentError("dim $(dim) is invalid: must be in 1:$N"))
    end
    expected_dims = collect(size(al))
    expected_dims[dim] = npoints(sampling)
    size(output) == tuple(expected_dims...) ||
        throw(DimensionMismatch("Output array has wrong dimensions"))

    # Prepare arguments for C API
    ndim = N
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR

    if !_is_column_major_contiguous(al)
        throw(ArgumentError("Input array must be contiguous"))
    end
    if !_is_column_major_contiguous(output)
        throw(ArgumentError("Output array must be contiguous"))
    end

    # Call appropriate C function based on input/output types
    backend = _spir_default_backend[]
    if Tin == Float64 && Tout == Float64
        ret = C_API.spir_sampling_eval_dd(
            sampling.ptr, backend, order, ndim, input_dims, target_dim, al, output)
        op = "spir_sampling_eval_dd"
    elseif Tin == ComplexF64 && Tout == ComplexF64
        ret = C_API.spir_sampling_eval_zz(
            sampling.ptr, backend, order, ndim, input_dims, target_dim, al, output)
        op = "spir_sampling_eval_zz"
    else
        throw(ArgumentError("Type combination not supported for TauSampling evaluate!: input=$Tin, output=$Tout"))
    end

    # Handle by success: every status other than SPIR_COMPUTATION_SUCCESS is an
    # error, including codes this wrapper does not enumerate.
    _check_status(ret, op)
    return output
end

function evaluate!(output::Array{Tout,N}, sampling::MatsubaraSampling,
        al::Array{Tin,N}; dim=1) where {Tout,Tin,N}
    # Check dimensions
    if dim < 1 || dim > N
        throw(ArgumentError("dim $(dim) is invalid: must be in 1:$N"))
    end
    expected_dims = collect(size(al))
    expected_dims[dim] = npoints(sampling)
    size(output) == tuple(expected_dims...) ||
        throw(DimensionMismatch("Output array has wrong dimensions"))

    # Prepare arguments for C API
    ndim = N
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR

    if !_is_column_major_contiguous(al)
        throw(ArgumentError("Input array must be contiguous"))
    end
    if !_is_column_major_contiguous(output)
        throw(ArgumentError("Output array must be contiguous"))
    end

    # Call appropriate C function based on input/output types
    backend = _spir_default_backend[]
    if Tin == Float64 && Tout == ComplexF64
        ret = C_API.spir_sampling_eval_dz(
            sampling.ptr, backend, order, ndim, input_dims, target_dim, al, output)
        op = "spir_sampling_eval_dz"
    elseif Tin == ComplexF64 && Tout == ComplexF64
        ret = C_API.spir_sampling_eval_zz(
            sampling.ptr, backend, order, ndim, input_dims, target_dim, al, output)
        op = "spir_sampling_eval_zz"
    else
        throw(ArgumentError("Type combination not supported for MatsubaraSampling evaluate!: input=$Tin, output=$Tout"))
    end

    _check_status(ret, op)
    return output
end

"""
    fit(sampling::AbstractSampling, al::Array; dim=1)

Fit basis coefficients from values at sampling points using the C API.

For multidimensional arrays, `dim` specifies which dimension corresponds to the sampling points.

# Element type of the result

  - `TauSampling`: the element type of `al` (`Float64` in → `Float64` out,
    `ComplexF64` in → `ComplexF64` out).
  - `MatsubaraSampling`: always `ComplexF64`, because the IR expansion
    coefficients of a general Green's function are complex. The imaginary part
    is never projected away.

Only `Float64` and `ComplexF64` input are supported; anything else throws
`ArgumentError`.

# `positive_only`

When `sampling` was built with `positive_only = true`, the caller asserts the
symmetry `g(-iω) = conj(g(iω))` — equivalently, that the underlying quantity is
real in imaginary time, so that its IR coefficients are real. Only the
non-negative frequencies are then sampled, and each complex sampling point
contributes two real equations, so the fit solves an exactly determined *real*
system and always returns coefficients whose imaginary part is exactly zero.

!!! warning "`positive_only = true` is an unchecked contract"

    Because the default point set makes the real system exactly determined, data
    that violates `g(-iω) = conj(g(iω))` is fitted with a vanishing residual and
    produces silently meaningless coefficients: the violation cannot be detected
    from the sampled values alone, and neither this wrapper nor `libsparseir`
    raises. Use `positive_only = true` only for a quantity you know to be real
    in imaginary time; otherwise use the default `positive_only = false`.
"""
function fit(
        sampling::Union{TauSampling,MatsubaraSampling}, al::Array{T,N}; dim=1) where {
        T,N}
    if !(T ∈ [Float64, ComplexF64])
        throw(ArgumentError("Type not supported for fit: input=$T (expected Float64 or ComplexF64)"))
    end
    if dim < 1 || dim > N
        throw(ArgumentError("dim $(dim) is invalid: must be in 1:$N"))
    end
    # Determine output dimensions
    output_dims = collect(size(al))
    output_dims[dim] = length(sampling.basis)

    # Determine output type
    if sampling isa TauSampling
        output_type = T
    else # MatsubaraSampling
        # For Matsubara sampling, we need to be careful about type matching
        # The C API might expect complex output even for real input
        output_type = ComplexF64
    end

    output = Array{output_type,N}(undef, output_dims...)
    fit!(output, sampling, al; dim=dim)

    return output
end

"""
    fit!(output::Array, sampling::AbstractSampling, al::Array; dim=1)

In-place version of [`fit`](@ref). Write results to the pre-allocated `output` array.
"""
function fit!(
        output::Array{Tout,N}, sampling::TauSampling, al::Array{
            Tin,N}; dim=1) where {Tout,Tin,N}
    # Check dimensions
    if dim < 1 || dim > N
        throw(ArgumentError("dim $(dim) is invalid: must be in 1:$N"))
    end
    if !(Tin ∈ [Float64, ComplexF64])
        throw(ArgumentError("Type not supported for TauSampling fit: input=$Tin"))
    end
    if !(Tout ∈ [Float64, ComplexF64])
        throw(ArgumentError("Type not supported for TauSampling fit: output=$Tout"))
    end
    expected_dims = collect(size(al))
    expected_dims[dim] = length(sampling.basis)
    size(output) == tuple(expected_dims...) ||
        throw(DimensionMismatch("Output array has wrong dimensions"))

    # Prepare arguments for C API
    ndim = N
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR

    if !_is_column_major_contiguous(al)
        throw(ArgumentError("Input array must be contiguous"))
    end
    if !_is_column_major_contiguous(output)
        throw(ArgumentError("Output array must be contiguous"))
    end
    backend = _spir_default_backend[]
    # Call appropriate C function
    if Tin == Float64 && Tout == Float64
        ret = C_API.spir_sampling_fit_dd(
            sampling.ptr, backend, order, ndim, input_dims, target_dim, al, output)
        op = "spir_sampling_fit_dd"
    elseif Tin == ComplexF64 && Tout == ComplexF64
        ret = C_API.spir_sampling_fit_zz(
            sampling.ptr, backend, order, ndim, input_dims, target_dim, al, output)
        op = "spir_sampling_fit_zz"
    else
        throw(ArgumentError("Type combination not supported for TauSampling fit!: input=$Tin, output=$Tout"))
    end

    _check_status(ret, op)
    return output
end

function fit!(
        output::Array{Tout,N}, sampling::MatsubaraSampling, al::Array{
            Tin,N}; dim=1) where {Tout,Tin,N}
    # Check dimensions
    if dim < 1 || dim > N
        throw(ArgumentError("dim $(dim) is invalid: must be in 1:$N"))
    end
    expected_dims = collect(size(al))
    expected_dims[dim] = length(sampling.basis)
    size(output) == tuple(expected_dims...) ||
        throw(DimensionMismatch("Output array has wrong dimensions"))

    # Prepare arguments for C API
    ndim = N
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR

    if !_is_column_major_contiguous(al)
        throw(ArgumentError("Input array must be contiguous"))
    end
    if !_is_column_major_contiguous(output)
        throw(ArgumentError("Output array must be contiguous"))
    end

    # Call appropriate C function based on input/output types
    backend = _spir_default_backend[]
    if Tin == ComplexF64 && Tout == ComplexF64
        ret = C_API.spir_sampling_fit_zz(
            sampling.ptr, backend, order, ndim, input_dims, target_dim, al, output)
        _check_status(ret, "spir_sampling_fit_zz")
        return output
    elseif Tin == ComplexF64 && Tout == Float64
        # Real output was explicitly requested. Fit in full complex arithmetic,
        # then report — rather than silently discard — a non-negligible
        # imaginary part, which means the coefficients are genuinely complex.
        temp_output = Array{ComplexF64,N}(undef, size(output)...)
        ret = C_API.spir_sampling_fit_zz(
            sampling.ptr, backend, order, ndim, input_dims, target_dim, al, temp_output)
        _check_status(ret, "spir_sampling_fit_zz")
        max_imag = isempty(temp_output) ? 0.0 : maximum(abs ∘ imag, temp_output)
        scale = isempty(temp_output) ? 0.0 : maximum(abs ∘ real, temp_output)
        tol = _imag_tolerance(sampling)
        if max_imag > tol * max(scale, one(scale))
            throw(ArgumentError("real output requested, but the fitted coefficients are \
                                 genuinely complex (max |imag| = $max_imag, max |real| = \
                                 $scale, tolerance = $(tol * max(scale, one(scale)))). \
                                 Fit into a ComplexF64 array instead."))
        end
        output .= real.(temp_output)
        return output
    else
        throw(ArgumentError("Type combination not supported for MatsubaraSampling fit!: input=$Tin, output=$Tout"))
    end
end

# Coefficients are compared against the accuracy the basis was built for, not
# against machine epsilon: a basis with target accuracy `eps` only resolves
# quantities down to `eps`.
function _imag_tolerance(sampling::MatsubaraSampling)
    acc = accuracy(sampling.basis)
    return max(isfinite(acc) ? 10 * acc : 1e-8, 1e-12)
end

# Convenience property accessors (similar to SparseIR.jl)
function Base.getproperty(s::TauSampling, p::Symbol)
    p === :tau ? sampling_points(s) :
    getfield(s, p)
end
function Base.getproperty(s::MatsubaraSampling, p::Symbol)
    p === :ωn ? sampling_points(s) :
    getfield(s, p)
end
