"""
    DiscreteLehmannRepresentation{S,B} <: AbstractBasis{S}

Discrete Lehmann representation (DLR) with poles selected according to extrema of IR.

This type wraps the C API DLR functionality. The DLR basis is a variant of the IR basis
that uses a "sketching" approach - representing functions as a linear combination of
poles on the real-frequency axis:

    G(iv) == sum(a[i] / (iv - w[i]) for i in 1:npoles)

# Fields

  - `ptr::Ptr{spir_basis}`: Pointer to the C DLR object
  - `basis::B`: The underlying IR basis
  - `poles::Vector{Float64}`: Pole locations on the real-frequency axis
"""
mutable struct DiscreteLehmannRepresentation{S<:Statistics,B<:AbstractBasis{S}} <:
               AbstractBasis{S}
    ptr::Ptr{spir_basis}
    basis::B
    poles::Vector{Float64}

    function DiscreteLehmannRepresentation{S,B}(ptr::Ptr{spir_basis}, basis::B,
            poles::Vector{Float64}) where {S<:Statistics,B<:AbstractBasis{S}}
        obj = new{S,B}(ptr, basis, poles)
        finalizer(s -> spir_basis_release(s.ptr), obj)
        return obj
    end
end

"""
    DiscreteLehmannRepresentation(basis::AbstractBasis, poles=default_omega_sampling_points(basis))

Construct a DLR basis from an IR basis.

If `poles` is not provided, uses the default omega sampling points from the IR basis.

`poles` may be any real-valued `AbstractVector` (including `Vector{Int}` and
`Vector{Float32}`); it is converted to `Vector{Float64}` — the element type the
C API reads — before the pointer is taken, so no narrower type is ever
reinterpreted as `Float64`. The poles must be finite and pairwise distinct;
otherwise an `ArgumentError` is thrown.
"""
function DiscreteLehmannRepresentation(basis::AbstractBasis,
        poles::AbstractVector{<:Real}=default_omega_sampling_points(basis))
    # Normalize the element type explicitly: the C entry point reads a
    # Ptr{Cdouble}, so the pointer must come from a Vector{Float64} we own.
    poles_d = convert(Vector{Float64}, poles)
    isempty(poles_d) && throw(ArgumentError("poles must not be empty"))
    _check_all_finite(poles_d, "poles")
    _check_unique(poles_d, "poles")

    status = Ref{Int32}(-100)
    dlr_ptr = GC.@preserve poles_d C_API.spir_dlr_new_with_poles(
        _get_ptr(basis), length(poles_d), pointer(poles_d), status)
    _check_status(status[], "spir_dlr_new_with_poles")
    _check_handle(dlr_ptr, "spir_dlr_new_with_poles")
    return DiscreteLehmannRepresentation{typeof(statistics(basis)),typeof(basis)}(
        dlr_ptr, basis, poles_d)
end

function DiscreteLehmannRepresentation(::AbstractBasis, poles::AbstractVector)
    throw(ArgumentError("poles must be a real-valued vector, got $(typeof(poles))"))
end

"""
    from_IR(dlr::DiscreteLehmannRepresentation, gl::Array, dims=1)

Transform from IR basis coefficients to DLR coefficients.

# Arguments

  - `dlr`: The DLR basis
  - `gl`: IR basis coefficients
  - `dims`: Dimension along which the basis coefficients are stored

# Returns

DLR coefficients with the same shape as input, but with size `length(dlr)` along dimension `dims`.

The element type of the result equals the element type of `gl`: `Float64` in,
`Float64` out; `ComplexF64` in, `ComplexF64` out. Only `Float64` and
`ComplexF64` input are supported — narrower element types (`Float32`,
`ComplexF32`, integers) throw `ArgumentError` rather than being reinterpreted
as `Float64`/`ComplexF64` at the C boundary.
"""
function from_IR(dlr::DiscreteLehmannRepresentation, gl::Array{T,N}, dims=1) where {T,N}
    # Validate the element type before any ccall: the C entry points read
    # Ptr{Cdouble} / Ptr{Complex64} (i.e. ComplexF64), and a narrower element
    # type handed to them reads out of bounds.
    T === Float64 || T === ComplexF64 ||
        throw(ArgumentError("from_IR supports Float64 and ComplexF64 input, got $T"))

    # Validate target dimension
    if dims < 1 || dims > N
        throw(ArgumentError("Invalid target dimension: $dims. Must be in range [1, $N]"))
    end

    # Check dimensions
    size(gl, dims) == length(dlr.basis) ||
        throw(DimensionMismatch("Input array has wrong size along dimension $dims"))

    # Prepare output dimensions
    output_dims = collect(size(gl))
    output_dims[dims] = length(dlr)

    # Determine output type
    output_type = T
    output = Array{output_type,N}(undef, output_dims...)

    # Safety checks
    if !_is_column_major_contiguous(gl)
        throw(ArgumentError("Input array must be contiguous"))
    end
    if !_is_column_major_contiguous(output)
        throw(ArgumentError("Output array must be contiguous"))
    end

    # Call appropriate C function
    ndim = N
    input_dims = Int32[size(gl)...]
    target_dim = Int32(dims - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR
    backend = _spir_default_backend[]
    if T === Float64
        ret = C_API.spir_ir2dlr_dd(
            dlr.ptr, backend, order, ndim, input_dims, target_dim, gl, output)
        op = "spir_ir2dlr_dd"
    else
        ret = C_API.spir_ir2dlr_zz(
            dlr.ptr, backend, order, ndim, input_dims, target_dim, gl, output)
        op = "spir_ir2dlr_zz"
    end

    _check_status(ret, op)
    return output
end

"""
    to_IR(dlr::DiscreteLehmannRepresentation, g_dlr::Array, dims=1)

Transform from DLR coefficients to IR basis coefficients.

# Arguments

  - `dlr`: The DLR basis
  - `g_dlr`: DLR coefficients
  - `dims`: Dimension along which the DLR coefficients are stored

# Returns

IR basis coefficients with the same shape as input, but with size `length(dlr.basis)` along dimension `dims`.

The element type of the result equals the element type of `g_dlr`. Only
`Float64` and `ComplexF64` input are supported — narrower element types
(`Float32`, `ComplexF32`, integers) throw `ArgumentError` rather than being
reinterpreted as `Float64`/`ComplexF64` at the C boundary.
"""
function to_IR(dlr::DiscreteLehmannRepresentation, g_dlr::Array{T,N}, dims=1) where {T,N}
    T === Float64 || T === ComplexF64 ||
        throw(ArgumentError("to_IR supports Float64 and ComplexF64 input, got $T"))

    # Validate target dimension
    if dims < 1 || dims > N
        throw(ArgumentError("Invalid target dimension: $dims. Must be in range [1, $N]"))
    end

    # Check dimensions
    size(g_dlr, dims) == length(dlr) ||
        throw(DimensionMismatch("Input array has wrong size along dimension $dims"))

    # Prepare output dimensions
    output_dims = collect(size(g_dlr))
    output_dims[dims] = length(dlr.basis)

    # Determine output type
    output_type = T
    output = Array{output_type,N}(undef, output_dims...)

    # Safety checks
    if !_is_column_major_contiguous(g_dlr)
        throw(ArgumentError("Input array must be contiguous"))
    end
    if !_is_column_major_contiguous(output)
        throw(ArgumentError("Output array must be contiguous"))
    end

    # Call appropriate C function
    ndim = N
    input_dims = Int32[size(g_dlr)...]
    target_dim = Int32(dims - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR

    backend = _spir_default_backend[]
    if T === Float64
        ret = C_API.spir_dlr2ir_dd(
            dlr.ptr, backend, order, ndim, input_dims, target_dim, g_dlr, output)
        op = "spir_dlr2ir_dd"
    else
        ret = C_API.spir_dlr2ir_zz(
            dlr.ptr, backend, order, ndim, input_dims, target_dim, g_dlr, output)
        op = "spir_dlr2ir_zz"
    end

    _check_status(ret, op)
    return output
end

# Pole access functions

"""
    npoles(dlr::DiscreteLehmannRepresentation)

Get the number of poles in the DLR basis.
"""
function npoles(dlr::DiscreteLehmannRepresentation)
    n_poles = Ref{Int32}(-1)
    ret = C_API.spir_dlr_get_npoles(dlr.ptr, n_poles)
    _check_status(ret, "spir_dlr_get_npoles")
    return Int(n_poles[])
end

"""
    get_poles(dlr::DiscreteLehmannRepresentation)

Get the pole locations for the DLR basis.

Returns a vector of pole locations on the real-frequency axis.
"""
function get_poles(dlr::DiscreteLehmannRepresentation)
    n = npoles(dlr)
    poles = Vector{Float64}(undef, n)
    ret = C_API.spir_dlr_get_poles(dlr.ptr, poles)
    _check_status(ret, "spir_dlr_get_poles")
    return poles
end

# Convenience function for getting default omega sampling points
"""
    default_omega_sampling_points(basis::AbstractBasis)

Get the default real-frequency sampling points for a basis.

These are the extrema of the highest-order basis function on the real-frequency axis,
which provide near-optimal conditioning for the DLR.
"""
function default_omega_sampling_points(basis::AbstractBasis)
    n_points = Ref{Int32}(-1)
    ret = C_API.spir_basis_get_n_default_ws(_get_ptr(basis), n_points)
    _check_status(ret, "spir_basis_get_n_default_ws")

    points = Vector{Float64}(undef, n_points[])
    ret = C_API.spir_basis_get_default_ws(_get_ptr(basis), points)
    _check_status(ret, "spir_basis_get_default_ws")

    return points
end

# AbstractBasis interface implementation

Base.length(dlr::DiscreteLehmannRepresentation) = length(dlr.poles)
Base.size(dlr::DiscreteLehmannRepresentation) = (length(dlr),)

# Pass through to underlying basis
β(dlr::DiscreteLehmannRepresentation) = β(dlr.basis)
ωmax(dlr::DiscreteLehmannRepresentation) = ωmax(dlr.basis)
Λ(dlr::DiscreteLehmannRepresentation) = Λ(dlr.basis)
accuracy(dlr::DiscreteLehmannRepresentation) = accuracy(dlr.basis)

# DLR-specific methods
sampling_points(dlr::DiscreteLehmannRepresentation) = dlr.poles
significance(dlr::DiscreteLehmannRepresentation) = ones(size(dlr))

function default_tau_sampling_points(dlr::DiscreteLehmannRepresentation; kwargs...)
    default_tau_sampling_points(dlr.basis; kwargs...)
end

function default_matsubara_sampling_points(dlr::DiscreteLehmannRepresentation; kwargs...)
    default_matsubara_sampling_points(dlr.basis; kwargs...)
end

# DLR is not as well-conditioned as IR
iswellconditioned(::DiscreteLehmannRepresentation) = false

# Accessor for the underlying basis
basis(dlr::DiscreteLehmannRepresentation) = dlr.basis
