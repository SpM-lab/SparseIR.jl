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
  - `u`: the DLR basis functions in imaginary time, `u[i](τ)` being the
    single-pole function `-exp(-τ ω_i) / (1 + exp(-β ω_i))`, so that
    `transpose(dlr.u(τ)) * g_dlr` evaluates DLR coefficients `g_dlr`
  - `uhat`: their Fourier transforms, `1/(iν - ω_i)` for fermions and
    `tanh(β ω_i / 2)/(iν - ω_i)` for bosons

The DLR basis functions are not piecewise polynomials: `deriv`, `knots` and
`overlap` are not supported for them and throw [`SparseIRError`](@ref).
"""
mutable struct DiscreteLehmannRepresentation{S<:Statistics,B<:AbstractBasis{S}} <:
               AbstractBasis{S}
    ptr::Ptr{spir_basis}
    basis::B
    poles::Vector{Float64}
    u::PiecewiseLegendrePolyVector
    uhat::PiecewiseLegendreFTVector

    function DiscreteLehmannRepresentation{S,B}(ptr::Ptr{spir_basis}, basis::B,
            poles::Vector{Float64}) where {S<:Statistics,B<:AbstractBasis{S}}
        # The DLR basis functions come from the DLR handle itself; they are the
        # single-pole functions, not the functions of the underlying IR basis.
        βb = β(basis)
        status = Ref{Int32}(-100)
        u_ptr = uhat_ptr = Ptr{spir_funcs}(C_NULL)
        try
            u_ptr = C_API.spir_basis_get_u(ptr, status)
            _check_status(status[], "spir_basis_get_u")
            _check_handle(u_ptr, "spir_basis_get_u")
            uhat_ptr = C_API.spir_basis_get_uhat(ptr, status)
            _check_status(status[], "spir_basis_get_uhat")
            _check_handle(uhat_ptr, "spir_basis_get_uhat")
        catch
            # Nothing owns the handles yet: release them before rethrowing.
            u_ptr == C_NULL || spir_funcs_release(u_ptr)
            uhat_ptr == C_NULL || spir_funcs_release(uhat_ptr)
            spir_basis_release(ptr)
            rethrow()
        end
        u = PiecewiseLegendrePolyVector(u_ptr, -βb, βb, βb, (0.0, βb))
        uhat = PiecewiseLegendreFTVector(uhat_ptr, zeta(S()))
        obj = new{S,B}(ptr, basis, poles, u, uhat)
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
reinterpreted as `Float64`. The poles must be finite and pairwise distinct
(otherwise `ArgumentError`) and lie in `[-ωmax, ωmax]` (otherwise
`DomainError`).
"""
function DiscreteLehmannRepresentation(basis::FiniteTempBasis,
        poles::AbstractVector{<:Real}=default_omega_sampling_points(basis))
    # Normalize the element type explicitly: the C entry point reads a
    # Ptr{Cdouble}, so the pointer must come from a Vector{Float64} we own.
    poles_d = convert(Vector{Float64}, poles)
    isempty(poles_d) && throw(ArgumentError("poles must not be empty"))
    _check_all_finite(poles_d, "poles")
    _check_unique(poles_d, "poles")
    # The C library panics on a pole outside the frequency window
    # (SpM-lab/sparse-ir-rs#266).
    for ω in poles_d
        abs(ω) ≤ ωmax(basis) ||
            throw(DomainError(
                ω, "poles must lie in [-ωmax, ωmax] = [$(-ωmax(basis)), $(ωmax(basis))]"))
    end

    status = Ref{Int32}(-100)
    dlr_ptr = GC.@preserve poles_d C_API.spir_dlr_new_with_poles(
        _get_ptr(basis), length(poles_d), pointer(poles_d), status)
    _check_status(status[], "spir_dlr_new_with_poles")
    _check_handle(dlr_ptr, "spir_dlr_new_with_poles")
    return DiscreteLehmannRepresentation{typeof(statistics(basis)),typeof(basis)}(
        dlr_ptr, basis, poles_d)
end

function DiscreteLehmannRepresentation(::FiniteTempBasis, poles::AbstractVector)
    throw(ArgumentError("poles must be a real-valued vector, got $(typeof(poles))"))
end

# The DLR is built on the C handle of an IR basis.
function DiscreteLehmannRepresentation(basis::AbstractBasis, poles...)
    throw(ArgumentError("a DiscreteLehmannRepresentation is built on a FiniteTempBasis, \
                         got $(nameof(typeof(basis)))"))
end

u(dlr::DiscreteLehmannRepresentation) = dlr.u
uhat(dlr::DiscreteLehmannRepresentation) = dlr.uhat

"""
    from_IR(dlr::DiscreteLehmannRepresentation, gl::AbstractArray, dims=1)

Transform from IR basis coefficients to DLR coefficients.

# Arguments

  - `dlr`: The DLR basis
  - `gl`: IR basis coefficients
  - `dims`: Dimension along which the basis coefficients are stored

# Returns

DLR coefficients with the same shape as input, but with size `length(dlr)` along dimension `dims`.

`gl` may be any `AbstractArray` with a real or complex element type. It is
converted to `Array{Float64}` or `Array{ComplexF64}` — the element types the C
entry points read — before the call, so `Float32`, `ComplexF32` or integer input
contributes only its own precision. The result is `Float64` for real input and
`ComplexF64` for complex input. Non-finite entries throw `ArgumentError`, a
wrong length along `dims` `DimensionMismatch`.
"""
function from_IR(dlr::DiscreteLehmannRepresentation, gl::AbstractArray, dims=1)
    gl = _as_input_array(gl, "IR coefficients")
    return _dlr_transform(dlr, gl, dims, length(dlr.basis), length(dlr), true)
end

"""
    to_IR(dlr::DiscreteLehmannRepresentation, g_dlr::AbstractArray, dims=1)

Transform from DLR coefficients to IR basis coefficients.

# Arguments

  - `dlr`: The DLR basis
  - `g_dlr`: DLR coefficients
  - `dims`: Dimension along which the DLR coefficients are stored

# Returns

IR basis coefficients with the same shape as input, but with size `length(dlr.basis)` along dimension `dims`.

Element types, conversion and validation are as for [`from_IR`](@ref).
"""
function to_IR(dlr::DiscreteLehmannRepresentation, g_dlr::AbstractArray, dims=1)
    g_dlr = _as_input_array(g_dlr, "DLR coefficients")
    return _dlr_transform(dlr, g_dlr, dims, length(dlr), length(dlr.basis), false)
end

function _dlr_transform(dlr::DiscreteLehmannRepresentation, input::Array{T,N}, dims,
        n_in::Int, n_out::Int, ir_to_dlr::Bool) where {T,N}
    dims isa Integer && 1 ≤ dims ≤ N ||
        throw(ArgumentError("Invalid target dimension: $dims. Must be in range [1, $N]"))
    size(input, dims) == n_in ||
        throw(DimensionMismatch("Input array has length $(size(input, dims)) along \
                                 dimension $dims, expected $n_in"))
    output_dims = collect(size(input))
    output_dims[dims] = n_out
    output = Array{T,N}(undef, output_dims...)

    input_dims = Int32[size(input)...]
    target_dim = Int32(dims - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR
    backend = _spir_default_backend[]
    if ir_to_dlr
        f, op = T === Float64 ? (C_API.spir_ir2dlr_dd, "spir_ir2dlr_dd") :
                (C_API.spir_ir2dlr_zz, "spir_ir2dlr_zz")
    else
        f, op = T === Float64 ? (C_API.spir_dlr2ir_dd, "spir_dlr2ir_dd") :
                (C_API.spir_dlr2ir_zz, "spir_dlr2ir_zz")
    end
    ret = f(dlr.ptr, backend, order, N, input_dims, target_dim, input, output)
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
