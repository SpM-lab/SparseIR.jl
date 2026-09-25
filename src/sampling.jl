
"""
TauSampling{T,B} <: AbstractSampling

Sparse sampling in imaginary time using the C API.

Allows transformation between IR basis coefficients `G_l` and the values
`G(τ_i) = Σ_l G_l U_l(τ_i)` at the sampling points `τ_i` in imaginary time.
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

Allows transformation between IR basis coefficients `G_l` and the values
`G(iν_i) = Σ_l G_l Û_l(iν_i)` at the sampling frequencies `ν_i = n_i π/β`.
"""
mutable struct MatsubaraSampling{T<:MatsubaraFreq,B<:AbstractBasis} <:
               AbstractSampling{T,ComplexF64,Nothing}
    ptr::Ptr{spir_sampling}
    sampling_points::Vector{T}
    positive_only::Bool
    basis::B
    # The C library holds the points in ascending order. `order` maps that order
    # to `sampling_points` (sorted = sampling_points[order]) and `inverse_order`
    # back; both are empty when the points are sorted already.
    order::Vector{Int}
    inverse_order::Vector{Int}

    function MatsubaraSampling{T,B}(ptr::Ptr{spir_sampling}, sampling_points::Vector{T},
            positive_only::Bool, basis::B, order::Vector{Int}=Int[]
    ) where {T<:MatsubaraFreq,B<:AbstractBasis}
        obj = new{T,B}(ptr, sampling_points, positive_only, basis, order,
            isempty(order) ? Int[] : invperm(order))
        finalizer(s -> spir_sampling_release(s.ptr), obj)
        return obj
    end
end

# The permutation that sorts the points, or an empty vector if they are sorted.
_matsubara_order(indices::Vector{Int64}) = issorted(indices) ? Int[] : sortperm(indices)

const MatsubaraSampling64F = MatsubaraSampling{
    FermionicFreq,FiniteTempBasis{Fermionic,LogisticKernel}}
const MatsubaraSampling64B = MatsubaraSampling{
    BosonicFreq,FiniteTempBasis{Bosonic,LogisticKernel}}

# Convenience constructors

"""
    TauSampling(basis::AbstractBasis; sampling_points=nothing, use_positive_taus=true)

Construct a `TauSampling` object from a basis. If `sampling_points` is not provided,
the default tau sampling points from the basis are used: the roots of `U_L`,
the first basis function beyond a basis of size `L` (see
[`default_tau_sampling_points`](@ref)).

If `use_positive_taus=true`, the sampling points are folded into `(0, β)` and sorted [default].

If `use_positive_taus=false`, the sampling points are unfolded, in `(-β/2, β/2]`:
pairs ±τ, plus β/2 when their number is odd.

For an [`AugmentedBasis`](@ref) there is no `use_positive_taus` keyword: its
default points, the roots of `U_L` for `L = length(basis)`, are always folded
into `(0, β)`.

Given points are read as for `basis.u`: `τ < 0` stands for `τ + β` with the
sign `(-1)^ζ`, and `-0.0` is `0⁻` (see [`FiniteTempBasis`](@ref)).

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
        sampling_points = default_tau_sampling_points(
            basis; use_positive_taus=use_positive_taus)
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
    MatsubaraSampling(basis::AbstractBasis; positive_only=false, sampling_points=nothing)

Construct a `MatsubaraSampling` object from a basis. If `sampling_points` is not provided,
the default Matsubara sampling points from the basis are used: the sign changes of
the first discarded transform `Û_l`, with `l ≥ L = length(basis)` chosen to fit
the parity (see [`default_matsubara_sampling_points`](@ref)). Bosonic sets always
include `n = 0`.

`sampling_points`, when given, are reduced frequencies `n` (integers of the
parity of the statistics: odd for fermions, even for bosons) or
[`MatsubaraFreq`](@ref)s of the statistics of the basis; they are stored, and
returned by [`sampling_points`](@ref), as a `Vector{FermionicFreq}` or
`Vector{BosonicFreq}`.

`positive_only = true` asserts that the caller's data satisfies the symmetry
`G(-iν) = conj(G(iν))`, i.e. that the underlying quantity is real in imaginary
time; the sampling object then holds only the non-negative frequencies, `n ≥ 0`
(for bosons including `n = 0`). It is a statement about the data, not a display
option, and its default is `false` (the general case). The assertion is **not**
checked and cannot be checked from the sampled values alone — see the warning in
[`fit`](@ref) — so data violating it is fitted to silently meaningless
coefficients.

The sampling points must be non-empty and pairwise distinct; otherwise an
`ArgumentError` is thrown before any call into `libsparseir`. They may be given
in any order: [`evaluate`](@ref) and [`fit`](@ref) follow the order of
`sampling_points`.
"""
function MatsubaraSampling(
        basis::AbstractBasis; positive_only=false, sampling_points=nothing)
    if sampling_points === nothing
        # Get default Matsubara sampling points from basis
        status = Ref{Int32}(-100)
        n_points = Ref{Int32}(-1)
        # A DLR has no default Matsubara points of its own in the C API; it
        # samples at those of its IR basis, as in the Python wrapper.
        source = basis isa DiscreteLehmannRepresentation ? basis.basis : basis
        basis_ptr = _get_ptr(source)
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
        S = typeof(statistics(basis))
        sampling_points = MatsubaraFreq{S}[_to_freq(S, p) for p in sampling_points]
    end

    # Extract indices for the C API; the entry point reads a Ptr{Int64}, so the
    # pointer is taken from this Vector{Int64}.
    indices = Int64[Int64(Int(p)) for p in sampling_points]

    # Safety checks, all before the ccall
    isempty(indices) && throw(ArgumentError("sampling_points must not be empty"))
    _check_unique(indices, "sampling_points")
    # The C library panics on a negative point in the positive-only case
    # (SpM-lab/sparse-ir-rs#247).
    if positive_only && any(<(0), indices)
        throw(ArgumentError("positive_only=true requires non-negative sampling points, \
                             got $(first(filter(<(0), indices)))π/β"))
    end

    # The C library orders the points ascending; pass them sorted and keep the
    # permutation, so that evaluate and fit follow the order of `sampling_points`.
    order = _matsubara_order(indices)
    c_indices = isempty(order) ? indices : indices[order]

    status = Ref{Int32}(-100)
    ptr = GC.@preserve c_indices C_API.spir_matsu_sampling_new(
        _get_ptr(basis), positive_only, length(c_indices), pointer(c_indices), status)
    _check_status(status[], "spir_matsu_sampling_new")
    _check_handle(ptr, "spir_matsu_sampling_new")

    return MatsubaraSampling{eltype(sampling_points),typeof(basis)}(
        ptr, sampling_points, positive_only, basis, order)
end

# A sampling point given as a number must be an exact integer of the right
# parity; a `MatsubaraFreq` must have the statistics of the basis.
_to_freq(::Type{S}, p::MatsubaraFreq{S}) where {S<:Statistics} = p
function _to_freq(::Type{S}, p::MatsubaraFreq) where {S<:Statistics}
    throw(ArgumentError("sampling point $(Int(p))π/β is \
                         $(nameof(typeof(statistics(p)))), but the basis is $(nameof(S))"))
end
function _to_freq(::Type{S}, p::Real) where {S<:Statistics}
    isinteger(p) && typemin(Int) ≤ p ≤ typemax(Int) || throw(ArgumentError(
        "Matsubara sampling points must be integers, got $p (no rounding is performed)"))
    return MatsubaraFreq{S}(Int(p))   # DomainError for the wrong parity
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
#
# The public methods accept any AbstractArray: they convert it with
# `_as_input_array` and validate `dim` and the length along it. The
# `_evaluate!`/`_fit!` kernels below receive only the dense `Array{Float64}` or
# `Array{ComplexF64}` that the C API reads, laid out column-major.

"""
    evaluate(sampling::AbstractSampling, al::AbstractArray; dim=1)

Evaluate basis coefficients at the sampling points using the C API.

For multidimensional arrays, `dim` specifies which dimension corresponds to the basis coefficients.

`al` may be any `AbstractArray` with a real or complex element type: it is
converted to `Array{Float64}` or `Array{ComplexF64}` (the types the C API reads)
before the call, so narrower types such as `Float32` contribute only their own
precision. The result is `Float64`/`ComplexF64` for `TauSampling` (following the
input) and always `ComplexF64` for `MatsubaraSampling`. Non-finite entries and a
wrong length along `dim` throw before the call.

For a `MatsubaraSampling` built with `positive_only = true`, genuinely complex
coefficients violate its assumption of real coefficients and throw
`ArgumentError`.

The result holds `G(τ_i) = Σ_l G_l U_l(τ_i)` for a `TauSampling` and
`G(iν_i) = Σ_l G_l Û_l(iν_i)` for a `MatsubaraSampling`.
"""
function evaluate(
        sampling::Union{TauSampling,MatsubaraSampling}, al::AbstractArray{<:Any,N};
        dim=1) where {N}
    al = _as_input_array(al, "basis coefficients")
    _check_sampling_dims(al, dim, length(sampling.basis), "basis coefficients")
    output_dims = collect(size(al))
    output_dims[dim] = npoints(sampling)
    output_type = sampling isa TauSampling ? eltype(al) : ComplexF64
    output = Array{output_type,N}(undef, output_dims...)
    return _evaluate!(output, sampling, al, dim)
end

"""
    evaluate!(output::Array, sampling::AbstractSampling, al::AbstractArray; dim=1)

In-place version of [`evaluate`](@ref). Write results to the pre-allocated
`output`, which must be an `Array{Float64}` or `Array{ComplexF64}` of the right
shape (`ComplexF64` for `MatsubaraSampling`); any other `output` throws
`ArgumentError`.
"""
function evaluate!(output::AbstractArray, sampling::Union{TauSampling,MatsubaraSampling},
        al::AbstractArray; dim=1)
    _check_output_buffer(output)
    al = _as_input_array(al, "basis coefficients")
    _check_no_alias(output, al)
    _check_sampling_dims(al, dim, length(sampling.basis), "basis coefficients")
    return _evaluate!(output, sampling, al, dim)
end

"""
    fit(sampling::AbstractSampling, al::AbstractArray; dim=1)

Fit basis coefficients from values at sampling points using the C API.

This is the least-squares inverse of [`evaluate`](@ref): it returns the IR
coefficients `G_l` whose values `Σ_l G_l U_l(τ_i)` or `Σ_l G_l Û_l(iν_i)` best
match the given values at the sampling points.

For multidimensional arrays, `dim` specifies which dimension corresponds to the sampling points.

# Element type of the result

  - `TauSampling`: `Float64` for real input, `ComplexF64` for complex input.
  - `MatsubaraSampling`: always `ComplexF64`, because the IR expansion
    coefficients of a general Green's function are complex. The imaginary part
    is never projected away.

`al` may be any `AbstractArray` with a real or complex element type; it is
converted to `Array{Float64}` or `Array{ComplexF64}` before the call (real
Matsubara data to `ComplexF64`), so narrower types contribute only their own
precision. Non-finite entries and a wrong length along `dim` throw before the
call.

# `positive_only`

When `sampling` was built with `positive_only = true`, the caller asserts the
symmetry `G(-iν) = conj(G(iν))` — equivalently, that the underlying quantity is
real in imaginary time, so that its IR coefficients are real. Only the
non-negative frequencies are then sampled, and each complex sampling point
contributes two real equations, so the fit solves an exactly determined *real*
system and always returns coefficients whose imaginary part is exactly zero.

!!! warning "`positive_only = true` is an unchecked contract"

    Because the default point set makes the real system exactly determined, data
    that violates `G(-iν) = conj(G(iν))` is fitted with a vanishing residual and
    produces silently meaningless coefficients: the violation cannot be detected
    from the sampled values alone, and neither this wrapper nor `libsparseir`
    raises. Use `positive_only = true` only for a quantity you know to be real
    in imaginary time; otherwise use the default `positive_only = false`.
"""
function fit(
        sampling::Union{TauSampling,MatsubaraSampling}, al::AbstractArray{<:Any,N};
        dim=1) where {N}
    al = _as_fit_input(sampling, al)
    _check_sampling_dims(al, dim, npoints(sampling), "values at the sampling points")
    output_dims = collect(size(al))
    output_dims[dim] = length(sampling.basis)
    # TauSampling follows the input; MatsubaraSampling is always complex.
    output_type = sampling isa TauSampling ? eltype(al) : ComplexF64
    output = Array{output_type,N}(undef, output_dims...)
    return _fit!(output, sampling, al, dim)
end

"""
    fit!(output::Array, sampling::AbstractSampling, al::AbstractArray; dim=1)

In-place version of [`fit`](@ref). Write results to the pre-allocated
`output`, which must be an `Array{Float64}` or `Array{ComplexF64}` of the right
shape; any other `output` throws `ArgumentError`. A `Float64` output for a
`MatsubaraSampling` is accepted only if the fitted coefficients are real to the
accuracy of the basis.
"""
function fit!(output::AbstractArray, sampling::Union{TauSampling,MatsubaraSampling},
        al::AbstractArray; dim=1)
    _check_output_buffer(output)
    al = _as_fit_input(sampling, al)
    _check_no_alias(output, al)
    _check_sampling_dims(al, dim, npoints(sampling), "values at the sampling points")
    return _fit!(output, sampling, al, dim)
end

_as_fit_input(::TauSampling, al) = _as_input_array(al, "values at the sampling points")
function _as_fit_input(::MatsubaraSampling, al::AbstractArray{<:Any,N}) where {N}
    # Only the complex C entry point exists; real data is a special case of it.
    return convert(
        Array{ComplexF64,N}, _as_input_array(al, "values at the sampling points"))
end

function _check_sampling_dims(a::AbstractArray{<:Any,N}, dim, n, name) where {N}
    dim isa Integer && 1 ≤ dim ≤ N ||
        throw(ArgumentError("dim $(dim) is invalid: must be in 1:$N"))
    size(a, dim) == n || throw(DimensionMismatch(
        "$name has length $(size(a, dim)) along dimension $dim, expected $n"))
    return nothing
end

# The C library reads the input while it writes the output.
function _check_no_alias(output, input)
    Base.mightalias(output, input) &&
        throw(ArgumentError("output must not share memory with the input"))
    return nothing
end

# Output buffers are written by the C library and must be dense arrays of the
# element type it writes.
function _check_output_buffer(output)
    output isa Array{Float64} || output isa Array{ComplexF64} ||
        throw(ArgumentError(
            "output must be an Array{Float64} or Array{ComplexF64}, got $(typeof(output))"))
    return nothing
end

function _check_output_dims(output, al, dim, n)
    expected = ntuple(d -> d == dim ? n : size(al, d), ndims(al))
    size(output) == expected || throw(DimensionMismatch(
        "output has size $(size(output)), expected $expected"))
    return nothing
end

function _evaluate!(output::Array{Tout,N}, sampling::TauSampling, al::Array{Tin,N},
        dim) where {Tout,Tin,N}
    _check_output_dims(output, al, dim, npoints(sampling))
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR
    backend = _spir_default_backend[]
    if Tin == Float64 && Tout == Float64
        ret = C_API.spir_sampling_eval_dd(
            sampling.ptr, backend, order, N, input_dims, target_dim, al, output)
        op = "spir_sampling_eval_dd"
    elseif Tin == ComplexF64 && Tout == ComplexF64
        ret = C_API.spir_sampling_eval_zz(
            sampling.ptr, backend, order, N, input_dims, target_dim, al, output)
        op = "spir_sampling_eval_zz"
    else
        throw(ArgumentError("Type combination not supported for TauSampling evaluate!: \
                             input=$Tin, output=$Tout"))
    end
    # Handle by success: every status other than SPIR_COMPUTATION_SUCCESS is an
    # error, including codes this wrapper does not enumerate.
    _check_status(ret, op)
    return output
end

function _evaluate!(output::Array{Tout,N}, sampling::MatsubaraSampling, al::Array{Tin,N},
        dim) where {Tout,Tin,N}
    _check_output_dims(output, al, dim, npoints(sampling))
    if !isempty(sampling.inverse_order)
        # The C library writes the values in ascending order of the points.
        sorted = similar(output)
        _evaluate_sorted!(sorted, sampling, al, dim)
        output .= sorted[ntuple(d -> d == dim ? sampling.inverse_order : Colon(), N)...]
        return output
    end
    return _evaluate_sorted!(output, sampling, al, dim)
end

function _evaluate_sorted!(output::Array{Tout,N}, sampling::MatsubaraSampling,
        al::Array{Tin,N}, dim) where {Tout,Tin,N}
    Tin <: Complex && _check_real_coefficients(sampling, al)
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR
    backend = _spir_default_backend[]
    if Tin == Float64 && Tout == ComplexF64
        ret = C_API.spir_sampling_eval_dz(
            sampling.ptr, backend, order, N, input_dims, target_dim, al, output)
        op = "spir_sampling_eval_dz"
    elseif Tin == ComplexF64 && Tout == ComplexF64
        ret = C_API.spir_sampling_eval_zz(
            sampling.ptr, backend, order, N, input_dims, target_dim, al, output)
        op = "spir_sampling_eval_zz"
    else
        throw(ArgumentError("Type combination not supported for MatsubaraSampling \
                             evaluate!: input=$Tin, output=$Tout"))
    end
    _check_status(ret, op)
    return output
end

# An output with another number of dimensions than the input.
function _evaluate!(output, sampling, al, dim)
    throw(DimensionMismatch("output has $(ndims(output)) dimensions, expected $(ndims(al))"))
end
function _fit!(output, sampling, al, dim)
    throw(DimensionMismatch("output has $(ndims(output)) dimensions, expected $(ndims(al))"))
end

function _fit!(output::Array{Tout,N}, sampling::TauSampling, al::Array{Tin,N},
        dim) where {Tout,Tin,N}
    _check_output_dims(output, al, dim, length(sampling.basis))
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR
    backend = _spir_default_backend[]
    if Tin == Float64 && Tout == Float64
        ret = C_API.spir_sampling_fit_dd(
            sampling.ptr, backend, order, N, input_dims, target_dim, al, output)
        op = "spir_sampling_fit_dd"
    elseif Tin == ComplexF64 && Tout == ComplexF64
        ret = C_API.spir_sampling_fit_zz(
            sampling.ptr, backend, order, N, input_dims, target_dim, al, output)
        op = "spir_sampling_fit_zz"
    else
        throw(ArgumentError("Type combination not supported for TauSampling fit!: \
                             input=$Tin, output=$Tout"))
    end
    _check_status(ret, op)
    return output
end

function _fit!(output::Array{Tout,N}, sampling::MatsubaraSampling, al::Array{Tin,N},
        dim) where {Tout,Tin,N}
    _check_output_dims(output, al, dim, length(sampling.basis))
    if !isempty(sampling.order)
        # The C library reads the values in ascending order of the points.
        al = al[ntuple(d -> d == dim ? sampling.order : Colon(), N)...]
    end
    input_dims = Int32[size(al)...]
    target_dim = Int32(dim - 1)  # C uses 0-based indexing
    order = C_API.SPIR_ORDER_COLUMN_MAJOR
    backend = _spir_default_backend[]
    if Tin == ComplexF64 && Tout == ComplexF64
        ret = C_API.spir_sampling_fit_zz(
            sampling.ptr, backend, order, N, input_dims, target_dim, al, output)
        _check_status(ret, "spir_sampling_fit_zz")
        return output
    elseif Tin == ComplexF64 && Tout == Float64
        # Real output was explicitly requested. Fit in full complex arithmetic,
        # then report — rather than silently discard — a non-negligible
        # imaginary part, which means the coefficients are genuinely complex.
        temp_output = Array{ComplexF64,N}(undef, size(output)...)
        ret = C_API.spir_sampling_fit_zz(
            sampling.ptr, backend, order, N, input_dims, target_dim, al, temp_output)
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
        throw(ArgumentError("Type combination not supported for MatsubaraSampling fit!: \
                             input=$Tin, output=$Tout"))
    end
end

# Coefficients are compared against the accuracy the basis was built for, not
# against machine epsilon: a basis with target accuracy `eps` only resolves
# quantities down to `eps`.
function _imag_tolerance(sampling::MatsubaraSampling)
    acc = accuracy(sampling.basis)
    return max(isfinite(acc) ? 10 * acc : 1e-8, 1e-12)
end

# `positive_only = true` asserts real IR coefficients; genuinely complex ones
# are rejected where the assumption can be checked (evaluate). An imaginary part
# at the level of the basis accuracy, such as that of a fit result, is accepted.
function _check_real_coefficients(sampling::MatsubaraSampling, al::AbstractArray{<:Complex})
    sampling.positive_only || return nothing
    isempty(al) && return nothing
    max_imag = maximum(abs ∘ imag, al)
    tol = _imag_tolerance(sampling) * max(maximum(abs ∘ real, al), 1.0)
    max_imag > tol && throw(ArgumentError(
        "positive_only=true assumes real IR coefficients (g(-iω) = conj(g(iω))), \
         but the coefficients have max |imag| = $max_imag > $tol; use \
         positive_only=false for complex coefficients"))
    return nothing
end

"""
    cond(sampling::MatsubaraSampling)

Condition number of the sampling problem. With `positive_only = true` this is
the condition number of the real least-squares problem `[Re A; Im A] x = [Re g; Im g]`
that [`fit`](@ref) solves; the C library reports that of the complex matrix `A`
instead, which understates it (SpM-lab/sparse-ir-rs#270).
"""
function LinearAlgebra.cond(sampling::MatsubaraSampling)
    sampling.positive_only || return _cond_from_c(sampling)
    A = transpose(sampling.basis.uhat(sampling.sampling_points))
    return LinearAlgebra.cond(vcat(real(A), imag(A)))
end

# Convenience property accessors: `.tau` and `.ωn` return the sampling points
function Base.getproperty(s::TauSampling, p::Symbol)
    p === :tau ? sampling_points(s) :
    getfield(s, p)
end
function Base.getproperty(s::MatsubaraSampling, p::Symbol)
    p === :ωn ? sampling_points(s) :
    getfield(s, p)
end
