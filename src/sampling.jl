"""
    AbstractSampling

Abstract class for sparse sampling.

Encodes the "basis transformation" of a propagator from the truncated IR
basis coefficients `G_ir[l]` to time/frequency sampled on sparse points
`G(x[i])` together with its inverse, a least squares fit:

         ________________                   ___________________
        |                |    evaluate     |                   |
        |     Basis      |---------------->|     Value on      |
        |  coefficients  |<----------------|  sampling points  |
        |________________|      fit        |___________________|

"""
abstract type AbstractSampling{T,Tmat,F<:Factorization} end

cond(sampling::AbstractSampling) = cond(sampling.matrix)

function Base.show(io::IO, smpl::S) where {S<:AbstractSampling}
    println(io, S)
    print(io, "Sampling points: ")
    return println(io, smpl.sampling_points)
end

"""
    TauSampling <: AbstractSampling

Sparse sampling in imaginary time.

Allows the transformation between the IR basis and a set of sampling points
in (scaled/unscaled) imaginary time.
"""
struct TauSampling{T,Tmat,F<:Factorization} <: AbstractSampling{T,Tmat,F}
    sampling_points::Vector{T}
    matrix::Matrix{Tmat}
    matrix_fact::F
end

"""
    TauSampling(basis[, sampling_points])

Construct a `TauSampling` object. If not given, the `sampling_points` are chosen as 
the extrema of the highest-order basis function in imaginary time. This turns out 
to be close to optimal with respect to conditioning for this size (within a few percent).
"""
function TauSampling(
    basis::AbstractBasis, sampling_points=default_tau_sampling_points(basis)
)
    matrix = eval_matrix(TauSampling, basis, sampling_points)
    if iswellconditioned(basis) && cond(matrix) > 1e8
        @warn "Sampling matrix is poorly conditioned (cond = $(cond(sampling)))."
    end

    return TauSampling(sampling_points, matrix, factorize(matrix))
end

"""
    MatsubaraSampling <: AbstractSampling

Sparse sampling in Matsubara frequencies.

Allows the transformation between the IR basis and a set of sampling points
in (scaled/unscaled) imaginary frequencies.
"""
struct MatsubaraSampling{T,Tmat,F<:Factorization} <: AbstractSampling{T,Tmat,F}
    sampling_points::Vector{T}
    matrix::Matrix{Tmat}
    matrix_fact::F
end

"""
    MatsubaraSampling(basis[, sampling_points])

Construct a `MatsubaraSampling` object. If not given, the `sampling_points` are chosen as 
the (discrete) extrema of the highest-order basis function in Matsubara. This turns out 
to be close to optimal with respect to conditioning for this size (within a few percent).
"""
function MatsubaraSampling(
    basis::AbstractBasis, sampling_points=default_matsubara_sampling_points(basis)
)
    matrix = eval_matrix(MatsubaraSampling, basis, sampling_points)
    if iswellconditioned(basis) && cond(matrix) > 1e8
        @warn "Sampling matrix is poorly conditioned (cond = $(cond(sampling)))."
    end

    return MatsubaraSampling(sampling_points, matrix, factorize(matrix))
end

"""
    eval_matrix(T, basis, x)

Return evaluation matrix from coefficients to sampling points. `T <: AbstractSampling`.
"""
eval_matrix(::Type{TauSampling}, basis, x) = permutedims(basis.u(x))
eval_matrix(::Type{MatsubaraSampling}, basis, x) = permutedims(basis.uhat(x))

"""
    evaluate(sampling, al; dim=1)

Evaluate the basis coefficients aₗ at the sparse sampling points.
"""
function evaluate(
    smpl::AbstractSampling{Ts,Tmat}, al::AbstractArray{T,N}; dim=1
) where {Ts,Tmat,T,N}
    if size(smpl.matrix, 2) ≠ size(al, dim)
        msg = "Number of columns (got $(size(smpl.matrix, 2))) has to match aₗ's size in dim (got $(size(al, dim)))"
        throw(DimensionMismatch(msg))
    end
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 1), size(al)[(dim + 1):end]...)
    buffer = Array{promote_type(Tmat, T),N}(undef, bufsize)
    return evaluate!(buffer, smpl, al; dim)
end

"""
    evaluate!(buffer::AbstractArray, sampling, al; dim=1)

Like [`evaluate`](@ref), but write the result to `buffer`.
"""
function evaluate!(buffer::AbstractArray, smpl::AbstractSampling, al; dim=1)
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 1), size(al)[(dim + 1):end]...)
    if size(buffer) ≠ bufsize
        msg = "Buffer has the wrong size (got $(size(buffer)), expected $bufsize)"
        throw(DimensionMismatch(msg))
    end
    return matop_along_dim!(buffer, smpl.matrix, al, dim; op=mul!)
end

"""
    fit(sampling, aₗ; dim=1)

Fit basis coefficients from the sparse sampling points
"""
function fit(
    smpl::AbstractSampling{Ts,Tmat}, aₗ::AbstractArray{T,N}; dim=1
) where {Ts,Tmat,T,N}
    if size(smpl.matrix, 1) ≠ size(aₗ, dim)
        msg = "Number of rows (got $(size(smpl.matrix, 1))) has to match aₗ's size in dim (got $(size(aₗ, dim)))"
        throw(DimensionMismatch(msg))
    end
    bufsize = (size(smpl.matrix, 2), size(aₗ)[1:(dim - 1)]..., size(aₗ)[(dim + 1):N]...)
    buffer = Array{promote_type(Tmat, T),N}(undef, bufsize)
    return fit!(buffer, smpl, aₗ; dim)
end

"""
    fit!(buffer::AbstractArray, sampling, aₗ; dim=1)

Like [`fit`](@ref), but write the result to `buffer`.
"""
function fit!(buffer, smpl::AbstractSampling, aₗ; dim=1)
    bufsize = (size(smpl.matrix, 2), size(aₗ)[1:(dim - 1)]..., size(aₗ)[(dim + 1):end]...)
    if size(buffer) ≠ bufsize
        msg = "Buffer has the wrong size (got $(size(buffer)), expected $bufsize)"
        throw(DimensionMismatch(msg))
    end
    return matop_along_dim!(buffer, smpl.matrix_fact, aₗ, dim; op=ldiv!)
end

"""
    movedim(arr::AbstractArray, src => dst)

Move `arr`'s dimension at `src` to `dst` while keeping the order of the remaining 
dimensions unchanged.
"""
function movedim(arr::AbstractArray{T,N}, dims::Pair) where {T,N}
    src, dst = dims
    src == dst && return arr

    perm = collect(1:N)
    deleteat!(perm, src)
    insert!(perm, dst, src)
    return permutedims(arr, perm)
end

"""
    matop_along_dim(mat, arr::AbstractArray, dim::Integer; op=*)

Apply the operator `op` to the matrix `mat` and to the array `arr` along the dimension `dim`.
"""
function matop_along_dim!(buffer, mat, arr::AbstractArray{T,N}, dim=1; op=mul!) where {T,N}
    1 ≤ dim ≤ N || throw(DomainError(dim, "Dimension must be in [1, $N]"))

    # Move the target dim to the first position
    arr = movedim(arr, dim => 1)
    buffer = movedim(buffer, dim => 1)
    matop!(buffer, mat, arr; op)
    return movedim(buffer, 1 => dim)
end

"""
    matop!(buffer, mat, arr::AbstractArray; op=*)

Apply the operator `op` to the matrix `mat` and to the array `arr` along the first dimension.
"""
function matop!(buffer, mat, arr::AbstractArray; op=mul!)
    flatarr = reshape(arr, (size(arr, 1), :))
    flatbuffer = reshape(buffer, size(buffer)[1], :)
    op(flatbuffer, mat, flatarr)
    return buffer
end
matop!(buffer, mat, arr::AbstractVector; op=mul!) = op(buffer, mat, arr)
