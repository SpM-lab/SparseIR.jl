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
abstract type AbstractSampling{T,Tmat,F<:SVD} end

function cond(sampling::AbstractSampling)
    return first(sampling.matrix_svd.S) / last(sampling.matrix_svd.S)
end

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
struct TauSampling{T,Tmat,F<:SVD} <: AbstractSampling{T,Tmat,F}
    sampling_points::Vector{T}
    matrix::Matrix{Tmat}
    matrix_svd::F
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

    return TauSampling(sampling_points, matrix, svd(matrix))
end

"""
    MatsubaraSampling <: AbstractSampling

Sparse sampling in Matsubara frequencies.

Allows the transformation between the IR basis and a set of sampling points
in (scaled/unscaled) imaginary frequencies.
"""
struct MatsubaraSampling{T,Tmat,F<:SVD} <: AbstractSampling{T,Tmat,F}
    sampling_points::Vector{T}
    matrix::Matrix{Tmat}
    matrix_svd::F
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

    return MatsubaraSampling(sampling_points, matrix, svd(matrix))
end

"""
    eval_matrix(T, basis, x)

Return evaluation matrix from coefficients to sampling points. `T <: AbstractSampling`.
"""
eval_matrix(::Type{TauSampling}, basis, x) = permutedims(basis.u(x))
eval_matrix(::Type{MatsubaraSampling}, basis, x) = permutedims(basis.uhat(x))

"""
    evaluate(sampling, al; dim=1)

Evaluate the basis coefficients `al` at the sparse sampling points.
"""
function evaluate(
    smpl::AbstractSampling{S,Tmat}, al::AbstractArray{T,N}; dim=1
) where {S,Tmat,T,N}
    if size(smpl.matrix, 2) ≠ size(al, dim)
        msg = "Number of columns (got $(size(smpl.matrix, 2))) has to match al's size in dim (got $(size(al, dim)))"
        throw(DimensionMismatch(msg))
    end
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 1), size(al)[(dim + 1):end]...)
    buffer = Array{promote_type(Tmat, T),N}(undef, bufsize)
    return evaluate!(buffer, smpl, al; dim)
end

"""
    evaluate!(buffer, sampling, al; dim=1)

Like [`evaluate`](@ref), but write the result to `buffer`.
"""
function evaluate!(buffer::AbstractArray, smpl::AbstractSampling, al; dim=1)
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 1), size(al)[(dim + 1):end]...)
    if size(buffer) ≠ bufsize
        msg = "Buffer has the wrong size (got $(size(buffer)), expected $bufsize)"
        throw(DimensionMismatch(msg))
    end
    return matop_along_dim!(buffer, smpl.matrix, al, dim, mul!)
end

"""
    fit(sampling, al; dim=1)

Fit basis coefficients from the sparse sampling points
"""
function fit(
    smpl::AbstractSampling{S,Tmat}, al::AbstractArray{T,N}; dim=1
) where {S,Tmat,T,N}
    if size(smpl.matrix, 1) ≠ size(al, dim)
        msg = "Number of rows (got $(size(smpl.matrix, 1))) has to match al's size in dim (got $(size(al, dim)))"
        throw(DimensionMismatch(msg))
    end
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 2), size(al)[(dim + 1):N]...)
    buffer = Array{promote_type(Tmat, T),N}(undef, bufsize)
    return fit!(buffer, smpl, al; dim)
end

"""
    workarrsizefit(smpl::AbstractSampling, al; dim=1)

Return size of workarr for `fit!`.
"""
function workarrsizefit(smpl::AbstractSampling, al; dim=1)
    return length(smpl.matrix_svd.S), (length(al) ÷ size(al, dim))
end

"""
    fit!(buffer, sampling, al; dim=1)

Like [`fit`](@ref), but write the result to `buffer`.
"""
function fit!(
    buffer, smpl::AbstractSampling, al::Array{T,N};
    dim=1,
    workarr::Matrix{T}=Matrix{T}(undef, workarrsizefit(smpl, al, dim=dim)...)
) where {T,N}
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 2), size(al)[(dim + 1):end]...)
    if size(buffer) ≠ bufsize
        msg = "Buffer has the wrong size (got $(size(buffer)), expected $bufsize)"
        throw(DimensionMismatch(msg))
    end
    size(workarr) == workarrsizefit(smpl, al, dim=dim) || throw(ArgumentError("Invalid size of workarr"))
    return matop_along_dim!(buffer, smpl.matrix_svd, al, workarr, dim, ldiv_noalloc!)
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
function matop_along_dim!(
    buffer, mat, arr::AbstractArray{T,N}, dim, op
) where {T,N}
    1 ≤ dim ≤ N || throw(DomainError(dim, "Dimension must be in [1, $N]"))

    if dim == 1
        matop!(buffer, mat, arr, op, 1)
    elseif dim != N
        # Move the target dim to the first position
        arr = movedim(arr, dim => 1)
        buffer = movedim(buffer, dim => 1)
        matop!(buffer, mat, arr, op, 1)
        buffer = movedim(buffer, 1 => dim)
    else
        # Apply the operator to the last dimension
        matop!(buffer, mat, arr, op, N)
    end
    return buffer
end
function matop_along_dim!(
    buffer, mat, arr::AbstractArray{T,N}, workarr, dim, op
) where {T,N}
    1 ≤ dim ≤ N || throw(DomainError(dim, "Dimension must be in [1, $N]"))

    if dim == 1
        matop!(buffer, mat, arr, workarr, op, 1)
    elseif dim != N
        # Move the target dim to the first position
        arr = movedim(arr, dim => 1)
        buffer = movedim(buffer, dim => 1)
        matop!(buffer, mat, arr, workarr, op, 1)
        buffer = movedim(buffer, 1 => dim)
    else
        # Apply the operator to the last dimension
        matop!(buffer, mat, arr, workarr, op, N)
    end
    return buffer
end

"""
    matop!(buffer, mat, arr::AbstractArray; op=*, dim=1)

Apply the operator `op` to the matrix `mat` and to the array `arr` along the first dimension (dim=1) or the last dimension (dim=N)
"""
function matop!(
    buffer::AbstractArray{S,N}, mat, arr::AbstractArray{T,N}, op, dim
) where {S,T,N}
    if dim == 1
        flatarr = reshape(arr, (size(arr, 1), :))
        flatbuffer = reshape(buffer, (size(buffer, 1), :))
        op(flatbuffer, mat, flatarr)
    elseif dim == N
        flatarr = reshape(arr, (:, size(arr, N)))
        flatbuffer = reshape(buffer, (:, size(buffer, N)))
        op(flatbuffer, flatarr, transpose(mat))
    else
        throw(DomainError("Dimension must be 1 or $N"))
    end
    return buffer
end

function matop!(
    buffer::AbstractArray{S,N}, mat, arr::AbstractArray{T,N}, workarr, op, dim
) where {S,T,N}
    if dim == 1
        flatarr = reshape(arr, (size(arr, 1), :))
        flatbuffer = reshape(buffer, (size(buffer, 1), :))
        op(flatbuffer, mat, flatarr, workarr)
    elseif dim == N
        flatarr = reshape(arr, (:, size(arr, N)))
        flatbuffer = reshape(buffer, (:, size(buffer, N)))
        op(flatbuffer, flatarr, transpose(mat), workarr)
    else
        throw(DomainError("Dimension must be 1 or $N"))
    end
    return buffer
end

function ldiv_noalloc!(Y::AbstractMatrix, A::SVD, B::AbstractMatrix, workarr)
    size(workarr) == (size(A.U, 2), size(B, 2)) || throw(DimensionMismatch())
    mul!(workarr, A.U', B)
    workarr ./= A.S
    return mul!(Y, A.V, workarr)
end
