"""
    TauSampling <: AbstractSampling

Sparse sampling in imaginary time.

Allows the transformation between the IR basis and a set of sampling points
in (scaled/unscaled) imaginary time.
"""
struct TauSampling{T,TMAT,F<:SVD} <: AbstractSampling{T,TMAT,F}
    sampling_points :: Vector{T}
    matrix          :: Matrix{TMAT}
    matrix_svd      :: F
end

"""
    TauSampling(basis[, sampling_points])

Construct a `TauSampling` object. If not given, the `sampling_points` are chosen
as the extrema of the highest-order basis function in imaginary time. This turns
out to be close to optimal with respect to conditioning for this size (within a
few percent).
"""
function TauSampling(basis::AbstractBasis,
                     sampling_points=default_tau_sampling_points(basis))
    matrix   = eval_matrix(TauSampling, basis, sampling_points)
    sampling = TauSampling(sampling_points, matrix, svd(matrix))
    if iswellconditioned(basis) && cond(sampling) > 1e8
        @warn "Sampling matrix is poorly conditioned (cond = $(cond(sampling)))."
    end
    return sampling
end

Base.getproperty(s::TauSampling, p::Symbol) = p === :τ ? sampling_points(s) : getfield(s, p)

"""
    MatsubaraSampling <: AbstractSampling

Sparse sampling in Matsubara frequencies.

Allows the transformation between the IR basis and a set of sampling points
in (scaled/unscaled) imaginary frequencies.
"""
struct MatsubaraSampling{T<:MatsubaraFreq,TMAT,F} <: AbstractSampling{T,TMAT,F}
    sampling_points :: Vector{T}
    matrix          :: Matrix{TMAT}
    matrix_svd      :: F
    positive_only   :: Bool
end

"""
    MatsubaraSampling(basis; positive_only=false,
                      sampling_points=default_matsubara_sampling_points(basis; positive_only))

Construct a `MatsubaraSampling` object. If not given, the `sampling_points` are chosen as
the (discrete) extrema of the highest-order basis function in Matsubara. This turns out
to be close to optimal with respect to conditioning for this size (within a few percent).

By setting `positive_only=true`, one assumes that functions to be fitted are symmetric in
Matsubara frequency, i.e.:
```math
    Ĝ(iν) = conj(Ĝ(-iν))
```
or equivalently, that they are purely real in imaginary time. In this case, sparse sampling
is performed over non-negative frequencies only, cutting away half of the necessary sampling
space.
"""
function MatsubaraSampling(basis::AbstractBasis; positive_only=false,
                           sampling_points=default_matsubara_sampling_points(basis; positive_only))
    issorted(sampling_points) || sort!(sampling_points)
    if positive_only
        Int(first(sampling_points)) ≥ 0 || error("invalid negative sampling frequencies")
    end
    matrix = eval_matrix(MatsubaraSampling, basis, sampling_points)
    has_zero = iszero(first(sampling_points))
    svd_matrix = positive_only ? SplitSVD(matrix; has_zero) : svd(matrix)
    sampling = MatsubaraSampling(sampling_points, matrix, svd_matrix, positive_only)
    if iswellconditioned(basis) && cond(sampling) > 1e8
        @warn "Sampling matrix is poorly conditioned (cond = $(cond(sampling)))."
    end
    return sampling
end

Base.getproperty(s::MatsubaraSampling, p::Symbol) = p === :ωn ? sampling_points(s) : getfield(s, p)

"""
    eval_matrix(T, basis, x)

Return evaluation matrix from coefficients to sampling points. `T <: AbstractSampling`.
"""
function eval_matrix end
eval_matrix(::Type{TauSampling},       basis, x) = permutedims(basis.u(x))
eval_matrix(::Type{MatsubaraSampling}, basis, x) = permutedims(basis.uhat(x))

"""
    evaluate(sampling, al; dim=1)

Evaluate the basis coefficients `al` at the sparse sampling points.
"""
function evaluate(smpl::AbstractSampling{S,Tmat}, al::AbstractArray{T,N};
                  dim=1) where {S,Tmat,T,N}
    if size(smpl.matrix, 2) ≠ size(al, dim)
        msg = "Number of columns (got $(size(smpl.matrix, 2))) has to match " *
              "al's size in dim (got $(size(al, dim)))."
        throw(DimensionMismatch(msg))
    end
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 1), size(al)[(dim + 1):end]...)
    buffer = Array{promote_type(Tmat, T),N}(undef, bufsize)
    return evaluate!(buffer, smpl, al; dim)
end

"""
    evaluate!(buffer::AbstractArray{T,N}, sampling, al; dim=1) where {T,N}

Like [`evaluate`](@ref), but write the result to `buffer`.
Please use dim = 1 or N to avoid allocating large temporary arrays internally.
"""
function evaluate!(buffer::AbstractArray{T, N}, smpl::AbstractSampling,
                   al::AbstractArray{S, N}; dim=1) where {S,T,N}
    resultsize = ntuple(j -> j == dim ? size(smpl.matrix, 1) : size(al, j), N)
    if size(buffer) ≠ resultsize
        msg = "Buffer has the wrong size (got $(size(buffer)), expected $resultsize)."
        throw(DimensionMismatch(msg))
    end
    return matop_along_dim!(buffer, smpl.matrix, al, dim, mul!)
end

"""
    fit(sampling, al::AbstractArray{T,N}; dim=1)

Fit basis coefficients from the sparse sampling points
Please use dim = 1 or N to avoid allocating large temporary arrays internally.
"""
function fit(smpl::AbstractSampling{S,Tmat}, al::AbstractArray{T,N};
             dim=1) where {S,Tmat,T,N}
    if size(smpl.matrix, 1) ≠ size(al, dim)
        msg = "Number of rows (got $(size(smpl.matrix, 1))) "
              "has to match al's size in dim (got $(size(al, dim)))."
        throw(DimensionMismatch(msg))
    end
    bufsize = (size(al)[1:(dim - 1)]..., size(smpl.matrix, 2), size(al)[(dim + 1):N]...)
    buffer = Array{promote_type(Tmat, T),N}(undef, bufsize)
    return fit!(buffer, smpl, al; dim)
end

"""
    workarrlength(smpl::AbstractSampling, al; dim=1)

Return length of workarr for `fit!`.
"""
workarrlength(smpl::AbstractSampling, al::AbstractArray; dim=1) =
    length(smpl.matrix_svd.S) * (length(al) ÷ size(al, dim))

"""
    fit!(buffer::Array{S,N}, smpl::AbstractSampling, al::Array{T,N}; 
        dim=1, workarr::Vector{S}) where {S,T,N}

Like [`fit`](@ref), but write the result to `buffer`.
Use `dim = 1` or `dim = N` to avoid allocating large temporary arrays internally.
The length of `workarr` cannot be smaller than [`SparseIR.workarrlength`](@ref)`(smpl, al)`.
"""
function fit!(buffer::Array{S,N}, smpl::AbstractSampling, al::Array{T,N}; dim=1,
              workarr::Vector{S}=Vector{S}(undef, workarrlength(smpl, al; dim))) where {S,T,N}
    resultsize = ntuple(j -> j == dim ? size(smpl.matrix, 2) : size(al, j), N)
    if size(buffer) ≠ resultsize
        msg = "Buffer has the wrong size (got $(size(buffer)), expected $resultsize)."
        throw(DimensionMismatch(msg))
    end
    length(workarr) ≥ workarrlength(smpl, al; dim) ||
        throw(ArgumentError("workarr too small"))
    return div_noalloc!(buffer, smpl.matrix_svd, al, workarr, dim)
end

"""
    movedim(arr::AbstractArray, src => dst)

Move `arr`'s dimension at `src` to `dst` while keeping the order of the remaining
dimensions unchanged.
"""
function movedim(arr::AbstractArray{T,N}, dims::Pair) where {T,N}
    src, dst = dims
    src == dst && return arr
    return permutedims(arr, getperm(N, dims))
end

function getperm(N, dims::Pair)
    src, dst = dims
    perm = collect(1:N)
    deleteat!(perm, src)
    insert!(perm, dst, src)
    return perm
end

"""
    matop_along_dim!(buffer, mat, arr::AbstractArray, dim::Integer, op)

Apply the operator `op` to the matrix `mat` and to the array `arr` along the dimension
`dim`, writing the result to `buffer`.
"""
function matop_along_dim!(buffer, mat, arr::AbstractArray{T,N}, dim, op) where {T,N}
    1 ≤ dim ≤ N || throw(DomainError(dim, "Dimension must be in [1, $N]"))

    if dim == 1
        matop!(buffer, mat, arr, op, 1)
    elseif dim != N
        # Move the target dim to the first position
        perm        = getperm(N, dim => 1)
        arr_perm    = permutedims(arr, perm)
        buffer_perm = permutedims(buffer, perm)
        matop!(buffer_perm, mat, arr_perm, op, 1)
        permutedims!(buffer, buffer_perm, getperm(N, 1 => dim))
    else
        # Apply the operator to the last dimension
        matop!(buffer, mat, arr, op, N)
    end
    return buffer
end

"""
    matop!(buffer, mat, arr::AbstractArray, op, dim)

Apply the operator `op` to the matrix `mat` and to the array `arr` along the first
dimension (dim=1) or the last dimension (dim=N).
"""
function matop!(buffer::AbstractArray{S,N}, mat, arr::AbstractArray{T,N}, op,
                dim) where {S,T,N}
    if dim == 1
        flatarr    = reshape(arr, (size(arr, 1), :))
        flatbuffer = reshape(buffer, (size(buffer, 1), :))
        op(flatbuffer, mat, flatarr)
    elseif dim == N
        flatarr    = reshape(arr, (:, size(arr, N)))
        flatbuffer = reshape(buffer, (:, size(buffer, N)))
        op(flatbuffer, flatarr, transpose(mat))
    else
        throw(DomainError(dim, "Dimension must be 1 or $N."))
    end
    return buffer
end

function div_noalloc!(buffer::AbstractArray{S,N}, mat, arr::AbstractArray{T,N},
                      workarr, dim) where {S,T,N}
    1 ≤ dim ≤ N || throw(DomainError(dim, "Dimension must be in [1, $N]"))

    if dim == 1
        flatarr    = reshape(arr, (size(arr, 1), :))
        flatbuffer = reshape(buffer, (size(buffer, 1), :))
        ldiv_noalloc!(flatbuffer, mat, flatarr, workarr)
    elseif dim != N
        # Move the target dim to the first position
        arr_perm    = movedim(arr, dim => 1)
        buffer_perm = movedim(buffer, dim => 1)
        flatarr     = reshape(arr_perm, (size(arr_perm, 1), :))
        flatbuffer  = reshape(buffer_perm, (size(buffer_perm, 1), :))
        ldiv_noalloc!(flatbuffer, mat, flatarr, workarr)
        buffer .= movedim(buffer_perm, 1 => dim)
    else
        flatarr    = reshape(arr, (:, size(arr, N)))
        flatbuffer = reshape(buffer, (:, size(buffer, N)))
        rdiv_noalloc!(flatbuffer, flatarr, mat, workarr)
    end
    return buffer
end

function ldiv_noalloc!(Y::AbstractMatrix, A::SVD, B::AbstractMatrix, workarr)
    # Setup work space
    worksize = (size(A.U, 2), size(B, 2))
    worklength = prod(worksize)
    length(workarr) ≥ worklength ||
        throw(DimensionMismatch("size(workarr)=$(size(workarr)), min worksize=$worklength"))
    workarr_view = reshape(view(workarr, 1:worklength), worksize)

    mul!(workarr_view, A.U', B)
    workarr_view ./= A.S
    return mul!(Y, A.V, workarr_view)
end

function rdiv_noalloc!(Y::AbstractMatrix, A::AbstractMatrix, B::SVD, workarr)
    # Setup work space
    worksize = (size(A, 1), size(B.U, 2))
    worklength = prod(worksize)
    length(workarr) ≥ worklength ||
        throw(DimensionMismatch("size(workarr)=$(size(workarr)), min worksize=$worklength"))
    workarr_view = reshape(view(workarr, 1:worklength), worksize)

    # Note: conj creates a temporary matrix
    mul!(workarr_view, A, conj(B.U))
    workarr_view ./= reshape(B.S, 1, :)
    return mul!(Y, workarr_view, conj(B.Vt))
end

struct SplitSVD{T}
    A :: Matrix{Complex{T}}
    UrealT :: Matrix{T}
    UimagT :: Matrix{T}
    S :: Vector{T}
    V :: Matrix{T}
end

function SplitSVD(a::Matrix{<:Complex}, (u, s, v)::Tuple{AbstractMatrix{<:Complex}, AbstractVector{<:Real}, AbstractMatrix{<:Real}})
    if any(iszero, s)
        nonzero = findall(!iszero, s)
        u, s, v = u[:, nonzero], s[nonzero], v[nonzero, :]
    end
    ut = transpose(u)
    SplitSVD(a, real(ut), imag(ut), s, copy(v))
end

SplitSVD(a::Matrix{<:Complex}; has_zero=false) = SplitSVD(a, split_complex(a; has_zero))

function ldiv_noalloc!(Y::AbstractMatrix, A::SplitSVD, B::AbstractMatrix, workarr)
    # Setup work space
    worksize = (size(A.UrealT, 1), size(B, 2))
    worklength = prod(worksize)
    length(workarr) ≥ worklength ||
        throw(DimensionMismatch("size(workarr)=$(size(workarr)), min worksize=$worklength"))
    workarr_view = reshape(view(workarr, 1:worklength), worksize)

    mul!(workarr_view, A.UrealT, real(B))
    mul!(workarr_view, A.UimagT, imag(B), true, true)
    workarr_view ./= A.S
    return mul!(Y, A.V, workarr_view)
end

function split_complex(mat::Matrix{<:Complex}; has_zero=false, svd_algo=svd)
    # split real and imaginary part into separate matrices
	offset_imag = has_zero ? 2 : 1
	rmat = [real(mat)
            imag(mat)[offset_imag:end, :]]
    
    # perform real-valued SVD
    ur, s, v = svd_algo(rmat)
    
	# undo the split of the resulting ur matrix
	n = size(mat, 1)
	u = complex(ur[1:n, :])
    u[offset_imag:end, :] .+= im .* ur[n+1:end, :]
    return u, s, v
end

const MatsubaraSampling64F = @static if VERSION ≥ v"1.8-"
    MatsubaraSampling{FermionicFreq,ComplexF64,SVD{ComplexF64,Float64,Matrix{ComplexF64},Vector{Float64}}}
else
    MatsubaraSampling{FermionicFreq,ComplexF64,SVD{ComplexF64,Float64,Matrix{ComplexF64}}}
end

const MatsubaraSampling64B = @static if VERSION ≥ v"1.8-"
    MatsubaraSampling{BosonicFreq,ComplexF64,SVD{ComplexF64,Float64,Matrix{ComplexF64},Vector{Float64}}}
else
    MatsubaraSampling{BosonicFreq,ComplexF64,SVD{ComplexF64,Float64,Matrix{ComplexF64}}}
end

const TauSampling64 = @static if VERSION ≥ v"1.8-"
    TauSampling{Float64,Float64,SVD{Float64,Float64,Matrix{Float64},Vector{Float64}}}
else
    TauSampling{Float64,Float64,SVD{Float64,Float64,Matrix{Float64}}}
end
