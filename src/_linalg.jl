module _LinAlg

import LinearAlgebra
export tsvd, tsvd!, svd_jacobi, svd_jacobi!, rrqr, rrqr!

"""
Compute Givens rotation `R` matrix that satisfies:

    [  c  s ] [ f ]     [ r ]
    [ -s  c ] [ g ]  =  [ 0 ]
"""
function givens_params(f::T, g::T) where T <: AbstractFloat
    # ACM Trans. Math. Softw. 28(2), 206, Alg 1
    if iszero(g)
        c, s = one(T), zero(T)
        r = f
    elseif iszero(f)
        c, s = zero(T), T(signbit(g))
        r = abs(g)
    else
        r = copysign(hypot(f, g), f)
        c = f / r
        s = g / r
    end
    return (c, s), r
end

"""
Apply Givens rotation to vector:

      [ a ]  =  [  c   s ] [ x ]
      [ b ]     [ -s   c ] [ y ]
"""
function givens_lmul((c, s)::Tuple{T, T}, (x, y)) where T
    a = c * x + s * y
    b = c * y - s * x
    return a, b
end

"""
Perform the SVD of upper triangular two-by-two matrix:

      [ f    g   ]  =  [  cu  -su ] [ smax     0 ] [  cv   sv ]
      [ 0    h   ]     [  su   cu ] [    0  smin ] [ -sv   cv ]

Note that smax and smin can be negative.
"""
function svd2x2(f::T, g::T, h::T) where T <: AbstractFloat
    # Code taken from LAPACK xSLAV2:
    fa = abs(f)
    ga = abs(g)
    ha = abs(h)
    if fa < ha
        # switch h <-> f, cu <-> sv, cv <-> su
        (sv, cv), (smax, smin), (su, cu) = svd2x2(h, g, f)
    elseif iszero(ga)
        # already diagonal, fa > ha
        smax, smin = fa, ha
        cv, sv = cu, su = one(T), zero(T)
    elseif fa < eps(T) * ga
        # ga is very large
        smax = ga
        if ha > one(T)
            smin = fa / (ga / ha)
        else
            smin = (fa / ga) * ha
        end
        cv, sv = one(T), f / g
        cu, su = one(T), h / g
    else
        # normal case
        fmh = fa - ha
        d = fmh / fa
        q = g / f
        s = T(2) - d
        spq = hypot(q, s)
        dpq = hypot(d, q)
        a = (spq + dpq) / T(2)
        smax, smin = abs(fa * a), abs(ha / a)

        tmp = (q / (spq + s) + q / (dpq + d)) * (one(T) + a)
        tt = hypot(tmp, T(2))
        cv = T(2) / tt
        sv = tmp / tt
        cu = (cv + sv * q) / a
        su = ((h / f) * sv) / a
    end
    return (cu, su), (smax, smin), (cv, sv)
end

"""
Perform the SVD of an arbitrary two-by-two matrix:

      [ a11  a12 ]  =  [  cu  -su ] [ smax     0 ] [  cv   sv ]
      [ a21  a22 ]     [  su   cu ] [    0  smin ] [ -sv   cv ]

Note that smax and smin can be negative.
"""
function svd2x2(a11::T, a12::T, a21::T, a22::T) where T
    abs_a12 = abs(a12)
    abs_a21 = abs(a21)
    if iszero(a21)
        # upper triangular case
        (cu, su), (smax, smin), (cv, sv) = svd2x2(a11, a12, a22)
    elseif abs_a12 > abs_a21
        # closer to lower triangular - transpose matrix
        (cv, sv), (smax, smin), (cu, su) = svd2x2(a11, a21, a12, a22)
    else
        # First, we use a givens rotation  Rx
        # [  cx   sx ] [ a11  a12 ] = [ rx  a12' ]
        # [ -sx   cx ] [ a21  a22 ]   [ 0   a22' ]
        (cx, sx), rx = givens_params(a11, a21)
        a11, a21 = rx, zero(rx)
        a12, a22 = givens_lmul((cx, sx), (a12, a22))

        # Next, use the triangular routine
        # [ f  g ]  =  [  cu  -su ] [ smax     0 ] [  cv   sv ]
        # [ 0  h ]     [  su   cu ] [    0  smin ] [ -sv   cv ]
        (cu, su), (smax, smin), (cv, sv) = svd2x2(a11, a12, a22)

        # Finally, update the LHS (U) transform as follows:
        # [  cx  -sx ] [  cu  -su ] = [  cu'  -su' ]
        # [  sx   cx ] [  su   cu ]   [  su'   cu' ]
        cu, su = givens_lmul((cx, -sx), (cu, su))
    end
    return (cu, su), (smax, smin), (cv, sv)
end

function jacobi_sweep!(U::AbstractMatrix, VT::AbstractMatrix)
    ii, jj = size(U)
    if ii < jj
        throw(ArgumentError("matrix must be 'tall'"))
    elseif size(VT, 1) != jj
        throw(ArgumentError("U and VT must be compatible"))
    end
    Base.require_one_based_indexing(U)
    Base.require_one_based_indexing(VT)

    # TODO: non-traditional indexing
    offd = zero(eltype(U))
    @inbounds for i in 1:ii
        for j in i+1:jj
            # Construct the 2x2 matrix to be diagonalized
            Hii = Hij = Hjj = zero(eltype(U))
            for k in 1:ii
                Hii += abs2(U[k, i])
                Hij += U[k, i] * U[k, j]
                Hjj += abs2(U[k, j])
            end
            offd += abs2(Hij)

            # diagonalize
            (_, _), (_, _), (cv, sv) = svd2x2(Hii, Hij, Hij, Hjj)

            # apply rotation to VT
            rot = LinearAlgebra.Givens(i, j, cv, sv)
            LinearAlgebra.lmul!(rot, VT)
            LinearAlgebra.rmul!(U, adjoint(rot))
        end
    end
    return sqrt(offd)
end

"""Singular value decomposition using Jacobi rotations."""
function svd_jacobi!(U::AbstractMatrix{T}; rtol=eps(T), maxiter=20) where T
    m, n = size(U)
    if m < n
        throw(ArgumentError("matrix must be 'tall'"))
    end
    Base.require_one_based_indexing(U)   # TODO

    VT = Matrix(one(T) * LinearAlgebra.I, n, n)
    Unorm = LinearAlgebra.norm(@view U[1:n, 1:n])
    for _ in 1:maxiter
        offd = jacobi_sweep!(U, VT)
        if offd < rtol * Unorm
            break
        end
    end

    s = Vector{T}(undef, n)
    @inbounds for i in 1:n
        s[i] = LinearAlgebra.norm(@view U[:, i])
        # For some reason, U[:,i] ./= s[i] creates a copy
        for j in axes(U, 1)
            U[j, i] /= s[i]
        end
    end
    return LinearAlgebra.SVD(U, s, VT)
end

"""Singular value decomposition using Jacobi rotations."""
svd_jacobi(U::AbstractMatrix{T}; rtol=eps(T), maxiter=20) where T =
    svd_jacobi!(copy(U); rtol=rtol, maxiter=maxiter)

"""
Truncated rank-revealing QR decomposition with full column pivoting.

Decomposes a `(m, n)` matrix `A` into the product:

    A[:,piv] == Q @ R

where `Q` is an `(m, k)` isometric matrix, `R` is a `(k, n)` upper
triangular matrix, `piv` is a permutation vector, and `k` is chosen such
that the relative tolerance `tol` is met in the equality above.
"""
function rrqr!(A::AbstractMatrix{T}; rtol=eps(T)) where T <: AbstractFloat
    # DGEQPF
    m, n = size(A)
    k = min(m, n)
    Base.require_one_based_indexing(A)   # TODO

    jpvt = Vector(1:n)
    taus = Vector{T}(undef, k)
    swapcol = Vector{T}(undef, m)

    xnorms = map(LinearAlgebra.norm, eachcol(A))
    pnorms = copy(xnorms)
    sqrteps = sqrt(eps(T))

    @inbounds for i in 1:k
        pvt = argmax(@view pnorms[i:end]) + i - 1
        if i != pvt
            jpvt[i], jpvt[pvt] = jpvt[pvt], jpvt[i]
            xnorms[pvt] = xnorms[i]
            pnorms[pvt] = pnorms[i]

            swapcol .= @view A[:, i]
            A[:, i] .= @view A[:, pvt]
            A[:, pvt] .= swapcol
        end

        tau_i = LinearAlgebra.reflector!(@view A[i:end, i])
        taus[i] = tau_i
        LinearAlgebra.reflectorApply!(
            (@view A[i:end, i]), tau_i, @view A[i:end, i+1:end])

        # Lapack Working Note 176.
        for j in i+1:n
            temp = abs(A[i,j]) / pnorms[j]
            temp = max(zero(T), (one(T) + temp) * (one(T) - temp))
            temp2 = temp * abs2(pnorms[j] / xnorms[j])
            if temp2 < sqrteps
                recomputed = LinearAlgebra.norm(@view A[i+1:end, j])
                pnorms[j] = recomputed
                xnorms[j] = recomputed
            else
                pnorms[j] *= sqrt(temp)
            end
        end

        # Since we did pivoting, R[i,:] is bounded by R[i,i], so we can
        # simply ignore all the other rows
        if abs(A[i, i]) < rtol * abs(A[1, 1])
            A[i:end, i:end] .= zero(T)
            taus[i:end] .= zero(T)
            k = i - 1
            break
        end
    end
    return LinearAlgebra.QRPivoted{T, typeof(A)}(A, taus, jpvt), k
end

"""Truncated rank-revealing QR decomposition with full column pivoting."""
rrqr(A::AbstractMatrix{T}; rtol=eps(T)) where T <: AbstractFloat =
    rrqr!(copy(A); rtol=rtol)

"""Truncate RRQR result low-rank"""
function truncate_qr_result(qr::LinearAlgebra.QRPivoted{T}, k::Integer) where T
    m, n = size(qr)
    if k < 0 || k > min(m, n)
        throw(ArgumentError("Invalid rank"))
    end
    Qfull = LinearAlgebra.QRPackedQ((@view qr.factors[:, 1:k]), qr.τ[1:k])

    Q = LinearAlgebra.lmul!(Qfull, Matrix{T}(LinearAlgebra.I, m, k))
    R = LinearAlgebra.triu!(qr.factors[1:k, :])
    return Q, R
end

"""
Truncated singular value decomposition.

Decomposes a `(m, n)` matrix `A` into the product:

    A == U @ (s[:,None] * VT)

where `U` is a `(m, k)` matrix with orthogonal columns, `VT` is a `(k, n)`
matrix with orthogonal rows and `s` are the singular values, a set of `k`
nonnegative numbers in non-ascending order.  The SVD is truncated in the
sense that singular values below `tol` are discarded.
"""
function tsvd!(A::AbstractMatrix{T}; rtol=eps(T)) where T <: AbstractFloat
    # Perform RRQR of the m x n matrix at a cost of O(m*n*k), where k is
    # the QR rank (a mild upper bound for the true rank)
    A_qr, k = rrqr!(A, rtol=rtol)
    Q, R = truncate_qr_result(A_qr, k)

    # RRQR is an excellent preconditioner for Jacobi.  One should then perform
    # Jacobi on RT
    RT_svd = svd_jacobi(R')

    # Reconstruct A from QR
    U = Q * RT_svd.V
    V = A_qr.P * RT_svd.U
    s = RT_svd.S
    return LinearAlgebra.SVD(U, s, V')
end

"""Truncated singular value decomposition."""
tsvd(A::AbstractMatrix{T}; rtol=eps(T)) where T <: AbstractFloat =
    tsvd!(copy(A); rtol=rtol)

end # module _LinAlg
