using StaticArrays: SMatrix, @SMatrix, @SVector
using LinearAlgebra: diag, Diagonal

"""
    svd2x2(A)

Compute the singular value decomposition of the 2x2 matrix `A`.
"""
svd2x2(A::AbstractMatrix) = svd2x2(SMatrix{2,2}(A))
function svd2x2(A::SMatrix{2,2})
    Su = A * A'
    ϕ = 0.5 * atan(Su[1, 2] + Su[2, 1], Su[1, 1] - Su[2, 2])
    sϕ, cϕ = sincos(ϕ)
    U = @SMatrix [cϕ -sϕ; sϕ cϕ]

    Sw = A' * A
    θ = 0.5 * atan(Sw[1, 2] + Sw[2, 1], Sw[1, 1] - Sw[2, 2])
    sθ, cθ = sincos(θ)
    W = @SMatrix [cθ -sθ; sθ cθ]

    SUsum = Su[1, 1] + Su[2, 2]
    SUdif = √((Su[1, 1] - Su[2, 2])^2 + 4 * Su[1, 2] * Su[2, 1])
    svals = @SVector [√((SUsum + SUdif) / 2), √((SUsum - SUdif) / 2)]

    S = U' * A * W
    C = Diagonal(sign.(diag(S)))
    V = W * C

    return U, svals, V
end