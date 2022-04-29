using LowRankApprox: psvd
using LinearAlgebra: svd, QRIteration

const MAX_T = Float64
const MAX_EPS = eps(MAX_T) # approximately 5e-32

function compute(a_matrix::AbstractMatrix{MAX_T}; n_sv_hint=nothing, strategy=:fast)
    m, n = size(a_matrix)
    isnothing(n_sv_hint) && (n_sv_hint = min(m, n))
    n_sv_hint = min(m, n, n_sv_hint)

    if strategy == :fast
        u, s, v = psvd(a_matrix; rank=n_sv_hint, rtol=0.0)
    elseif strategy == :default
        u, s, v = svd(a_matrix)
    elseif strategy == :accurate
        u, s, v = svd(a_matrix; alg=QRIteration())
    else
        throw(DomainError(strategy, "unknown strategy"))
    end

    return u, s, v
end
