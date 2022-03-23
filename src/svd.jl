import LowRankApprox: psvd
import LinearAlgebra: svd

export MAX_EPS, MAX_T, compute

const MAX_T = Float64
const MAX_EPS = eps(MAX_T)

function compute(a_matrix::AbstractMatrix, n_sv_hint=nothing, strategy=:fast)
    m, n = size(a_matrix)
    isnothing(n_sv_hint) && (n_sv_hint = min(m, n))
    n_sv_hint = min(m, n, n_sv_hint)

    # TODO: extended precision
    if strategy == :fast
        u, s, v = psvd(a_matrix; rank=n_sv_hint)
    elseif strategy == :default
        u, s, v = svd(a_matrix)
    elseif strategy == :accurate
        u, s, v = svd(a_matrix; alg=QRIteration())
    else
        error("unknown strategy: $strategy")
    end

    return u, s, v
end