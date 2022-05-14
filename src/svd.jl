const _T_MAX = Double64
const _ε_MAX = eps(_T_MAX)

function compute_svd(a_matrix::AbstractMatrix; n_sv_hint=nothing, strategy=:fast)
    if !isnothing(n_sv_hint)
        @warn "n_sv_hint is set but will not be used in the current implementation!"
    end
    m, n = size(a_matrix)
    isnothing(n_sv_hint) && (n_sv_hint = min(m, n))
    n_sv_hint = min(m, n, n_sv_hint)

    if eltype(a_matrix) == _T_MAX
        u, s, v = tsvd(a_matrix)
    elseif strategy == :default
        u, s, v = svd(a_matrix)
    elseif strategy == :accurate
        u, s, v = svd(a_matrix; alg=QRIteration())
    else
        throw(DomainError(strategy, "unknown strategy"))
    end

    return u, s, v
end
