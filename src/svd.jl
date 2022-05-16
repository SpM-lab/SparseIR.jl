const _T_MAX = Double64
const _ε_MAX = eps(_T_MAX)

function compute_svd(a_matrix::AbstractMatrix{_T_MAX}; n_sv_hint=nothing, strategy=:default)
    if !isnothing(n_sv_hint)
        @info "n_sv_hint is set but will not be used in the current implementation!"
    end
    if strategy ≠ :default
        @info "strategy is set but will not be used in the current implementation!"
    end

    return tsvd(a_matrix)
end


function compute_svd(a_matrix::AbstractMatrix; n_sv_hint=nothing, strategy=:default)
    if !isnothing(n_sv_hint)
        @info "n_sv_hint is set but will not be used in the current implementation!"
    end
    m, n = size(a_matrix)
    # isnothing(n_sv_hint) && (n_sv_hint = min(m, n))
    # n_sv_hint = min(m, n, n_sv_hint)

    if strategy == :default
        u, s, v = svd(a_matrix)
    elseif strategy == :accurate
        u, s, v = svd(a_matrix; alg=QRIteration())
    else
        throw(DomainError(strategy, "unknown strategy"))
    end

    return u, s, v
end
