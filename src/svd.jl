const T_MAX = Double64
const ε_MAX = eps(T_MAX)

function compute_svd(A::AbstractMatrix{T_MAX}; n_sv_hint=nothing, strategy=:default)
    if !isnothing(n_sv_hint)
        @info "n_sv_hint is set but will not be used in the current implementation!"
    end
    if strategy ≠ :default
        @info "strategy is set but will not be used in the current implementation!"
    end

    return tsvd(A)
end

function compute_svd(A::AbstractMatrix; n_sv_hint=nothing, strategy=:default)
    if !isnothing(n_sv_hint)
        @info "n_sv_hint is set but will not be used in the current implementation!"
    end
    # m, n = size(A)
    # isnothing(n_sv_hint) && (n_sv_hint = min(m, n))
    # n_sv_hint = min(m, n, n_sv_hint)

    if strategy == :default
        u, s, v = svd(A)
    elseif strategy == :accurate
        u, s, v = svd(A; alg=QRIteration())
    else
        throw(DomainError(strategy, "unknown strategy"))
    end

    return u, s, v
end
