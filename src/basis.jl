using SparseIR

abstract type AbstractBasis end

struct IRBasis <: AbstractBasis
    kernel::Any
    u::Any
    û::Any
    s::Any
    v::Any
    sampling_points_v::Any
    statistics::Any
end

function IRBasis(statistics, Λ, ε=nothing; kernel=nothing, sve_result=nothing)
    Λ >= 0 || error("Kernel cutoff Λ must be non-negative")

    kernel = get_kernel(statistics, Λ, kernel)
    if isnothing(sve_result)
        u, s, v = compute(kernel, ε)
    else
        u, s, v = sve_result
        size(u) == size(s) == size(v) || error("Mismatched shapes in SVE")
    end

    if isnothing(ε) && isnothing(sve_result) && !HAVE_XPREC
        @warn """No extended precision is being used.
        Expect single precision (1.5e-8) only as both cutoff
        and accuracy of the basis functions."""
    end

    # The radius of convergence of the asymptotic expansion is Λ/2,
    # so for significantly larger frequencies we use the asymptotics,
    # since it has lower relative error.
    even_odd = Dict(:F => :odd, :B => :even)[statistics]
    û = hat(u, even_odd)
end