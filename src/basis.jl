using SparseIR

export IRBasis, FiniteTempBasis

abstract type AbstractBasis end

struct IRBasis{K<:AbstractKernel, T<:Real} <: AbstractBasis
    kernel::K
    u::PiecewiseLegendrePolyArray{T}
    û::PiecewiseLegendreFTArray{T}
    s::Vector{T}
    v::PiecewiseLegendrePolyArray{T}
    sampling_points_v::Vector{T}
    statistics::Symbol
end

function IRBasis(statistics, Λ, ε=nothing; kernel=nothing, sve_result=nothing)
    Λ >= 0 || error("Kernel cutoff Λ must be non-negative")

    self_kernel = get_kernel(statistics, Λ, kernel) # TODO better name
    if isnothing(sve_result)
        u, s, v = compute(self_kernel; ε)
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
    û = hat.(u, even_odd; n_asymp=conv_radius(self_kernel))
    rts = roots(last(v))
    sampling_points_v = [v.xmin; (rts[begin:(end - 1)] .+ rts[(begin + 1):end]) / 2; v.xmax]
    return IRBasis(self_kernel, u, û, s, v, sampling_points_v, statistics)
end

Λ(basis::IRBasis) = basis.kernel.Λ

function Base.getindex(basis::IRBasis, i)
    sve_result = basis.u[i], basis.s[i], basis.v[i]
    return IRBasis(basis.statistics, Λ(basis); kernel=basis.kernel, sve_result)
end

function get_kernel(statistics, Λ, kernel)
    statistics ∈ (:F, :B) ||
        error("""statistics must be either :B (for fermionic basis) or :F (for bosonic basis)""")
    if isnothing(kernel)
        kernel = LogisticKernel(Λ)
    else
        @assert kernel.Λ ≈ Λ
    end
    return kernel
end

struct FiniteTempBasis{K<:AbstractKernel, T<:Real} <: AbstractBasis
    kernel::K
    sve_result::Tuple{PiecewiseLegendrePolyArray{T},Vector{T},PiecewiseLegendrePolyArray{T}}
    statistics::Symbol
    β::T
    u::PiecewiseLegendrePolyArray{T}
    v::PiecewiseLegendrePolyArray{T}
    s::Vector{T}
    û::PiecewiseLegendreFTArray{T}
end

function FiniteTempBasis(statistics, β, wmax, ε=nothing; kernel=nothing, sve_result=nothing)
    β > 0 || error("Inverse temperature β must be positive")
    wmax >= 0 || error("Frequency cutoff wmax must be non-negative")

    kernel = get_kernel(statistics, β * wmax, kernel)
    if isnothing(sve_result)
        u, s, v = compute(kernel; ε)
    else
        u, s, v = sve_result
        size(u) == size(s) == size(v) || error("Mismatched shapes in SVE")
    end

    if isnothing(ε) && isnothing(sve_result) && !HAVE_XPREC
        @warn """No extended precision is being used.
        Expect single precision (1.5e-8) only as both cutoff
        and accuracy of the basis functions."""
    end

    # The polynomials are scaled to the new variables by transforming the
    # knots according to: tau = beta/2 * (x + 1), w = wmax * y.  Scaling
    # the data is not necessary as the normalization is inferred.
    wmax = kernel.Λ / β
    u_ = PiecewiseLegendrePolyArray(u, β / 2 * (u.knots .+ 1); dx=β / 2 * u.dx, symm=u.symm)
    v_ = PiecewiseLegendrePolyArray(v, wmax * v.knots; dx=wmax * v.dx, symm=v.symm)

    # The singular values are scaled to match the change of variables, with
    # the additional complexity that the kernel may have an additional
    # power of w.
    s_ = √(β / 2 * wmax) * wmax^(-ypower(kernel)) * s

    # HACK: as we don't yet support Fourier transforms on anything but the
    # unit interval, we need to scale the underlying data.  This breaks
    # the correspondence between U.hat and Uhat though.
    û_base = scale.(u, √β)

    conv_radius = 40 * kernel.Λ
    even_odd = Dict(:F => :odd, :B => :even)[statistics]
    û = hat.(û_base, even_odd; n_asymp=conv_radius)

    return FiniteTempBasis(kernel, (u, s, v), statistics, β, u_, v_, s_, û)
end

Base.firstindex(::AbstractBasis) = 1
Base.length(basis::AbstractBasis) = length(basis.s)

function Base.getindex(basis::FiniteTempBasis, i)
    u, s, v = basis.sve_result
    sve_result = u[i], s[i], v[i]
    return FiniteTempBasis(basis.statistics, basis.β, wmax(basis); kernel=basis.kernel,
                           sve_result)
end

wmax(basis::FiniteTempBasis) = basis.kernel.Λ / basis.β