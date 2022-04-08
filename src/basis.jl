using SparseIR

export IRBasis, FiniteTempBasis, finite_temp_bases, fermion, boson

@enum Statistics fermion boson

abstract type AbstractBasis end

struct IRBasis{K<:AbstractKernel,T<:Real} <: AbstractBasis
    kernel::K
    u::PiecewiseLegendrePolyArray{T}
    uhat::PiecewiseLegendreFTArray{T}
    s::Vector{T}
    v::PiecewiseLegendrePolyArray{T}
    sampling_points_v::Vector{T}
    statistics::Statistics
end

function IRBasis(statistics, Λ, ε=nothing; kernel=nothing, sve_result=nothing)
    Λ ≥ 0 || error("Kernel cutoff Λ must be non-negative")

    kernel = get_kernel(statistics, Λ, kernel) # TODO better name
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

    # The radius of convergence of the asymptotic expansion is Λ/2,
    # so for significantly larger frequencies we use the asymptotics,
    # since it has lower relative error.
    even_odd = Dict(fermion => :odd, boson => :even)[statistics]
    uhat = hat.(u, even_odd; n_asymp=conv_radius(kernel))
    rts = roots(last(v))
    sampling_points_v = [v.xmin; (rts[begin:(end - 1)] .+ rts[(begin + 1):end]) / 2; v.xmax]
    return IRBasis(kernel, u, uhat, s, v, sampling_points_v, statistics)
end

Λ(basis::IRBasis) = basis.kernel.Λ
is_well_conditioned(::IRBasis) = true

function Base.getindex(basis::IRBasis, i)
    sve_result = basis.u[i], basis.s[i], basis.v[i]
    return IRBasis(basis.statistics, Λ(basis); kernel=basis.kernel, sve_result)
end

function get_kernel(statistics, Λ, kernel)
    statistics ∈ (fermion, boson) ||
        error("""statistics must be either boson (for fermionic basis) or fermion (for bosonic basis)""")
    if isnothing(kernel)
        kernel = LogisticKernel(Λ)
    else
        @assert kernel.Λ ≈ Λ
    end
    return kernel
end

struct FiniteTempBasis{K<:AbstractKernel,T<:Real} <: AbstractBasis
    kernel::K
    sve_result::Tuple{PiecewiseLegendrePolyArray{T},Vector{T},PiecewiseLegendrePolyArray{T}}
    statistics::Statistics
    β::T
    u::PiecewiseLegendrePolyArray{T}
    v::PiecewiseLegendrePolyArray{T}
    s::Vector{T}
    uhat::PiecewiseLegendreFTArray{T}
end

function FiniteTempBasis(statistics, β, wmax, ε=nothing; kernel=nothing, sve_result=nothing)
    β > 0 || error("Inverse temperature β must be positive")
    wmax ≥ 0 || error("Frequency cutoff wmax must be non-negative")

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
    even_odd = Dict(fermion => :odd, boson => :even)[statistics]
    uhat = hat.(û_base, even_odd; n_asymp=conv_radius)

    return FiniteTempBasis(kernel, (u, s, v), statistics, β, u_, v_, s_, uhat)
end

Base.firstindex(::AbstractBasis) = 1
Base.length(basis::AbstractBasis) = length(basis.s)
is_well_conditioned(::IRBasis) = true

function Base.getindex(basis::FiniteTempBasis, i)
    u, s, v = basis.sve_result
    sve_result = u[i], s[i], v[i]
    return FiniteTempBasis(basis.statistics, basis.β, wmax(basis); kernel=basis.kernel,
                           sve_result)
end

wmax(basis::FiniteTempBasis) = basis.kernel.Λ / basis.β

function finite_temp_bases(β, wmax, ε, sve_result=compute(LogisticKernel(β * wmax); ε))
    basis_f = FiniteTempBasis(fermion, β, wmax, ε; sve_result)
    basis_b = FiniteTempBasis(boson, β, wmax, ε; sve_result)
    return basis_f, basis_b
end

default_tau_sampling_points(basis::AbstractBasis) = _default_sampling_points(basis.u)
default_matsubara_sampling_points(basis::AbstractBasis; mitigate=true) = _default_matsubara_sampling_points(basis.uhat, mitigate)

function _default_sampling_points(u)
    poly = last(u)
    maxima = roots(deriv(poly))
    left = (first(maxima) + poly.xmin) / 2
    right = (last(maxima) + poly.xmax) / 2
    return [left; maxima; right]
end

function _default_matsubara_sampling_points(uhat, mitigate=true)
    # Use the (discrete) extrema of the corresponding highest-order basis
    # function in Matsubara.  This turns out to be close to optimal with
    # respect to conditioning for this size (within a few percent).
    polyhat = last(uhat)
    wn = extrema(polyhat)

    # While the condition number for sparse sampling in tau saturates at a
    # modest level, the conditioning in Matsubara steadily deteriorates due
    # to the fact that we are not free to set sampling points continuously.
    # At double precision, tau sampling is better conditioned than iwn
    # by a factor of ~4 (still OK). To battle this, we fence the largest
    # frequency with two carefully chosen oversampling points, which brings
    # the two sampling problems within a factor of 2.
    if mitigate
        wn_outer = [first(wn), last(wn)]
        wn_diff = 2 * round.(Int, 0.025 * wn_outer)
        length(wn) ≥ 20 && append!(wn, wn_outer - wn_diff)
        length(wn) ≥ 42 && append!(wn, wn_outer + wn_diff)
        unique!(wn)
    end

    if iseven(first(wn))
        pushfirst!(wn, 0)
        unique!(wn)
    end

    return wn
end