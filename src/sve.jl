using SparseIR
import Tullio: @tullio

# TODO reduce exports
export compute, SamplingSVE, CentrosymmSVE, choose_accuracy, is_centrosymmetric, matrices

const HAVE_XPREC = false # TODO

abstract type AbstractSVE end

struct SamplingSVE{K<:AbstractKernel,T<:Real} <: AbstractSVE
    kernel::K
    ε::T
    n_gauss::Int
    nsvals_hint::Int

    # internal
    rule::Rule
    segs_x::Vector{T}
    segs_y::Vector{T}
    gauss_x::Rule{T}
    gauss_y::Rule{T}
    sqrtw_x::Vector{T}
    sqrtw_y::Vector{T}
end

struct CentrosymmSVE{K<:AbstractKernel,T<:Real,SVEeven<:AbstractSVE,SVEodd<:AbstractSVE} <:
       AbstractSVE
    kernel::K
    ε::T
    even::SVEeven
    odd::SVEodd
    nsvals_hint::Int
end

function SamplingSVE(kernel, ε; n_gauss=nothing, T::Type{<:Real}=Float64)
    sve_hints_ = sve_hints(kernel, ε)
    isnothing(n_gauss) && (n_gauss = ngauss(sve_hints_))
    rule = legendre(n_gauss, T)
    segs_x, segs_y = T.(segments_x(sve_hints_)), T.(segments_y(sve_hints_))
    gauss_x, gauss_y = piecewise(rule, segs_x), piecewise(rule, segs_y)

    return SamplingSVE(kernel, ε, n_gauss, nsvals(sve_hints_), rule, segs_x, segs_y,
                       gauss_x, gauss_y, sqrt.(gauss_x.w), sqrt.(gauss_y.w))
end

# TODO: surely this can be done more elegantly
# TODO: remove explicit use of Float64
function compute(kernel::AbstractKernel; ε::Union{Float64,Nothing}=nothing,
                 n_sv=typemax(Int), n_gauss::Union{Int,Nothing}=nothing,
                 T::Type{<:Real}=Float64, work_T::Union{Type{<:Real},Nothing}=nothing,
                 sve_strat::Type{<:AbstractSVE}=is_centrosymmetric(kernel) ? CentrosymmSVE :
                                                SamplingSVE,
                 svd_strat::Union{Symbol,Nothing}=nothing)
    if isnothing(ε) || isnothing(work_T) || isnothing(svd_strat)
        ε, work_T, default_svd_strat = choose_accuracy(ε, work_T)
    end
    if isnothing(svd_strat)
        svd_strat = default_svd_strat
    end
    sve = sve_strat(kernel, ε; n_gauss, T=work_T)
    svds = [compute(matrix; n_sv_hint=sve.nsvals_hint, strategy=svd_strat) for matrix in matrices(sve)]
    u, s, v = zip(svds...)
    u, s, v = truncate(u, s, v, ε, n_sv)
    return postprocess(sve, u, s, v, T)
end

function matrices(sve::SamplingSVE)
    result = matrix_from_gauss(sve.kernel, sve.gauss_x, sve.gauss_y)
    result .*= sve.sqrtw_x
    result .*= sve.sqrtw_y'
    return (result,)
end

function postprocess(sve::SamplingSVE, u, s, v, T=nothing)
    isnothing(T) && (T = promote_type(eltype(u), eltype(s), eltype(v)))

    s = T.(s)
    u_x = u ./ sve.sqrtw_x
    v_y = v ./ sve.sqrtw_y

    # TODO: Surely all this can be done much more elegantly

    u_x = permutedims(reshape(permutedims(u_x),
                              (length(s), sve.n_gauss, length(sve.segs_x) - 1)), (3, 2, 1))
    v_y = permutedims(reshape(permutedims(v_y),
                              (length(s), sve.n_gauss, length(sve.segs_y) - 1)), (3, 2, 1))

    cmat = legendre_collocation(sve.rule)
    @tullio u_data[l, i, s] := cmat[l, x] * u_x[i, x, s]
    @tullio v_data[l, i, s] := cmat[l, x] * v_y[i, x, s]

    dsegs_x = diff(sve.segs_x)
    dsegs_y = diff(sve.segs_y)
    @tullio u_data[i, j, k] *= sqrt.(dsegs_x / 2)[j]
    @tullio v_data[i, j, k] *= sqrt.(dsegs_y / 2)[j]

    # Construct polynomial
    ulx = PiecewiseLegendrePolyArray(T.(u_data), T.(sve.segs_x))
    vly = PiecewiseLegendrePolyArray(T.(v_data), T.(sve.segs_y))
    canonicalize!(ulx, vly)
    return ulx, s, vly
end

function CentrosymmSVE(kernel, ε; InnerSVE=SamplingSVE, inner_args...)
    even = InnerSVE(get_symmetrized(kernel, +1), ε; inner_args...)
    odd = InnerSVE(get_symmetrized(kernel, -1), ε; inner_args...)
    return CentrosymmSVE(kernel, ε, even, odd, max(even.nsvals_hint, odd.nsvals_hint))
end

function matrices(sve::CentrosymmSVE)
    return (matrices(sve.even)..., matrices(sve.odd)...)
end

function postprocess(sve::CentrosymmSVE, u, s, v, T)
    u_even, s_even, v_even = postprocess(sve.even, u[1], s[1], v[1], T)
    u_odd, s_odd, v_odd = postprocess(sve.odd, u[2], s[2], v[2], T)

    # Merge two sets
    u = [u_even; u_odd]
    v = [v_even; v_odd]
    s = [s_even; s_odd]
    signs = [fill(1, length(s_even)); fill(-1, length(s_odd))]

    # Sort: now for totally positive kernels like defined in this module,
    # this strictly speaking is not necessary as we know that the even/odd
    # functions intersperse.
    sort = sortperm(s; rev=true)
    u = u[sort]
    v = v[sort]
    s = s[sort]
    signs = signs[sort]

    # Extend to the negative side
    u_complete = similar(u)
    v_complete = similar(v)
    full_hints = sve_hints(sve.kernel, sve.ε)
    segs_x = segments_x(full_hints)
    segs_y = segments_y(full_hints)

    poly_flip_x = (-1) .^ range(0; length=size(first(u).data, 1))
    for i in eachindex(u, v)
        u_pos_data = u[i].data / √2
        v_pos_data = v[i].data / √2

        u_neg_data = reverse(u_pos_data; dims=2) .* poly_flip_x * signs[i]
        v_neg_data = reverse(v_pos_data; dims=2) .* poly_flip_x * signs[i]
        u_data = [u_neg_data;; u_pos_data]
        v_data = [v_neg_data;; v_pos_data]
        u_complete[i] = PiecewiseLegendrePoly(u_data, segs_x; symm=signs[i])
        v_complete[i] = PiecewiseLegendrePoly(v_data, segs_y; symm=signs[i])
    end

    return u_complete, s, v_complete
end

function choose_accuracy(ε, work_T)
    if isnothing(ε)
        if isnothing(work_T)
            return sqrt(MAX_EPS), MAX_T, :fast  # TODO: adjust for extended precision
        end
        safe_ε = sqrt(eps(work_T))
        return safe_ε, work_T, :fast
    end

    if isnothing(work_T)
        if ε ≥ sqrt(eps(Float64))
            return ε, Float64, :fast
        end
        work_T = MAX_T
    end

    safe_ε = sqrt(eps(work_T))
    if ε ≥ safe_ε
        svd_strat = :fast
    else
        svd_strat = :accurate
        @warn """Basis cutoff is $ε, which is below sqrt(eps) with eps = $(safe_ε^2).
        Expect singular values and basis functions for large l to have lower precision
        than the cutoff.
        """
    end

    return ε, work_T, svd_strat
end

function truncate(u, s, v, rtol=0, lmax=nothing)
    if !isnothing(lmax)
        lmax ≥ 0 || error("lmax must be non-negative")
        lmax isa Integer || error("lmax must be an integer")
    end
    0 ≤ rtol ≤ 1 || error("rtol must be in [0, 1]")

    sall = vcat(s...)

    # Determine singular value cutoff.  Note that by selecting a cutoff even
    # in the case of lmax, we make sure to never remove parts of a degenerate
    # singular value space, rather, we reduce the size of the basis.
    ssort = sort(sall)
    cutoff = rtol * last(ssort)
    if !isnothing(lmax) && lmax < length(sall)
        cutoff = max(cutoff, s[end - lmax])
    end

    # Determine how many singular values survive in each group
    scount = [count(>(cutoff), si) for si in s]

    u_cut = [ui[:, begin:counti] for (ui, counti) in zip(u, scount)]
    s_cut = [si[begin:counti] for (si, counti) in zip(s, scount)]
    v_cut = [vi[:, begin:counti] for (vi, counti) in zip(v, scount)]
    return u_cut, s_cut, v_cut
end

function canonicalize!(ulx, vly)
    gauge = sign.(ulx(1))
    for i in eachindex(ulx, vly, gauge)
        ulx[i].data .*= 1 / gauge[i]
        vly[i].data .*= gauge[i]
    end
end