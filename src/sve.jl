using SparseIR
import Tullio: @tullio

export compute, SamplingSVE, CentroSymmSVE, choose_acuracy

const HAVE_XPREC = false # TODO

struct SamplingSVE
    K::Any
    ε::Any
    n_gauss::Any
    nsvals_hint::Any

    # internal
    rule::Any
    segs_x::Any
    segs_y::Any
    gauss_x::Any
    gauss_y::Any
    sqrtw_x::Any
    sqrtw_y::Any
end

struct CentrosymmSVE
    K::Any
    ε::Any
    even::Any
    odd::Any
    nsvals_hint::Any
end

function compute(K, ε=nothing, n_sv=nothing, n_gauss=nothing, T=Float64, work_T=nothing,
                 sve_strat=nothing, svd_strat=nothing)
    if isnothing(ε) || isnothing(work_T) || isnothing(svd_strat)
        ε, work_T, default_svd_strat = choose_accuracy(ε, work_T)
    end
    if isnothing(svd_strat)
        svd_strat = default_svd_strat
    end
    if isnothing(sve_strat)
        sve_strat = is_centrosymmetric(K) ? CentrosymmSVE : SamplingSVE
    end
    sve = sve_strat(K, ε; n_gauss=n_gauss, T=work_T)
    u, s, v = zip((compute(matrix, sve.nsvals_hint, svd_strat) for matrix in
                                                                   matrices(sve))...)
    u, s, v = truncate(u, s, v, ε, n_sv)
    return postprocess(sve, u, s, v, T)
end

function SamplingSVE(K, ε; n_gauss=nothing, T=Float64)
    sve_hints_ = sve_hints(K, ε)
    isnothing(n_gauss) && (n_gauss = ngauss(sve_hints_))
    rule = legendre(n_gauss, T)
    segs_x, segs_y = T.(segments_x(sve_hints_)), T.(segments_y(sve_hints_))
    gauss_x, gauss_y = piecewise(rule, segs_x), piecewise(rule, segs_y)

    return SamplingSVE(K, ε, n_gauss, nsvals(sve_hints_), rule, segs_x, segs_y, gauss_x,
                       gauss_y, sqrt.(gauss_x.w), sqrt.(gauss_y.w))
end

function matrices(sve::SamplingSVE)
    result = matrix_from_gauss(sve.K, sve.gauss_x, sve.gauss_y)
    result .*= sve.sqrtw_x
    result .*= sve.sqrtw_y'
    return (result,)
end

function postprocess(sve::SamplingSVE, u, s, v, T=nothing)
    isnothing(T) && (T = promote_type(eltype(u), eltype(s), eltype(v)))

    s = T.(s)
    u_x = u ./ sve.sqrtw_x
    v_y = v ./ sve.sqrtw_y

    u_x = reshape(u_x, (length(sve.segs_x) - 1, sve.n_gauss, length(s)))
    v_y = reshape(v_y, (length(sve.segs_y) - 1, sve.n_gauss, length(s)))

    cmat = legendre_collocation(sve.rule)
    # lx,ixs -> ils -> lis
    @tullio u_data[i, j, k] := cmat[i, l] * u_x[j, l, k]
    @tullio v_data[i, j, k] := cmat[i, l] * v_y[j, l, k]
    # v_data = permutedims(cmat * v_y, (2, 1, 3))

    dsegs_x = diff(sve.segs_x)
    dsegs_y = diff(sve.segs_y)
    @tullio u_data[i, j, k] *= sqrt.(dsegs_x / 2)[j]
    @tullio v_data[i, j, k] *= sqrt.(dsegs_y / 2)[j]

    # Construct polynomial
    ulx = PiecewiseLegendrePoly(T.(u_data), T.(sve.segs_x))
    vly = PiecewiseLegendrePoly(T.(v_data), T.(sve.segs_y))
    canonicalize(ulx, vly)
    return ulx, s, vly
end

function CentrosymmSVE(K, ε; InnerSVE=nothing, inner_args...)
    isnothing(InnerSVE) && (InnerSVE = SamplingSVE)

    even = InnerSVE(get_symmetrized(K, +1), ε; inner_args...)
    odd = InnerSVE(get_symmetrized(K, -1), ε; inner_args...)
    return CentrosymmSVE(K, ε, even, odd, max(even.nsvals_hint, odd.nsvals_hint))
end

function matrices(sve::CentrosymmSVE)
    return (matrices(sve.even)..., matrices(sve.odd)...)
end

function postprocess(sve::CentrosymmSVE, u, s, v, T)
    u_even, s_even, v_even = postprocess(sve.even, u[1], s[1], v[1], T)
    u_odd, s_odd, v_odd = postprocess(sve.odd, u[2], s[2], v[2], T)

    # Merge two sets - data is [legendre, segment, l]
    u_data = [u_even.data;;; u_odd.data] # the ;;; makes it concatenate along the 3rd axis
    v_data = [v_even.data;;; v_odd.data]
    s = [s_even; s_odd]
    signs = [fill(1, length(s_even)); fill(-1, length(s_odd))]

    # Sort: now for totally positive kernels like defined in this module,
    # this strictly speaking is not necessary as we know that the even/odd
    # functions intersperse.
    sort = sortperm(s; rev=true)
    u_data = u_data[:, :, sort]
    v_data = v_data[:, :, sort]
    s = s[sort]
    signs = signs[sort]

    # Extend to the negative side
    inv_sqrt2 = 1 / √2
    u_data .*= inv_sqrt2
    v_data .*= inv_sqrt2
    poly_flip_x = (-1) .^ range(0; length=size(u_data, 1))

    u_neg = u_data[:, end:-1:begin, :] .* poly_flip_x[:, :, :] .* reshape(signs, 1, 1, :)
    v_neg = v_data[:, end:-1:begin, :] .* poly_flip_x[:, :, :] .* reshape(signs, 1, 1, :)
    u_data = [u_neg;; u_data]
    v_data = [v_neg;; v_data]

    full_hints = sve_hints(sve.K, sve.ε)
    u = PiecewiseLegendrePoly(u_data, segments_x(full_hints); symm=signs)
    v = PiecewiseLegendrePoly(v_data, segments_y(full_hints); symm=signs)
    return u, s, v
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
        if ε >= sqrt(eps(Float64))
            return ε, Float64, :fast
        end
        work_T = MAX_T
    end

    safe_ε = sqrt(eps(work_T))
    if ε >= safe_ε
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
        lmax >= 0 || error("lmax must be non-negative")
        lmax isa Integer || error("lmax must be an integer")
    end
    0 <= rtol <= 1 || error("rtol must be in [0, 1]")

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

function canonicalize(ulx, vly)
    gauge = sign.(ulx(1))
    @tullio ulx.data[i, j, k] *= 1 / gauge[k, 1]
    @tullio vly.data[i, j, k] *= gauge[k, 1]
end