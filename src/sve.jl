export compute

const HAVE_XPREC = false # TODO:

abstract type AbstractSVE end

"""
    SamplingSVE <: AbstractSVE

SVE to SVD translation by sampling technique [1].

Maps the singular value expansion (SVE) of a kernel `kernel` onto the singular
value decomposition of a matrix `A`.  This is achieved by chosing two
sets of Gauss quadrature rules: `(x, wx)` and `(y, wy)` and
approximating the integrals in the SVE equations by finite sums.  This
implies that the singular values of the SVE are well-approximated by the
singular values of the following matrix:

    A[i, j] = √(wx[i]) * K(x[i], y[j]) * √(wy[j])

and the values of the singular functions at the Gauss sampling points can
be reconstructed from the singular vectors `u` and `v` as follows:

    u[l,i] ≈ √(wx[i]) u[l](x[i])
    v[l,j] ≈ √(wy[j]) u[l](y[j])

[1] P. Hansen, Discrete Inverse Problems, Ch. 3.1
"""
struct SamplingSVE{K<:AbstractKernel} <: AbstractSVE
    kernel::K
    ε::Float64
    n_gauss::Int
    nsvals_hint::Int

    # internal
    rule::Rule{Float64}
    segs_x::Vector{Float64}
    segs_y::Vector{Float64}
    gauss_x::Rule{Float64}
    gauss_y::Rule{Float64}
    sqrtw_x::Vector{Float64}
    sqrtw_y::Vector{Float64}
end

function SamplingSVE(kernel, ε; n_gauss=nothing, T=Float64)
    sve_hints_ = sve_hints(kernel, ε)
    isnothing(n_gauss) && (n_gauss = ngauss(sve_hints_))
    rule = legendre(n_gauss, T)
    segs_x, segs_y = convert(Vector{T}, segments_x(sve_hints_)),
                     convert(Vector{T}, segments_y(sve_hints_))
    gauss_x, gauss_y = piecewise(rule, segs_x), piecewise(rule, segs_y)

    return SamplingSVE(kernel, ε, n_gauss, nsvals(sve_hints_), rule, segs_x, segs_y,
                       gauss_x, gauss_y, sqrt.(gauss_x.w), sqrt.(gauss_y.w))
end

"""
    CentrosymmSVE <: AbstractSVE

SVE of centrosymmetric kernel in block-diagonal (even/odd) basis.

For a centrosymmetric kernel `K`, i.e., a kernel satisfying:
`K(x, y) == K(-x, -y)`, one can make the following ansatz for the
singular functions:

    u[l](x) = ured[l](x) + sign[l] * ured[l](-x)
    v[l](y) = vred[l](y) + sign[l] * ured[l](-y)

where `sign[l]` is either `+1` or `-1`.  This means that the singular value
expansion can be block-diagonalized into an even and an odd part by
(anti-)symmetrizing the kernel:

    K_even = K(x, y) + K(x, -y)
    K_odd  = K(x, y) - K(x, -y)

The `l`th basis function, restricted to the positive interval, is then
the singular function of one of these kernels.  If the kernel generates a
Chebyshev system [1], then even and odd basis functions alternate.

[1]: A. Karlin, Total Positivity (1968).
"""
struct CentrosymmSVE{K<:AbstractKernel,T,SVEeven<:AbstractSVE,SVEodd<:AbstractSVE} <:
       AbstractSVE
    kernel::K
    ε::T
    even::SVEeven
    odd::SVEodd
    nsvals_hint::Int
end

function CentrosymmSVE(kernel, ε; InnerSVE=SamplingSVE, inner_args...)
    even = InnerSVE(get_symmetrized(kernel, +1), ε; inner_args...)
    odd = InnerSVE(get_symmetrized(kernel, -1), ε; inner_args...)
    return CentrosymmSVE(kernel, ε, even, odd, max(even.nsvals_hint, odd.nsvals_hint))
end

"""
    compute(kernel; 
        ε=nothing, n_sv=typemax(Int), n_gauss=nothing, T=Float64, Twork=nothing,
        sve_strat=iscentrosymmetric(kernel) ? CentrosymmSVE : SamplingSVE,
        svd_strat=nothing)

Perform truncated singular value expansion of a kernel.

Perform a truncated singular value expansion (SVE) of an integral
kernel `kernel : [xmin, xmax] x [ymin, ymax] -> ℝ`:

    kernel(x, y) == sum(s[l] * u[l](x) * v[l](y) for l in (1, 2, 3, ...)),

where `s[l]` are the singular values, which are ordered in non-increasing
fashion, `u[l](x)` are the left singular functions, which form an
orthonormal system on `[xmin, xmax]`, and `v[l](y)` are the right
singular functions, which form an orthonormal system on `[ymin, ymax]`.

The SVE is mapped onto the singular value decomposition (SVD) of a matrix
by expanding the kernel in piecewise Legendre polynomials (by default by
using a collocation).

# Arguments
- `eps::AbstractFloat`:  Relative cutoff for the singular values.
- `n_sv::Integer`: Maximum basis size. If given, only at most the `n_sv` most
significant singular values and associated singular functions are
returned.
- `n_gauss::Integer`: Order of Legendre polynomials. Defaults to hinted value
by the kernel.
- `T`: Data type of the result.
- `Twork`: Working data type. Defaults to a data type with
machine epsilon of at least `eps^2`, or otherwise most accurate data
type available.
- `sve_strat`: SVE to SVD translation strategy. Defaults to SamplingSVE.
- `svd_strat`: SVD solver. Defaults to fast (ID/RRQR) based solution 
when accuracy goals are moderate, and more accurate Jacobi-based 
algorithm otherwise.

# Return value
Return tuple `(u, s, v)`, where:
- `u::PiecewiseLegendrePoly`: the left singular functions
- `s::Vector`: singular values
- `v::PiecewiseLegendrePoly`: the right singular functions
"""
function compute(kernel::AbstractKernel; ε=nothing, n_sv=typemax(Int), n_gauss=nothing,
                 T=Float64, Twork=nothing,
                 sve_strat=iscentrosymmetric(kernel) ? CentrosymmSVE : SamplingSVE,
                 svd_strat=:default)
    if isnothing(ε) || isnothing(Twork) || isnothing(svd_strat)
        ε, Twork, default_svd_strat = choose_accuracy(ε, Twork)
    end
    if svd_strat == :default
        svd_strat = default_svd_strat
    end
    println("Twork", Twork)
    # return Twork
    sve = sve_strat(kernel, ε; n_gauss, T=Twork)
    svds = [compute(matrix; n_sv_hint=sve.nsvals_hint, strategy=svd_strat)
            for matrix in matrices(sve)]
    u, s, v = zip(svds...)
    u, s, v = truncate(u, s, v, ε, n_sv)
    return postprocess(sve, u, s, v, T)
end

"""
    matrices(sve::AbstractSVE)

SVD problems underlying the SVE.
"""
function matrices(sve::SamplingSVE)
    result = matrix_from_gauss(sve.kernel, sve.gauss_x, sve.gauss_y)
    result .*= sve.sqrtw_x
    result .*= sve.sqrtw_y'
    return (result,)
end
matrices(sve::CentrosymmSVE) = (only(matrices(sve.even)), only(matrices(sve.odd)))

"""
    postprocess(sve::AbstractSVE, u, s, v, T=nothing)

Construct the SVE result from the SVD.
"""
function postprocess(sve::SamplingSVE, u, s, v, T=nothing)
    isnothing(T) && (T = promote_type(eltype(u), eltype(s), eltype(v)))

    s = T.(s)
    u_x = u ./ sve.sqrtw_x
    v_y = v ./ sve.sqrtw_y

    # TODO: Surely this can be done much more elegantly.
    # As is it feels prety much unmaintenable
    u_x = permutedims(reshape(permutedims(u_x),
                              (length(s), sve.n_gauss, length(sve.segs_x) - 1)), (2, 3, 1))
    v_y = permutedims(reshape(permutedims(v_y),
                              (length(s), sve.n_gauss, length(sve.segs_y) - 1)), (2, 3, 1))

    cmat = legendre_collocation(sve.rule)
    u_data = reshape(cmat * reshape(u_x, (size(u_x, 1), :)),
                     (:, size(u_x, 2), size(u_x, 3)))
    v_data = reshape(cmat * reshape(v_y, (size(v_y, 1), :)),
                     (:, size(v_y, 2), size(v_y, 3)))

    dsegs_x = diff(sve.segs_x)
    dsegs_y = diff(sve.segs_y)
    u_data .*= sqrt.(0.5 * dsegs_x')
    v_data .*= sqrt.(0.5 * dsegs_y')

    # Construct polynomials
    ulx = PiecewiseLegendrePolyArray(T.(u_data), T.(sve.segs_x))
    vly = PiecewiseLegendrePolyArray(T.(v_data), T.(sve.segs_y))
    _canonicalize!(ulx, vly)
    return ulx, s, vly
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

"""
    choose_accuracy(ε, Twork)

Choose work type and accuracy based on specs and defaults
"""
function choose_accuracy(ε, Twork)
    if isnothing(ε)
        if isnothing(Twork)
            return sqrt(MAX_EPS), MAX_T, :fast  # TODO: adjust for extended precision
        end
        safe_ε = sqrt(eps(Twork))
        return safe_ε, Twork, :fast
    end

    if isnothing(Twork)
        if ε ≥ sqrt(eps(Float64))
            return ε, Float64, :fast
        end
        Twork = MAX_T
    end

    safe_ε = sqrt(eps(Twork))
    if ε ≥ safe_ε
        svd_strat = :fast
    else
        svd_strat = :accurate
        @warn """Basis cutoff is $ε, which is below sqrt(eps) with eps = $(safe_ε^2).
        Expect singular values and basis functions for large l to have lower precision
        than the cutoff.
        """
    end

    return ε, Twork, svd_strat
end

"""
    canonicalize!(u, v)

Canonicalize basis.

Each SVD `(u[l], v[l])` pair is unique only up to a global phase, which may
differ from implementation to implementation and also platform.  We
fix that gauge by demanding `u[l](1) > 0`.  This ensures a diffeomorphic
connection to the Legendre polynomials as `Λ → 0`.
"""
function _canonicalize!(ulx, vly)
    gauge = sign.(ulx(1))
    for i in eachindex(ulx, vly, gauge)
        ulx[i].data .*= 1 / gauge[i]
        vly[i].data .*= gauge[i]
    end
end

"""
    truncate(u, s, v, rtol=0, lmax=nothing)

Truncate singular value expansion.

# Arguments
    - `u`, `s`, `v`: Thin singular value expansion
    - `rtol` : If given, only singular values satisfying
    `s[l]/s[0] > rtol` are retained.
    - `lmax` : If given, at most the `lmax` most significant singular
    values are retained.
"""
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
