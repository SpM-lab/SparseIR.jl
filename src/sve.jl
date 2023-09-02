using LinearAlgebra

"""
    SamplingSVE <: AbstractSVE

SVE to SVD translation by sampling technique [1].

Maps the singular value expansion (SVE) of a kernel `kernel` onto the singular
value decomposition of a matrix `A`. This is achieved by choosing two
sets of Gauss quadrature rules: `(x, wx)` and `(y, wy)` and
approximating the integrals in the SVE equations by finite sums. This
implies that the singular values of the SVE are well-approximated by the
singular values of the following matrix:

    A[i, j] = √(wx[i]) * K(x[i], y[j]) * √(wy[j])

and the values of the singular functions at the Gauss sampling points can
be reconstructed from the singular vectors `u` and `v` as follows:

    u[l,i] ≈ √(wx[i]) u[l](x[i])
    v[l,j] ≈ √(wy[j]) u[l](y[j])

[1] P. Hansen, Discrete Inverse Problems, Ch. 3.1
"""
struct SamplingSVE{T<:AbstractFloat,K<:AbstractKernel} <: AbstractSVE
    kernel      :: K
    ε           :: Float64
    n_gauss     :: Int
    nsvals_hint :: Int

    rule    :: Rule{T}
    segs_x  :: Vector{T}
    segs_y  :: Vector{T}
    gauss_x :: Rule{T}
    gauss_y :: Rule{T}
end

function SamplingSVE(kernel, ε, ::Type{T}=Float64; n_gauss=nothing) where {T}
    sve_hints_ = sve_hints(kernel, ε)
    n_gauss = something(n_gauss, ngauss(sve_hints_))
    rule = legendre(n_gauss, T)
    segs_x, segs_y = segments_x(sve_hints_, T), segments_y(sve_hints_, T)
    gauss_x, gauss_y = piecewise(rule, segs_x), piecewise(rule, segs_y)

    return SamplingSVE(kernel, ε, n_gauss, nsvals(sve_hints_),
                       rule, segs_x, segs_y, gauss_x, gauss_y)
end

"""
    CentrosymmSVE <: AbstractSVE

SVE of centrosymmetric kernel in block-diagonal (even/odd) basis.

For a centrosymmetric kernel `K`, i.e., a kernel satisfying:
`K(x, y) == K(-x, -y)`, one can make the following ansatz for the
singular functions:

    u[l](x) = ured[l](x) + sign[l] * ured[l](-x)
    v[l](y) = vred[l](y) + sign[l] * ured[l](-y)

where `sign[l]` is either `+1` or `-1`. This means that the singular value
expansion can be block-diagonalized into an even and an odd part by
(anti-)symmetrizing the kernel:

    K_even = K(x, y) + K(x, -y)
    K_odd  = K(x, y) - K(x, -y)

The `l`th basis function, restricted to the positive interval, is then
the singular function of one of these kernels. If the kernel generates a
Chebyshev system [1], then even and odd basis functions alternate.

[1]: A. Karlin, Total Positivity (1968).
"""
struct CentrosymmSVE{K<:AbstractKernel,SVEEVEN<:AbstractSVE,SVEODD<:AbstractSVE} <:
       AbstractSVE
    kernel      :: K
    ε           :: Float64
    even        :: SVEEVEN
    odd         :: SVEODD
    nsvals_hint :: Int
end

function CentrosymmSVE(kernel, ε, ::Type{T}; InnerSVE=SamplingSVE,
                       n_gauss=nothing) where {T}
    even = InnerSVE(get_symmetrized(kernel, +1), ε, T; n_gauss)
    odd = InnerSVE(get_symmetrized(kernel, -1), ε, T; n_gauss)
    return CentrosymmSVE(kernel, ε, even, odd, max(even.nsvals_hint, odd.nsvals_hint))
end

struct SVEResult{K}
    u::PiecewiseLegendrePolyVector
    s::Vector{Float64}
    v::PiecewiseLegendrePolyVector

    kernel::K
    ε::Float64
end

"""
    SVEResult(kernel::AbstractKernel;
        Twork=nothing, ε=nothing, lmax=typemax(Int),
        n_gauss=nothing, svd_strat=:auto,
        sve_strat=iscentrosymmetric(kernel) ? CentrosymmSVE : SamplingSVE
    )

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

  - `K::AbstractKernel`: Integral kernel to take SVE from.

  - `ε::Real`: Accuracy target for the basis: attempt to have singular values down
    to a relative magnitude of `ε`, and have each singular value
    and singular vector be accurate to `ε`. A `Twork` with
    a machine epsilon of `ε^2` or lower is required to satisfy
    this. Defaults to `2.2e-16` if xprec is available, and `1.5e-8`
    otherwise.
  - `cutoff::Real`: Relative cutoff for the singular values. A `Twork` with
    machine epsilon of `cutoff` is required to satisfy this.
    Defaults to a small multiple of the machine epsilon.

    Note that `cutoff` and `ε` serve distinct purposes. `cutoff`
    reprsents the accuracy to which the kernel is reproduced, whereas
    `ε` is the accuracy to which the singular values and vectors
    are guaranteed.
  - `lmax::Integer`: Maximum basis size. If given, only at most the `lmax` most
    significant singular values and associated singular functions are returned.
  - `n_gauss (int): Order of Legendre polynomials. Defaults to kernel hinted value.
  - `Twork``: Working data type. Defaults to a data type with machine epsilon of at  most `ε^2`and at most `cutoff`, or otherwise most accurate data type available.
  - `sve_strat::AbstractSVE`: SVE to SVD translation strategy. Defaults to `SamplingSVE`,
    optionally wrapped inside of a `CentrosymmSVE` if the kernel is centrosymmetric.
  - `svd_strat` ('fast' or 'default' or 'accurate'): SVD solver. Defaults to fast
    (ID/RRQR) based solution when accuracy goals are moderate, and more accurate
    Jacobi-based algorithm otherwise.

Returns:
An `SVEResult` containing the truncated singular value expansion.
"""
function SVEResult(kernel::AbstractKernel;
                   Twork=nothing, cutoff=nothing, ε=nothing, lmax=typemax(Int),
                   n_gauss=nothing, svd_strat=:auto,
                   SVEstrat=iscentrosymmetric(kernel) ? CentrosymmSVE : SamplingSVE)
    safe_ε, Twork_actual, svd_strat = choose_accuracy(ε, Twork, svd_strat)
    sve = SVEstrat(kernel, safe_ε, Twork_actual; n_gauss)

    svds = compute_svd.(matrices(sve); strategy=svd_strat)
    u_, s_, v_ = zip(svds...)
    cutoff_actual = something(cutoff, 2eps(Twork_actual))
    u, s, v = truncate(u_, s_, v_; rtol=cutoff_actual, lmax)
    return postprocess(sve, u, s, v)
end

function part(sve::SVEResult; ε=nothing, max_size=nothing)
    ε = something(ε, sve.ε)
    cut = count(≥(ε * first(sve.s)), sve.s)
    if !isnothing(max_size)
        cut = min(cut, max_size)
    end
    return sve.u[1:cut], sve.s[1:cut], sve.v[1:cut]
end

Base.iterate(sve::SVEResult, n=1) = n ≤ 3 ? ((sve.u, sve.s, sve.v)[n], n + 1) : nothing

"""
    matrices(sve::AbstractSVE)

SVD problems underlying the SVE.
"""
function matrices(sve::SamplingSVE)
    result = matrix_from_gauss(sve.kernel, sve.gauss_x, sve.gauss_y)
    result .*= sqrt.(sve.gauss_x.w)
    result .*= sqrt.(permutedims(sve.gauss_y.w))
    return (result,)
end
matrices(sve::CentrosymmSVE) = (only(matrices(sve.even)), only(matrices(sve.odd)))

"""
    postprocess(sve::AbstractSVE, u, s, v)

Construct the SVE result from the SVD.
"""
function postprocess(sve::SamplingSVE, (u,), (s,), (v,))
    s = Float64.(s)
    u_x = u ./ sqrt.(sve.gauss_x.w)
    v_y = v ./ sqrt.(sve.gauss_y.w)

    u_x = reshape(u_x, (sve.n_gauss, length(sve.segs_x) - 1, length(s)))
    v_y = reshape(v_y, (sve.n_gauss, length(sve.segs_y) - 1, length(s)))

    cmat = legendre_collocation(sve.rule)
    u_data = reshape(cmat * reshape(u_x, (size(u_x, 1), :)),
                     (:, size(u_x, 2), size(u_x, 3)))
    v_data = reshape(cmat * reshape(v_y, (size(v_y, 1), :)),
                     (:, size(v_y, 2), size(v_y, 3)))

    dsegs_x = diff(sve.segs_x)
    dsegs_y = diff(sve.segs_y)
    u_data .*= sqrt.(0.5 .* reshape(dsegs_x, (1, :)))
    v_data .*= sqrt.(0.5 .* reshape(dsegs_y, (1, :)))

    # Construct polynomials
    ulx = PiecewiseLegendrePolyVector(Float64.(u_data), Float64.(sve.segs_x))
    vly = PiecewiseLegendrePolyVector(Float64.(v_data), Float64.(sve.segs_y))
    canonicalize!(ulx, vly)
    return SVEResult(ulx, s, vly, sve.kernel, sve.ε)
end

function postprocess(sve::CentrosymmSVE, u, s, v)
    u_even, s_even, v_even = postprocess(sve.even, u[1:1], s[1:1], v[1:1])
    u_odd, s_odd, v_odd = postprocess(sve.odd, u[2:2], s[2:2], v[2:2])

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
        u_pos_data = u[i].data / sqrt(2)
        v_pos_data = v[i].data / sqrt(2)

        u_neg_data = reverse(u_pos_data; dims=2) .* poly_flip_x * signs[i]
        v_neg_data = reverse(v_pos_data; dims=2) .* poly_flip_x * signs[i]
        u_data = hcat(u_neg_data, u_pos_data)
        v_data = hcat(v_neg_data, v_pos_data)
        u_complete[i] = PiecewiseLegendrePoly(u_data, segs_x, i - 1; symm=signs[i])
        v_complete[i] = PiecewiseLegendrePoly(v_data, segs_y, i - 1; symm=signs[i])
    end

    return SVEResult(u_complete, s, v_complete, sve.kernel, sve.ε)
end

"""
    choose_accuracy(ε, Twork[, svd_strat])

Choose work type and accuracy based on specs and defaults
"""
function choose_accuracy(ε, Twork, svd_strat)
    ε, Twork, auto_svd_strat = choose_accuracy(ε, Twork)
    svd_strat === :auto && (svd_strat = auto_svd_strat)
    return ε, Twork, svd_strat
end
function choose_accuracy(ε, Twork)
    if ε ≥ sqrt(eps(Twork))
        return ε, Twork, :default
    else
        @warn """Basis cutoff is $ε, which is below √ε with ε = $(eps(Twork)).
        Expect singular values and basis functions for large l to have lower precision
        than the cutoff."""
        return ε, Twork, :accurate
    end
end
function choose_accuracy(ε, ::Nothing)
    if ε ≥ sqrt(eps(Float64))
        return ε, Float64, :default
    else
        if ε < sqrt(eps(T_MAX))
            @warn """Basis cutoff is $ε, which is below √ε with ε = $(eps(T_MAX)).
            Expect singular values and basis functions for large l to have lower precision
            than the cutoff."""
        end
        return ε, T_MAX, :default
    end
end
choose_accuracy(::Nothing, Twork) = sqrt(eps(Twork)), Twork, :default
choose_accuracy(::Nothing, ::Nothing) = Float64(sqrt(eps(T_MAX))), T_MAX, :default

"""
    canonicalize!(u, v)

Canonicalize basis.

Each SVD `(u[l], v[l])` pair is unique only up to a global phase, which may
differ from implementation to implementation and also platform. We
fix that gauge by demanding `u[l](1) > 0`. This ensures a diffeomorphic
connection to the Legendre polynomials as `Λ → 0`.
"""
function canonicalize!(ulx, vly)
    for i in eachindex(ulx, vly)
        gauge = sign(ulx[i](1))
        ulx[i].data .*= gauge
        vly[i].data .*= gauge
    end
end

"""
    truncate(u, s, v; rtol=0.0, lmax=typemax(Int))

Truncate singular value expansion.

# Arguments

    - `u`, `s`, `v`: Thin singular value expansion
    - `rtol`: Only singular values satisfying `s[l]/s[1] > rtol` are retained.
    - `lmax`: At most the `lmax` most significant singular values are retained.
"""
function truncate(u, s, v; rtol=0.0, lmax=typemax(Int))
    lmax ≥ 0 || throw(DomainError(lmax, "lmax must be non-negative"))
    0 ≤ rtol ≤ 1 || throw(DomainError(rtol, "rtol must be in [0, 1]"))

    sall = sort!(vcat(s...); rev=true)

    # Determine singular value cutoff. Note that by selecting a cutoff even
    # in the case of lmax, we make sure to never remove parts of a degenerate
    # singular value space, rather, we reduce the size of the basis.
    cutoff = if lmax < length(sall)
        max(rtol * first(sall), sall[lmax])
    else
        rtol * first(sall)
    end

    # Determine how many singular values survive in each group
    scount = map(si -> count(>(cutoff), si), s)

    u_cut = map((ui, scounti) -> ui[:, 1:scounti], u, scount)
    s_cut = map((si, scounti) -> si[1:scounti], s, scount)
    v_cut = map((vi, scounti) -> vi[:, 1:scounti], v, scount)
    return u_cut, s_cut, v_cut
end
