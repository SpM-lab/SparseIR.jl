abstract type AbstractSVE end

"""
    SamplingSVE <: AbstractSVE

SVE to SVD translation by sampling technique [1].

Maps the singular value expansion (SVE) of a kernel `kernel` onto the singular
value decomposition of a matrix `A`.  This is achieved by choosing two
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
struct SamplingSVE{T<:AbstractFloat,K<:AbstractKernel} <: AbstractSVE
    kernel      :: K
    ε           :: T
    n_gauss     :: Int
    nsvals_hint :: Int

    rule    :: Rule{T}
    segs_x  :: Vector{T}
    segs_y  :: Vector{T}
    gauss_x :: Rule{T}
    gauss_y :: Rule{T}
end

function SamplingSVE(kernel, ε; n_gauss=-1, T=Float64)
    sve_hints_ = sve_hints(kernel, ε)
    n_gauss = (n_gauss < 0) ? ngauss(sve_hints_) : n_gauss
    rule = legendre(n_gauss, T)
    segs_x, segs_y = T.(segments_x(sve_hints_)), T.(segments_y(sve_hints_))
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
struct CentrosymmSVE{K<:AbstractKernel,T,SVEEVEN<:AbstractSVE,SVEODD<:AbstractSVE} <:
       AbstractSVE
    kernel      :: K
    ε           :: T
    even        :: SVEEVEN
    odd         :: SVEODD
    nsvals_hint :: Int
end

function CentrosymmSVE(kernel, ε; InnerSVE=SamplingSVE, n_gauss, T)
    even = InnerSVE(get_symmetrized(kernel, +1), ε; n_gauss, T)
    odd = InnerSVE(get_symmetrized(kernel, -1), ε; n_gauss, T)
    return CentrosymmSVE(kernel, ε, even, odd, max(even.nsvals_hint, odd.nsvals_hint))
end

"""
    compute_sve(kernel::AbstractKernel;
        Twork=nothing, ε=nothing, n_sv=typemax(Int),
        n_gauss=-1, T=Float64, svd_strat=:auto,
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

  - `ε::AbstractFloat`:  Relative cutoff for the singular values.
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
function compute_sve(kernel::AbstractKernel;
                     Twork=nothing, ε=nothing, n_sv=typemax(Int),
                     n_gauss=-1, T=Float64, svd_strat=:auto,
                     sve_strat=iscentrosymmetric(kernel) ? CentrosymmSVE : SamplingSVE)
    ε, Twork, svd_strat = choose_accuracy(ε, Twork, svd_strat)

    sve = sve_strat(kernel, Twork(ε); n_gauss, T=Twork)

    svds = compute_svd.(matrices(sve); strategy=svd_strat)
    u_, s_, v_ = zip(svds...)
    u, s, v = truncate(u_, s_, v_, ε, n_sv)
    return postprocess(sve, u, s, v, T)
end

"""
    matrices(sve::AbstractSVE)

SVD problems underlying the SVE.
"""
function matrices(sve::SamplingSVE)
    result = matrix_from_gauss(sve.kernel, sve.gauss_x, sve.gauss_y)
    result .*= sqrt.(sve.gauss_x.w)
    result .*= sqrt.(transpose(sve.gauss_y.w))
    return (result,)
end
matrices(sve::CentrosymmSVE) = (only(matrices(sve.even)), only(matrices(sve.odd)))

"""
    postprocess(sve::AbstractSVE, u, s, v, T=nothing)

Construct the SVE result from the SVD.
"""
function postprocess(sve::SamplingSVE, u, s, v,
                     T=promote_type(eltype(u), eltype(s), eltype(v)))
    s = T.(s)
    u_x = u ./ sqrt.(sve.gauss_x.w)
    v_y = v ./ sqrt.(sve.gauss_y.w)

    u_x = reshape(u_x, (sve.n_gauss, length(sve.segs_x) - 1, length(s)))
    v_y = reshape(v_y, (sve.n_gauss, length(sve.segs_y) - 1, length(s)))

    cmat = legendre_collocation(sve.rule)
    u_data = reshape(cmat * reshape(u_x, (size(u_x, 1), :)), (:, size(u_x)[2:3]...))
    v_data = reshape(cmat * reshape(v_y, (size(v_y, 1), :)), (:, size(v_y)[2:3]...))

    dsegs_x = diff(sve.segs_x)
    dsegs_y = diff(sve.segs_y)
    u_data .*= sqrt.(0.5 .* transpose(dsegs_x))
    v_data .*= sqrt.(0.5 .* transpose(dsegs_y))

    # Construct polynomials
    ulx = PiecewiseLegendrePolyVector(T.(u_data), T.(sve.segs_x))
    vly = PiecewiseLegendrePolyVector(T.(v_data), T.(sve.segs_y))
    canonicalize!(ulx, vly)
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
        u_data = hcat(u_neg_data, u_pos_data)
        v_data = hcat(v_neg_data, v_pos_data)
        u_complete[i] = PiecewiseLegendrePoly(u_data, segs_x, i - 1; symm=signs[i])
        v_complete[i] = PiecewiseLegendrePoly(v_data, segs_y, i - 1; symm=signs[i])
    end

    return u_complete, s, v_complete
end

"""
    choose_accuracy(ε, Twork[, svd_strat])

Choose work type and accuracy based on specs and defaults
"""
function choose_accuracy(ε, Twork, svd_strat)
    ε, Twork, auto_svd_strat = choose_accuracy(ε, Twork)
    if svd_strat == :auto
        svd_strat = auto_svd_strat
    end
    return ε, Twork, svd_strat
end
function choose_accuracy(ε, Twork)
    if ε ≥ sqrt(eps(Twork))
        return ε, Twork, :default
    else
        @warn """Basis cutoff is $ε, which is below sqrt(eps) with eps = $(eps(Twork)).
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
            @warn """Basis cutoff is $ε, which is below sqrt(eps) with eps = $(eps(T_MAX)).
            Expect singular values and basis functions for large l to have lower precision
            than the cutoff."""
        end
        return ε, T_MAX, :default
    end
end
choose_accuracy(::Nothing, Twork) = sqrt(eps(Twork)), Twork, :default
choose_accuracy(::Nothing, ::Nothing) = sqrt(eps(T_MAX)), T_MAX, :default

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
    truncate(u, s, v, rtol=0, lmax=typemax(Int))

Truncate singular value expansion.

# Arguments

    - `u`, `s`, `v`: Thin singular value expansion
    - `rtol`: Only singular values satisfying `s[l]/s[1] > rtol` are retained.
    - `lmax`: At most the `lmax` most significant singular values are retained.
"""
function truncate(u, s, v, rtol=0.0, lmax=typemax(Int))
    lmax ≥ 0 || throw(DomainError(lmax, "lmax must be non-negative"))
    0 ≤ rtol ≤ 1 || throw(DomainError(rtol, "rtol must be in [0, 1]"))

    sall = sort!(vcat(s...); rev=true)

    # Determine singular value cutoff.  Note that by selecting a cutoff even
    # in the case of lmax, we make sure to never remove parts of a degenerate
    # singular value space, rather, we reduce the size of the basis.
    cutoff = rtol * first(sall)
    if lmax < length(sall)
        cutoff = max(cutoff, sall[lmax])
    end

    # Determine how many singular values survive in each group
    scount = [count(>(cutoff), si) for si in s]

    u_cut = [ui[:, 1:counti] for (ui, counti) in zip(u, scount)]
    s_cut = [si[1:counti] for (si, counti) in zip(s, scount)]
    v_cut = [vi[:, 1:counti] for (vi, counti) in zip(v, scount)]
    return u_cut, s_cut, v_cut
end
