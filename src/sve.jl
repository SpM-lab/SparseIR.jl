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
mutable struct SVEResult{K<:AbstractKernel}
    ptr::Ptr{spir_sve_result}
    kernel::K
    function SVEResult(kernel::K, ε::Real; cutoff::Union{Nothing, Real}=nothing, lmax::Integer=typemax(Int32), n_gauss::Integer=-1, Twork::Union{Nothing, Integer}=nothing) where {K<:AbstractKernel}
        if isnothing(Twork)
          Twork = SPIR_TWORK_FLOAT64X2
        end

        if isnothing(cutoff)
          cutoff = -1
        end

        status = Ref{Int32}(-100)
        sve_result = spir_sve_result_new(kernel.ptr, ε, cutoff, lmax, n_gauss, Twork, status)
        status[] == 0 || error("Failed to create SVEResult")
        result = new{K}(sve_result, kernel)
        finalizer(r -> spir_sve_result_release(r.ptr), result)
        return result
    end
end

#=
function SVEResult(K::AbstractKernel, ε::Real; cutoff=nothing, lmax::Int32=typemax(Int32), n_gauss::Integer=-1, Twork=nothing)
    if isnothing(Twork)
      Twork = SPIR_TWORK_FLOAT64X2
    end

    if isnothing(cutoff)
      cutoff = -1
    end

    SVEResult(K, ε, cutoff, Int32(lmax), Int32(n_gauss), Twork)

    return result
end
=#