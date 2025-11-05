"""
    SVEResult(kernel::AbstractKernel;
        Twork=nothing, ε=nothing, n_sv=typemax(Int),
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

  - `ϵ::Real`: Accuracy target for the basis. Determines the relative magnitude for truncation of singular values
    and the accuracy of computed singular values and vectors. Defaults to `eps(Float64)` (≈ 2.22e-16).
  - `n_sv::Integer`: Maximum number of singular values to retain. If given, only at most the `n_sv` most
    significant singular values and associated singular functions are returned. Defaults to `typemax(Int32)` (all singular values meeting the cutoff criterion).
  - `n_gauss::Integer`: Order of Legendre polynomials. Defaults to kernel hinted value.
  - `Twork::Integer`: Working data type. Defaults to `SPIR_TWORK_AUTO` which automatically selects the appropriate precision based on the accuracy requirements.
    Available options:

      + `SPIR_TWORK_AUTO`: Automatically select the best precision (default)
      + `SPIR_TWORK_FLOAT64`: Use double precision (64-bit)
      + `SPIR_TWORK_FLOAT64X2`: Use extended precision (128-bit)
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
    function SVEResult(
            kernel::K, ε::Real=eps(Float64); n_sv::Integer=typemax(Int32),
            n_gauss::Integer=-1, Twork::Integer=SPIR_TWORK_AUTO) where {K<:AbstractKernel}

        # check Twork
        if Twork ∉ [SPIR_TWORK_AUTO, SPIR_TWORK_FLOAT64, SPIR_TWORK_FLOAT64X2]
            error("Invalid Twork value: $Twork")
        end

        status = Ref{Int32}(-100)
        sve_result = spir_sve_result_new(
            _get_ptr(kernel), ε, n_sv, n_gauss, Twork, status)
        if status[] != 0
            error("Failed to create SVEResult: status=$(status[])")
        end
        result = new{K}(sve_result, kernel)
        finalizer(r -> spir_sve_result_release(r.ptr), result)
        return result
    end
end

"""
    s(sve_result::SVEResult)

Get the singular values from an SVEResult.
"""
function s(sve_result::SVEResult)
    size_ref = Ref{Int32}(-1)
    status = spir_sve_result_get_size(sve_result.ptr, size_ref)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get SVE result size")
    svals = Vector{Float64}(undef, Int(size_ref[]))
    status = spir_sve_result_get_svals(sve_result.ptr, svals)
    status == SPIR_COMPUTATION_SUCCESS || error("Failed to get singular values")
    return svals
end

# Allow accessing s as a property for backward compatibility
Base.getproperty(sve_result::SVEResult, name::Symbol) = name === :s ? s(sve_result) : getfield(sve_result, name)
