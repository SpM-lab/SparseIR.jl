"""
    SVEResult(kernel::AbstractKernel, ε=eps(Float64);
        lmax=typemax(Int32), n_gauss=-1, Twork=SPIR_TWORK_AUTO)

Perform the singular value expansion (SVE) of a kernel, computed by
`libsparseir`.

The SVE of an integral kernel `kernel : [xmin, xmax] x [ymin, ymax] -> ℝ` in
the dimensionless variables `x` and `y` reads

    kernel(x, y) == sum(s[l+1] * u_l(x) * v_l(y) for l in 0, 1, 2, ...),

where `s[l+1]` is the singular value `s_l`, ordered in non-increasing fashion,
the left singular functions `u_l(x)` form an orthonormal system on
`[xmin, xmax]`, and the right singular functions `v_l(y)` form an orthonormal
system on `[ymin, ymax]` (both `[-1, 1]` for the kernels of this package). A
[`FiniteTempBasis`](@ref) built from the result scales them to `U_l(τ)`,
`S_l` and `V_l(ω)`.

The SVE is mapped onto the singular value decomposition (SVD) of a matrix
by expanding the kernel in piecewise Legendre polynomials.

# Arguments

  - `kernel::AbstractKernel`: Integral kernel to take SVE from.

  - `ε::Real`: Accuracy target (positive and finite). It selects the working
    precision (see `Twork`) and the discretization. It does not truncate the
    expansion: the result keeps the singular values down to about twice the
    machine epsilon of the working precision relative to the largest one (for
    example 38 values for `LogisticKernel(80.0)` and `ε = 1e-6`). The
    truncation to `s_l/s_0 ≥ ε` is done by [`FiniteTempBasis`](@ref). Defaults
    to `eps(Float64)` (≈ 2.22e-16).
  - `lmax::Integer`: Maximum number of singular values. Passed to
    `libsparseir`, which currently ignores it.
  - `n_gauss::Integer`: Number of Gauss points of the discretization; `-1`
    lets the library choose. Passed to `libsparseir`, which currently ignores
    it and always chooses the number itself.
  - `Twork::Integer`: Working precision. Available options:

      + `SPIR_TWORK_AUTO` (default): double precision for `ε ≥ 1e-8`,
        extended precision below
      + `SPIR_TWORK_FLOAT64`: Use double precision (64-bit)
      + `SPIR_TWORK_FLOAT64X2`: Use extended precision (128-bit, double-double)

    The constants are available as `SparseIR.SPIR_TWORK_AUTO` etc.

Returns:
An `SVEResult`, whose field `s` holds the singular values `s_l` of the
dimensionless expansion.
"""
mutable struct SVEResult{K<:AbstractKernel}
    ptr::Ptr{spir_sve_result}
    kernel::K
    s::Vector{Float64}
    function SVEResult(
            kernel::K, ε::Real=eps(Float64); lmax::Integer=typemax(Int32),
            n_gauss::Integer=-1, Twork::Integer=SPIR_TWORK_AUTO) where {K<:AbstractKernel}
        isfinite(ε) && ε > 0 ||
            throw(DomainError(ε, "accuracy ε must be positive and finite"))
        if Twork ∉ [SPIR_TWORK_AUTO, SPIR_TWORK_FLOAT64, SPIR_TWORK_FLOAT64X2]
            throw(ArgumentError("invalid Twork value $Twork: use SPIR_TWORK_AUTO, \
                                 SPIR_TWORK_FLOAT64 or SPIR_TWORK_FLOAT64X2"))
        end

        status = Ref{Int32}(-100)
        sve_result = spir_sve_result_new(
            kernel.ptr, ε, lmax, n_gauss, Twork, status)
        _check_status(status[], "spir_sve_result_new")
        _check_handle(sve_result, "spir_sve_result_new")
        size = Ref{Int32}(0)
        _check_status(
            spir_sve_result_get_size(sve_result, size), "spir_sve_result_get_size")
        s = Vector{Float64}(undef, size[])
        _check_status(spir_sve_result_get_svals(sve_result, s), "spir_sve_result_get_svals")
        result = new{K}(sve_result, kernel, s)
        finalizer(r -> spir_sve_result_release(r.ptr), result)
        return result
    end
end
