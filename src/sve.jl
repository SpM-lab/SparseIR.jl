# Import necessary functions and constants
using .C_API: SPIR_COMPUTATION_SUCCESS, SPIR_ORDER_ROW_MAJOR,
              spir_sve_result_from_matrix, spir_sve_result_from_matrix_centrosymmetric,
              spir_sve_result_release, spir_sve_result_get_size, spir_sve_result_get_svals

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

  - `kernel::AbstractKernel`: Integral kernel to take SVE from.

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
mutable struct SVEResult{KernelType<:AbstractKernel}
    ptr::Ptr{spir_sve_result}
    kernel::KernelType
    function SVEResult(
            kernel::KernelType, ε::Real=eps(Float64); n_sv::Integer=typemax(Int32),
            n_gauss::Integer=-1, Twork::Integer=SPIR_TWORK_AUTO) where {KernelType<:AbstractKernel}

        # check Twork
        if Twork ∉ [SPIR_TWORK_AUTO, SPIR_TWORK_FLOAT64, SPIR_TWORK_FLOAT64X2]
            error("Invalid Twork value: $Twork")
        end

        # For default kernels (LogisticKernel, RegularizedBoseKernel), use the direct C-API
        if kernel isa LogisticKernel || kernel isa RegularizedBoseKernel
        status = Ref{Int32}(-100)
            # spir_sve_result_new signature: (k, epsilon, cutoff, lmax, n_gauss, Twork, status)
            # epsilon and cutoff are both set to ε, lmax is set to n_sv
        sve_result = spir_sve_result_new(
                _get_ptr(kernel), Float64(ε), Float64(ε), Cint(n_sv), Cint(n_gauss), Cint(Twork), status)
            if status[] != 0
                error("Failed to create SVEResult: status=$(status[])")
            end
            result = new{KernelType}(sve_result, kernel)
            finalizer(r -> spir_sve_result_release(r.ptr), result)
            return result
        end

        # For custom kernels, create discretized matrix and use spir_sve_result_from_matrix
        # Get SVE hints
        hints = sve_hints(kernel, ε)
        segs_x = segments_x(hints)
        segs_y = segments_y(hints)
        n_gauss_hint = ngauss(hints)
        
        # Use provided n_gauss or hint value
        n_gauss_actual = n_gauss > 0 ? n_gauss : n_gauss_hint
        n_gauss_actual > 0 || error("n_gauss must be positive, got: $n_gauss_actual")
        
        # Get Gauss points from C-API
        gauss_x, _ = _get_gauss_points_from_capi(n_gauss_actual, segs_x)
        gauss_y, _ = _get_gauss_points_from_capi(n_gauss_actual, segs_y)
        
        # Validate segments
        length(segs_x) > 1 || error("segments_x must have at least 2 elements, got: $(length(segs_x))")
        length(segs_y) > 1 || error("segments_y must have at least 2 elements, got: $(length(segs_y))")
        
        # For centrosymmetric kernels, check that segments_y is symmetric
        if iscentrosymmetric(kernel)
            # Check symmetry: segments_y should satisfy segs_y[i] ≈ -segs_y[end-i+1]
            # For symmetric segments, first element should be negative of last, second should be negative of second-to-last, etc.
            n_segs_y = length(segs_y)
            if n_segs_y > 1
                # Check if segments are symmetric around 0
                # For centrosymmetric kernels, segments_y should be symmetric: [-y_max, ..., -y_min, 0, y_min, ..., y_max]
                # or similar patterns
                tolerance = 1e-10 * max(abs(segs_y[1]), abs(segs_y[end]))
                for i in 1:(n_segs_y ÷ 2)
                    j = n_segs_y - i + 1
                    if !isapprox(segs_y[i], -segs_y[j], atol=tolerance)
                        @warn "segments_y for centrosymmetric kernel is not symmetric: " *
                              "segs_y[$i] = $(segs_y[i]), segs_y[$j] = $(segs_y[j]), " *
                              "difference = $(abs(segs_y[i] + segs_y[j]))"
                    end
                end
                # Check if middle element is approximately 0 (if odd number of segments)
                if isodd(n_segs_y)
                    mid_idx = (n_segs_y + 1) ÷ 2
                    if !isapprox(segs_y[mid_idx], 0.0, atol=tolerance)
                        @warn "segments_y for centrosymmetric kernel: middle element should be 0, " *
                              "got segs_y[$mid_idx] = $(segs_y[mid_idx])"
                    end
                end
            end
        end
        
        # Create kernel matrix
        if iscentrosymmetric(kernel)
            # For centrosymmetric kernels, spir_sve_result_from_matrix expects segments in [0, xmax] and [0, ymax]
            # Convert segments from full domain [-xmax, xmax] to reduced domain [0, xmax]
            segs_x_reduced = _extract_reduced_segments(segs_x)
            segs_y_reduced = _extract_reduced_segments(segs_y)
            
            # Recompute Gauss points for reduced segments
            gauss_x_reduced, _ = _get_gauss_points_from_capi(n_gauss_actual, segs_x_reduced)
            gauss_y_reduced, _ = _get_gauss_points_from_capi(n_gauss_actual, segs_y_reduced)
            
            # Use centrosymmetric version
            K_even, K_odd = _matrix_from_gauss_even_odd(kernel, gauss_x_reduced, gauss_y_reduced)
            
            nx = length(gauss_x_reduced)
            ny = length(gauss_y_reduced)
            n_segments_x = length(segs_x_reduced) - 1
            n_segments_y = length(segs_y_reduced) - 1
            
            # Validate matrices are non-empty
            (nx > 0 && ny > 0) || error("Kernel matrices must be non-empty, got nx=$nx, ny=$ny")
            (size(K_even) == (nx, ny)) || error("K_even size mismatch: $(size(K_even)) != ($nx, $ny)")
            (size(K_odd) == (nx, ny)) || error("K_odd size mismatch: $(size(K_odd)) != ($nx, $ny)")
            
            # Validate reduced segments: first must be 0, last must be positive
            (segs_x_reduced[1] == 0.0 && segs_x_reduced[end] > 0.0) ||
                error("segments_x_reduced must start at 0 and end at positive value, got: $(segs_x_reduced)")
            (segs_y_reduced[1] == 0.0 && segs_y_reduced[end] > 0.0) ||
                error("segments_y_reduced must start at 0 and end at positive value, got: $(segs_y_reduced)")
            
            # Convert matrices to row-major format (flatten)
            K_even_flat = vec(K_even)
            K_odd_flat = vec(K_odd)
            
            status = Ref{Int32}(-100)
            sve_result = spir_sve_result_from_matrix_centrosymmetric(
                K_even_flat, C_NULL,  # K_even_high, K_even_low
                K_odd_flat, C_NULL,   # K_odd_high, K_odd_low
                Cint(nx), Cint(ny),
                SPIR_ORDER_ROW_MAJOR,
                segs_x_reduced, Cint(n_segments_x),
                segs_y_reduced, Cint(n_segments_y),
                Cint(n_gauss_actual),
                Float64(ε),
                status
            )
        else
            # Use regular version
            K = matrix_from_gauss(kernel, gauss_x, gauss_y)
            
            nx = length(gauss_x)
            ny = length(gauss_y)
            n_segments_x = length(segs_x) - 1
            n_segments_y = length(segs_y) - 1
            
            # Convert matrix to row-major format (flatten)
            K_flat = vec(K)
            
            status = Ref{Int32}(-100)
            sve_result = spir_sve_result_from_matrix(
                K_flat, C_NULL,  # K_high, K_low
                Cint(nx), Cint(ny),
                SPIR_ORDER_ROW_MAJOR,
                segs_x, Cint(n_segments_x),
                segs_y, Cint(n_segments_y),
                Cint(n_gauss_actual),
                Float64(ε),
                status
            )
        end
        
        if status[] != 0
            error("Failed to create SVEResult from matrix: status=$(status[])")
        end
        if sve_result == C_NULL
            error("Failed to create SVEResult from matrix: null pointer returned")
        end
        
        result = new{KernelType}(sve_result, kernel)
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

"""
    _compute_even_kernel(kernel::AbstractKernel, x, y)

Compute the even part of a centrosymmetric kernel: K_even(x, y) = K(x, y) + K(x, -y).

If the kernel implements `compute_even`, it is used for better numerical accuracy.
Otherwise, falls back to computing K(x, y) + K(x, -y) directly with a warning.
"""
function _compute_even_kernel(kernel::AbstractKernel, x::Real, y::Real)
    try
        # Try to call compute_even if it's implemented
        return compute_even(kernel, Float64(x), Float64(y))
    catch err
        if isa(err, MethodError) || isa(err, UndefVarError)
            @warn "compute_even not implemented for $(typeof(kernel)). Using K(x, y) + K(x, -y) which may have reduced numerical accuracy."
            return kernel(x, y) + kernel(x, -y)
        else
            rethrow(err)
        end
    end
end

"""
    _compute_odd_kernel(kernel::AbstractKernel, x, y)

Compute the odd part of a centrosymmetric kernel: K_odd(x, y) = K(x, y) - K(x, -y).

If the kernel implements `compute_odd`, it is used for better numerical accuracy.
Otherwise, falls back to computing K(x, y) - K(x, -y) directly with a warning.
"""
function _compute_odd_kernel(kernel::AbstractKernel, x::Real, y::Real)
    try
        # Try to call compute_odd if it's implemented
        return compute_odd(kernel, Float64(x), Float64(y))
    catch err
        if isa(err, MethodError) || isa(err, UndefVarError)
            @warn "compute_odd not implemented for $(typeof(kernel)). Using K(x, y) - K(x, -y) which may have reduced numerical accuracy."
            return kernel(x, y) - kernel(x, -y)
        else
            rethrow(err)
        end
    end
end

"""
    _extract_reduced_segments(segs::Vector{Float64})

Extract reduced segments [0, max] from symmetric segments [-max, max].

This function mimics the C++ `symm_segments` function behavior:
1. Extracts the second half of symmetric segments (from middle to end)
2. Ensures the first element is 0 (prepends 0 if needed)

# Arguments
- `segs`: Symmetric segments array (e.g., [-1, -0.5, 0, 0.5, 1])

# Returns
- Reduced segments array starting at 0 (e.g., [0, 0.5, 1])

# Note
For centrosymmetric kernels, spir_sve_result_from_matrix_centrosymmetric expects
segments in [0, xmax] and [0, ymax] range, not the full [-xmax, xmax] range.
"""
function _extract_reduced_segments(segs::Vector{Float64})
    n = length(segs)
    n > 1 || error("segments must have at least 2 elements")
    
    if isodd(n)
        # Odd number of segments: middle element should be 0
        mid = div(n, 2) + 1  # 1-based index of middle element
        segs_reduced = segs[mid:end]
        segs_reduced[1] = 0.0  # Overwrite with exact 0
    else
        # Even number of segments: find the first non-negative element
        zero_idx = nothing
        for i in 1:n
            if segs[i] >= 0.0
                zero_idx = i
                break
            end
        end
        
        if zero_idx === nothing
            error("No non-negative value found in segments, cannot extract reduced segments")
        end
        
        # Extract from zero_idx to end
        segs_reduced = segs[zero_idx:end]
        
        # Ensure the first element is exactly 0
        if abs(segs_reduced[1]) > eps(Float64)
            # If the first element is not 0, prepend 0
            segs_reduced = [0.0; segs_reduced]
        else
            segs_reduced[1] = 0.0  # Ensure exact 0
        end
    end
    
    # Validate that segments are monotonically increasing
    for i in 2:length(segs_reduced)
        segs_reduced[i] > segs_reduced[i-1] ||
            error("Reduced segments must be monotonically increasing, got: $segs_reduced")
    end
    
    return segs_reduced
end

"""
    _matrix_from_gauss_even_odd(kernel::AbstractKernel, gauss_x::Vector{Float64}, gauss_y::Vector{Float64})

Compute even and odd matrices for a centrosymmetric kernel from Gauss points.

# Arguments
- `kernel`: The centrosymmetric kernel to evaluate
- `gauss_x`: Gauss points for x direction (length nx)
- `gauss_y`: Gauss points for y direction (length ny)

# Returns
- `K_even`: Matrix of size (nx, ny) containing K_even(gauss_x[i], gauss_y[j])
- `K_odd`: Matrix of size (nx, ny) containing K_odd(gauss_x[i], gauss_y[j])
"""
function _matrix_from_gauss_even_odd(kernel::AbstractKernel, gauss_x::Vector{Float64}, gauss_y::Vector{Float64})
    nx = length(gauss_x)
    ny = length(gauss_y)
    K_even = Matrix{Float64}(undef, nx, ny)
    K_odd = Matrix{Float64}(undef, nx, ny)
    
    # Check if compute_even/compute_odd are implemented (only once)
    has_compute_even = false
    has_compute_odd = false
    
    try
        compute_even(kernel, 0.0, 0.0)
        has_compute_even = true
    catch err
        if !(isa(err, MethodError) || isa(err, UndefVarError))
            rethrow(err)
        end
    end
    
    try
        compute_odd(kernel, 0.0, 0.0)
        has_compute_odd = true
    catch err
        if !(isa(err, MethodError) || isa(err, UndefVarError))
            rethrow(err)
        end
    end
    
    if !has_compute_even || !has_compute_odd
        @warn "compute_even or compute_odd not implemented for $(typeof(kernel)). Using K(x, y) ± K(x, -y) which may have reduced numerical accuracy."
    end
    
    # Evaluate kernel at all pairs (x[i], y[j])
    for i in 1:nx
        for j in 1:ny
            x_val = gauss_x[i]
            y_val = gauss_y[j]
            
            if has_compute_even
                K_even[i, j] = Float64(compute_even(kernel, x_val, y_val))
            else
                K_even[i, j] = Float64(kernel(x_val, y_val) + kernel(x_val, -y_val))
            end
            
            if has_compute_odd
                K_odd[i, j] = Float64(compute_odd(kernel, x_val, y_val))
            else
                K_odd[i, j] = Float64(kernel(x_val, y_val) - kernel(x_val, -y_val))
            end
        end
    end
    
    return K_even, K_odd
end
