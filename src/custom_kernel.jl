# Wrapper functions for C callbacks
# These functions receive user_data as a pointer to a Ref containing the kernel object
#
# NOTE: These functions are currently unused because spir_function_kernel_new does not exist
# in the C-API. They are kept for potential future use if the C-API is extended to support
# custom kernel registration. Custom kernels should use SVEResult directly instead.

using MultiFloats

# Enable MultiFloats transcendentals for double-double precision evaluation
# This must be called at runtime, not during precompilation
let _multifloat_initialized = Ref{Bool}(false)
    global function _ensure_multifloat_transcendentals()
        if !_multifloat_initialized[]
            MultiFloats.use_bigfloat_transcendentals()
            _multifloat_initialized[] = true
        end
    end
end

# Currently unused: wrapper for kernel batch evaluation (Float64)
# Would be used with spir_function_kernel_new if it existed in the C-API
function _kernel_batch_wrapper(xs::Ptr{Cdouble}, ys::Ptr{Cdouble}, n::Cint, out::Ptr{Cdouble}, user_data::Ptr{Cvoid})::Cvoid
    try
        kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
        kernel = kernel_ref[]::AbstractKernel
        
        # Create unsafe views of the input arrays
        xs_v = unsafe_wrap(Vector{Float64}, xs, (Int(n),); own=false)
        ys_v = unsafe_wrap(Vector{Float64}, ys, (Int(n),); own=false)
        out_v = unsafe_wrap(Vector{Float64}, out, (Int(n),); own=false)
        
        # Evaluate kernel for all pairs and write to out
        @inbounds for i in 1:Int(n)
            out_v[i] = Float64(kernel(xs_v[i], ys_v[i]))
        end
    catch e
        # If error occurs, fill output with NaN
        out_v = unsafe_wrap(Vector{Float64}, out, (Int(n),); own=false)
        fill!(out_v, NaN)
    end
    return nothing
end

# Currently unused: wrapper for kernel batch evaluation (double-double precision)
# Would be used with spir_function_kernel_new if it existed in the C-API
function _kernel_batch_wrapper_ddouble(xs_hi::Ptr{Cdouble}, xs_lo::Ptr{Cdouble},
                                      ys_hi::Ptr{Cdouble}, ys_lo::Ptr{Cdouble},
                                      n::Cint,
                                      out_hi::Ptr{Cdouble}, out_lo::Ptr{Cdouble},
                                      user_data::Ptr{Cvoid})::Cvoid
    try
        # Ensure MultiFloats transcendentals are enabled (runtime initialization)
        _ensure_multifloat_transcendentals()
        
        kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
        kernel = kernel_ref[]::AbstractKernel
        
        # Create unsafe views of the input arrays
        xs_hi_v = unsafe_wrap(Vector{Float64}, xs_hi, (Int(n),); own=false)
        xs_lo_v = unsafe_wrap(Vector{Float64}, xs_lo, (Int(n),); own=false)
        ys_hi_v = unsafe_wrap(Vector{Float64}, ys_hi, (Int(n),); own=false)
        ys_lo_v = unsafe_wrap(Vector{Float64}, ys_lo, (Int(n),); own=false)
        out_hi_v = unsafe_wrap(Vector{Float64}, out_hi, (Int(n),); own=false)
        out_lo_v = unsafe_wrap(Vector{Float64}, out_lo, (Int(n),); own=false)
        
        # Evaluate kernel for all pairs and write to out_hi/out_lo
        # Use MultiFloats.Float64x2 for double-double precision evaluation
        # Check if kernel supports Float64x2 using the first element
        supports_float64x2 = false
        if n > 0
            try
                x_test = MultiFloat{Float64, 2}((xs_hi_v[1], xs_lo_v[1]))
                y_test = MultiFloat{Float64, 2}((ys_hi_v[1], ys_lo_v[1]))
                result_test = kernel(x_test, y_test)
                # Check if result is Float64x2 or can be converted
                if result_test isa MultiFloat{Float64, 2}
                    supports_float64x2 = true
                else
                    # Try converting to Float64x2
                    Float64x2(result_test)
                    supports_float64x2 = true
                end
            catch
                supports_float64x2 = false
            end
        end
        
        if supports_float64x2
            # Kernel supports Float64x2: use double-double precision evaluation
            for i in 1:Int(n)
                x_dd = MultiFloat{Float64, 2}((xs_hi_v[i], xs_lo_v[i]))
                y_dd = MultiFloat{Float64, 2}((ys_hi_v[i], ys_lo_v[i]))
                result_dd = Float64x2(kernel(x_dd, y_dd))
                result_hi, result_lo = result_dd._limbs
                out_hi_v[i] = result_hi
                out_lo_v[i] = result_lo
            end
        else
            # Kernel doesn't support Float64x2: use Float64 evaluation
            # Reconstruct double-double values as Float64 (x = hi + lo) for better precision
            @warn "Kernel $(typeof(kernel)) does not support Float64x2 (double-double precision). Falling back to Float64 evaluation, which may result in reduced precision."
            for i in 1:Int(n)
                x_f64 = xs_hi_v[i] + xs_lo_v[i]  # Preserve precision by adding hi + lo
                y_f64 = ys_hi_v[i] + ys_lo_v[i]
                result = Float64(kernel(x_f64, y_f64))
                out_hi_v[i] = result
                out_lo_v[i] = 0.0
            end
        end
    catch e
        # If error occurs, fill output with NaN
        out_hi_v = unsafe_wrap(Vector{Float64}, out_hi, (Int(n),); own=false)
        out_lo_v = unsafe_wrap(Vector{Float64}, out_lo, (Int(n),); own=false)
        fill!(out_hi_v, NaN)
        fill!(out_lo_v, NaN)
    end
    return nothing
end

# Currently unused: wrapper for SVE hints segments_x
# Would be used with spir_function_kernel_new if it existed in the C-API
function _segments_x_wrapper(epsilon::Cdouble, segments::Ptr{Cdouble}, n_segments::Ptr{Cint}, user_data::Ptr{Cvoid})::Cvoid
    kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
    kernel = kernel_ref[]::AbstractKernel
    hints = sve_hints(kernel, Float64(epsilon))
    segs = segments_x(hints)
    
    if segments == C_NULL
        # First call: return the number of segments
        unsafe_store!(n_segments, Cint(length(segs)))
    else
        # Second call: fill the segments
        n = length(segs)
        unsafe_store!(n_segments, Cint(n))
        for i in 1:n
            unsafe_store!(segments, Float64(segs[i]), i)
        end
    end
    return nothing
end

# Currently unused: wrapper for SVE hints segments_y
# Would be used with spir_function_kernel_new if it existed in the C-API
function _segments_y_wrapper(epsilon::Cdouble, segments::Ptr{Cdouble}, n_segments::Ptr{Cint}, user_data::Ptr{Cvoid})::Cvoid
    kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
    kernel = kernel_ref[]::AbstractKernel
    hints = sve_hints(kernel, Float64(epsilon))
    segs = segments_y(hints)
    
    if segments == C_NULL
        # First call: return the number of segments
        unsafe_store!(n_segments, Cint(length(segs)))
    else
        # Second call: fill the segments
        n = length(segs)
        unsafe_store!(n_segments, Cint(n))
        for i in 1:n
            unsafe_store!(segments, Float64(segs[i]), i)
        end
    end
    return nothing
end

# Currently unused: wrapper for SVE hints nsvals
# Would be used with spir_function_kernel_new if it existed in the C-API
function _nsvals_wrapper(epsilon::Cdouble, user_data::Ptr{Cvoid})::Cint
    kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
    kernel = kernel_ref[]::AbstractKernel
    hints = sve_hints(kernel, Float64(epsilon))
    return Cint(nsvals(hints))
end

# Currently unused: wrapper for SVE hints ngauss
# Would be used with spir_function_kernel_new if it existed in the C-API
function _ngauss_wrapper(epsilon::Cdouble, user_data::Ptr{Cvoid})::Cint
    kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
    kernel = kernel_ref[]::AbstractKernel
    hints = sve_hints(kernel, Float64(epsilon))
    return Cint(ngauss(hints))
end

# NOTE: _create_custom_kernel and weight_func wrappers were removed because
# spir_function_kernel_new does not exist in the C-API. Custom kernels should use
# SVEResult directly via spir_sve_result_from_matrix instead.
