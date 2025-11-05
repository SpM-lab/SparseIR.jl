# Wrapper functions for C callbacks
# These functions receive user_data as a pointer to a Ref containing the kernel object

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

function _kernel_batch_wrapper(xs::Ptr{Cdouble}, ys::Ptr{Cdouble}, n::Cint, out::Ptr{Cdouble}, user_data::Ptr{Cvoid})::Cvoid
    try
        # Write to file to avoid buffering issues
        open("/tmp/julia_kernel_debug.log", "a") do f
            println(f, "[DEBUG Julia] _kernel_batch_wrapper called with n=$n")
            flush(f)
            println(f, "[DEBUG Julia] Step 1")
            flush(f)
            println(f, "[DEBUG Julia] Step 2")
            flush(f)
            println(f, "[DEBUG Julia] Step 3")
            flush(f)
            println(f, "[DEBUG Julia] user_data pointer: $user_data")
            flush(f)
        end
        println("[DEBUG Julia] _kernel_batch_wrapper called with n=$n")
        flush(stdout)
        println("[DEBUG Julia] user_data pointer: $user_data")
        flush(stdout)
        open("/tmp/julia_kernel_debug.log", "a") do f
            println(f, "[DEBUG Julia] About to call unsafe_pointer_to_objref...")
            flush(f)
        end
        println("[DEBUG Julia] About to call unsafe_pointer_to_objref...")
        flush(stdout)
        open("/tmp/julia_kernel_debug.log", "a") do f
            println(f, "[DEBUG Julia] Calling unsafe_pointer_to_objref now...")
            flush(f)
        end
        kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
        open("/tmp/julia_kernel_debug.log", "a") do f
            println(f, "[DEBUG Julia] kernel_ref created: $kernel_ref")
            flush(f)
        end
        println("[DEBUG Julia] kernel_ref created: $kernel_ref")
        flush(stdout)
        println("[DEBUG Julia] About to access kernel_ref[]...")
        flush(stdout)
        kernel = kernel_ref[]::AbstractKernel
        println("[DEBUG Julia] Kernel type: $(typeof(kernel))")
        flush(stdout)
        
        # Create unsafe views of the input arrays
        xs_v = unsafe_wrap(Vector{Float64}, xs, (Int(n),); own=false)
        ys_v = unsafe_wrap(Vector{Float64}, ys, (Int(n),); own=false)
        out_v = unsafe_wrap(Vector{Float64}, out, (Int(n),); own=false)
        
        # Evaluate kernel for all pairs and write to out
        println("[DEBUG Julia] Starting kernel evaluation loop, n=$n")
        flush(stdout)
        @inbounds for i in 1:Int(n)
            out_v[i] = Float64(kernel(xs_v[i], ys_v[i]))
        end
        println("[DEBUG Julia] Kernel evaluation completed")
        flush(stdout)
    catch e
        # If error occurs, fill output with NaN
        out_v = unsafe_wrap(Vector{Float64}, out, (Int(n),); own=false)
        fill!(out_v, NaN)
    end
    return nothing
end

function _kernel_batch_wrapper_ddouble(xs_hi::Ptr{Cdouble}, xs_lo::Ptr{Cdouble},
                                      ys_hi::Ptr{Cdouble}, ys_lo::Ptr{Cdouble},
                                      n::Cint,
                                      out_hi::Ptr{Cdouble}, out_lo::Ptr{Cdouble},
                                      user_data::Ptr{Cvoid})::Cvoid
    try
        println("[DEBUG Julia] _kernel_batch_wrapper_ddouble called with n=$n")
        flush(stdout)
        println("[DEBUG Julia] user_data pointer: $user_data")
        flush(stdout)
        # Ensure MultiFloats transcendentals are enabled (runtime initialization)
        _ensure_multifloat_transcendentals()
        
        println("[DEBUG Julia] About to call unsafe_pointer_to_objref...")
        flush(stdout)
        kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
        println("[DEBUG Julia] kernel_ref created: $kernel_ref")
        flush(stdout)
        println("[DEBUG Julia] About to access kernel_ref[]...")
        flush(stdout)
        kernel = kernel_ref[]::AbstractKernel
        println("[DEBUG Julia] Kernel type: $(typeof(kernel))")
        flush(stdout)
        
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
            println("[DEBUG Julia] Using Float64x2 evaluation, n=$n")
            flush(stdout)
            for i in 1:Int(n)
                x_dd = MultiFloat{Float64, 2}((xs_hi_v[i], xs_lo_v[i]))
                y_dd = MultiFloat{Float64, 2}((ys_hi_v[i], ys_lo_v[i]))
                result_dd = Float64x2(kernel(x_dd, y_dd))
                result_hi, result_lo = result_dd._limbs
                out_hi_v[i] = result_hi
                out_lo_v[i] = result_lo
            end
            println("[DEBUG Julia] Float64x2 evaluation completed")
            flush(stdout)
        else
            # Kernel doesn't support Float64x2: use Float64 evaluation
            # Reconstruct double-double values as Float64 (x = hi + lo) for better precision
            @warn "Kernel $(typeof(kernel)) does not support Float64x2 (double-double precision). Falling back to Float64 evaluation, which may result in reduced precision."
            println("[DEBUG Julia] Using Float64 evaluation, n=$n")
            flush(stdout)
            for i in 1:Int(n)
                x_f64 = xs_hi_v[i] + xs_lo_v[i]  # Preserve precision by adding hi + lo
                y_f64 = ys_hi_v[i] + ys_lo_v[i]
                result = Float64(kernel(x_f64, y_f64))
                out_hi_v[i] = result
                out_lo_v[i] = 0.0
            end
            println("[DEBUG Julia] Float64 evaluation completed")
            flush(stdout)
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

function _nsvals_wrapper(epsilon::Cdouble, user_data::Ptr{Cvoid})::Cint
    kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
    kernel = kernel_ref[]::AbstractKernel
    hints = sve_hints(kernel, Float64(epsilon))
    return Cint(nsvals(hints))
end

function _ngauss_wrapper(epsilon::Cdouble, user_data::Ptr{Cvoid})::Cint
    kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
    kernel = kernel_ref[]::AbstractKernel
    hints = sve_hints(kernel, Float64(epsilon))
    return Cint(ngauss(hints))
end

function _weight_func_fermionic_wrapper(beta::Cdouble, omega::Cdouble, user_data::Ptr{Cvoid})::Cdouble
    kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
    kernel = kernel_ref[]::AbstractKernel
    wfunc = weight_func(kernel, Fermionic())
    # weight_func returns a function that accepts an array y = βω/Λ
    # We need to compute y = beta * omega / Λ and pass it as an array
    lambda = Λ(kernel)
    y = Float64(beta * omega / lambda)
    y_arr = [y]
    result = wfunc(y_arr)
    return Float64(result[1])
end

function _weight_func_bosonic_wrapper(beta::Cdouble, omega::Cdouble, user_data::Ptr{Cvoid})::Cdouble
    kernel_ref = Base.unsafe_pointer_to_objref(user_data)::Ref{Any}
    kernel = kernel_ref[]::AbstractKernel
    wfunc = weight_func(kernel, Bosonic())
    # weight_func returns a function that accepts an array y = βω/Λ
    # We need to compute y = beta * omega / Λ and pass it as an array
    lambda = Λ(kernel)
    y = Float64(beta * omega / lambda)
    y_arr = [y]
    result = wfunc(y_arr)
    return Float64(result[1])
end

function _create_custom_kernel(kernel::AbstractKernel)
    # Get kernel properties
    lambda = Λ(kernel)
    xmin, xmax = xrange(kernel)
    ymin, ymax = yrange(kernel)
    is_centrosym = iscentrosymmetric(kernel)
    
    # Create a Ref to store the kernel pointer for user_data
    # This allows the callback functions to access the kernel object
    # Store as Any to allow any kernel subtype, then convert to pointer
    kernel_ref = Ref{Any}(kernel)
    user_data = Base.pointer_from_objref(kernel_ref)
    
    # Create batch function pointers using @cfunction
    batch_func = @cfunction(_kernel_batch_wrapper, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cvoid}))
    batch_func_dd = @cfunction(_kernel_batch_wrapper_ddouble, Cvoid, 
                                (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, 
                                 Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
    
    # SVE hints functions
    segments_x_func = @cfunction(_segments_x_wrapper, Cvoid, (Cdouble, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}))
    segments_y_func = @cfunction(_segments_y_wrapper, Cvoid, (Cdouble, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid}))
    nsvals_func = @cfunction(_nsvals_wrapper, Cint, (Cdouble, Ptr{Cvoid}))
    ngauss_func = @cfunction(_ngauss_wrapper, Cint, (Cdouble, Ptr{Cvoid}))
    
    # Weight functions
    weight_func_fermionic = @cfunction(_weight_func_fermionic_wrapper, Cdouble, (Cdouble, Cdouble, Ptr{Cvoid}))
    weight_func_bosonic = @cfunction(_weight_func_bosonic_wrapper, Cdouble, (Cdouble, Cdouble, Ptr{Cvoid}))
    
    # Call spir_function_kernel_new
    status = Ref{Cint}(-100)
    ptr = spir_function_kernel_new(
        Float64(lambda),
        batch_func,
        batch_func_dd,
        Float64(xmin), Float64(xmax),
        Float64(ymin), Float64(ymax),
        is_centrosym ? 1 : 0,
        segments_x_func,
        segments_y_func,
        nsvals_func,
        ngauss_func,
        weight_func_fermionic,
        weight_func_bosonic,
        user_data,
        status
    )
    
    status[] == SPIR_COMPUTATION_SUCCESS || error("Failed to create custom kernel: status=$(status[])")
    ptr != C_NULL || error("Failed to create custom kernel: null pointer returned")
    
    return ptr
end
