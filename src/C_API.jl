module C_API

using CEnum: CEnum, @cenum

using Libdl: Libdl
using libsparseir_jll: libsparseir_jll

function get_libsparseir()
    deps_dir = joinpath(dirname(@__DIR__), "deps")
    local_libsparseir_path = joinpath(deps_dir, "libsparse_ir_capi.$(Libdl.dlext)")
    if isfile(local_libsparseir_path)
        return local_libsparseir_path
    else
        return libsparseir_jll.libsparseir
    end
end

const libsparseir = get_libsparseir()


"""
    spir_basis

Opaque basis type for C API (compatible with libsparseir)

Represents a finite temperature basis (IR or DLR).

Note: Named [`spir_basis`](@ref) to match libsparseir C++ API exactly. The internal structure is hidden using a void pointer to prevent exposing BasisType to C.
"""
struct spir_basis
    _private::Ptr{Cvoid}
end

"""
    spir_kernel

Opaque kernel type for C API (compatible with libsparseir)

This is a tagged union that can hold either LogisticKernel or RegularizedBoseKernel. The actual type is determined by which constructor was used.

Note: Named [`spir_kernel`](@ref) to match libsparseir C++ API exactly. The internal structure is hidden using a void pointer to prevent exposing KernelType to C.
"""
struct spir_kernel
    _private::Ptr{Cvoid}
end

"""
    spir_sve_result

Opaque SVE result type for C API (compatible with libsparseir)

Contains singular values and singular functions from SVE computation.

Note: Named [`spir_sve_result`](@ref) to match libsparseir C++ API exactly. The internal structure is hidden using a void pointer to prevent exposing Arc<SVEResult> to C.
"""
struct spir_sve_result
    _private::Ptr{Cvoid}
end

"""
Error codes for C API (compatible with libsparseir)
"""
const StatusCode = Cint

"""
    spir_funcs

Opaque funcs type for C API (compatible with libsparseir)

Wraps piecewise Legendre polynomial representations: - PiecewiseLegendrePolyVector for u and v - PiecewiseLegendreFTVector for uhat

Note: Named [`spir_funcs`](@ref) to match libsparseir C++ API exactly. The internal FuncsType is hidden using a void pointer, but beta is kept as a public field.
"""
struct spir_funcs
    _private::Ptr{Cvoid}
    beta::Cdouble
end

"""
    spir_gemm_backend

Opaque pointer type for GEMM backend handle

This type wraps a `GemmBackendHandle` and provides a C-compatible interface. The handle can be created, cloned, and passed to evaluate/fit functions.

Note: The internal structure is hidden using a void pointer to prevent exposing GemmBackendHandle to C.
"""
struct spir_gemm_backend
    _private::Ptr{Cvoid}
end

"""
    Complex64

Complex number type for C API (compatible with C's double complex)

This type is compatible with C99's `double complex` and C++'s `std::complex<double>`. Layout: `{double re; double im;}` with standard alignment.
"""
struct Complex64
    re::Cdouble
    im::Cdouble
end

"""
    spir_sampling

Sampling type for C API (unified type for all domains)

This wraps different sampling implementations: - TauSampling (for tau-domain) - MatsubaraSampling (for Matsubara frequencies, full range or positive-only) The internal structure is hidden using a void pointer to prevent exposing SamplingType to C.
"""
struct spir_sampling
    _private::Ptr{Cvoid}
end

"""
    spir_basis_release(basis)

Manual release function (replaces macro-generated one)
"""
function spir_basis_release(basis)
    ccall((:spir_basis_release, libsparseir), Cvoid, (Ptr{spir_basis},), basis)
end

"""
    spir_basis_clone(src)

Manual clone function (replaces macro-generated one)
"""
function spir_basis_clone(src)
    ccall((:spir_basis_clone, libsparseir), Ptr{spir_basis}, (Ptr{spir_basis},), src)
end

"""
    spir_basis_is_assigned(obj)

Manual is\\_assigned function (replaces macro-generated one)
"""
function spir_basis_is_assigned(obj)
    ccall((:spir_basis_is_assigned, libsparseir), Int32, (Ptr{spir_basis},), obj)
end

"""
    spir_basis_new(statistics, beta, omega_max, epsilon, k, sve, max_size, status)

Create a finite temperature basis (libsparseir compatible)

# Arguments * `statistics` - 0 for Bosonic, 1 for Fermionic * `beta` - Inverse temperature (must be > 0) * `omega_max` - Frequency cutoff (must be > 0) * `epsilon` - Accuracy target (must be > 0) * `k` - Kernel object (can be NULL if sve is provided) * `sve` - Pre-computed SVE result (can be NULL, will compute if needed) * `max_size` - Maximum basis size (-1 for no limit) * `status` - Pointer to store status code

# Returns * Pointer to basis object, or NULL on failure

# Safety The caller must ensure `status` is a valid pointer.
"""
function spir_basis_new(statistics, beta, omega_max, epsilon, k, sve, max_size, status)
    ccall((:spir_basis_new, libsparseir), Ptr{spir_basis}, (Cint, Cdouble, Cdouble, Cdouble, Ptr{spir_kernel}, Ptr{spir_sve_result}, Cint, Ptr{StatusCode}), statistics, beta, omega_max, epsilon, k, sve, max_size, status)
end

"""
    spir_basis_new_from_sve_and_regularizer(statistics, beta, omega_max, epsilon, lambda, _ypower, _conv_radius, sve, regularizer_funcs, max_size, status)

Create a finite temperature basis from SVE result and custom regularizer function

This function creates a basis from a pre-computed SVE result and a custom regularizer function. The regularizer function is used to scale the basis functions in the frequency domain.

# Arguments * `statistics` - 0 for Bosonic, 1 for Fermionic * `beta` - Inverse temperature (must be > 0) * `omega_max` - Frequency cutoff (must be > 0) * `epsilon` - Accuracy target (must be > 0) * `lambda` - Kernel parameter Λ = β * ωmax (must be > 0) * `ypower` - Power of y in kernel (typically 0 or 1) * `conv_radius` - Convergence radius for Fourier transform * `sve` - Pre-computed SVE result (must not be NULL) * `regularizer_funcs` - Custom regularizer function (must not be NULL) * `max_size` - Maximum basis size (-1 for no limit) * `status` - Pointer to store status code

# Returns * Pointer to basis object, or NULL on failure

# Note Currently, the regularizer function is evaluated but the custom weight is not fully integrated into the basis construction. The basis is created using the standard from\\_sve\\_result method with the kernel's default regularizer. This is a limitation of the current Rust implementation compared to the C++ version.

# Safety The caller must ensure `status` is a valid pointer.
"""
function spir_basis_new_from_sve_and_regularizer(statistics, beta, omega_max, epsilon, lambda, _ypower, _conv_radius, sve, regularizer_funcs, max_size, status)
    ccall((:spir_basis_new_from_sve_and_regularizer, libsparseir), Ptr{spir_basis}, (Cint, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cdouble, Ptr{spir_sve_result}, Ptr{spir_funcs}, Cint, Ptr{StatusCode}), statistics, beta, omega_max, epsilon, lambda, _ypower, _conv_radius, sve, regularizer_funcs, max_size, status)
end

"""
    spir_basis_get_size(b, size)

Get the number of basis functions

# Arguments * `b` - Basis object * `size` - Pointer to store the size

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or size is null * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_basis_get_size(b, size)
    ccall((:spir_basis_get_size, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cint}), b, size)
end

"""
    spir_basis_get_svals(b, svals)

Get singular values from a basis

# Arguments * `b` - Basis object * `svals` - Pre-allocated array to store singular values (size must be >= basis size)

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or svals is null * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_basis_get_svals(b, svals)
    ccall((:spir_basis_get_svals, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cdouble}), b, svals)
end

"""
    spir_basis_get_stats(b, statistics)

Get statistics type (Fermionic or Bosonic) of a basis

# Arguments * `b` - Basis object * `statistics` - Pointer to store statistics (0 = Bosonic, 1 = Fermionic)

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or statistics is null * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_basis_get_stats(b, statistics)
    ccall((:spir_basis_get_stats, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cint}), b, statistics)
end

"""
    spir_basis_get_singular_values(b, svals)

Get singular values (alias for [`spir_basis_get_svals`](@ref) for libsparseir compatibility)
"""
function spir_basis_get_singular_values(b, svals)
    ccall((:spir_basis_get_singular_values, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cdouble}), b, svals)
end

"""
    spir_basis_get_n_default_taus(b, num_points)

Get the number of default tau sampling points

# Arguments * `b` - Basis object * `num_points` - Pointer to store the number of points

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or num\\_points is null * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_basis_get_n_default_taus(b, num_points)
    ccall((:spir_basis_get_n_default_taus, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cint}), b, num_points)
end

"""
    spir_basis_get_default_taus(b, points)

Get default tau sampling points

# Arguments * `b` - Basis object * `points` - Pre-allocated array to store tau points

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or points is null * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_basis_get_default_taus(b, points)
    ccall((:spir_basis_get_default_taus, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cdouble}), b, points)
end

"""
    spir_basis_get_n_default_matsus(b, positive_only, num_points)

Get the number of default Matsubara sampling points

# Arguments * `b` - Basis object * `positive_only` - If true, return only positive frequencies * `num_points` - Pointer to store the number of points

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or num\\_points is null * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_basis_get_n_default_matsus(b, positive_only, num_points)
    ccall((:spir_basis_get_n_default_matsus, libsparseir), StatusCode, (Ptr{spir_basis}, Bool, Ptr{Cint}), b, positive_only, num_points)
end

"""
    spir_basis_get_default_matsus(b, positive_only, points)

Get default Matsubara sampling points

# Arguments * `b` - Basis object * `positive_only` - If true, return only positive frequencies * `points` - Pre-allocated array to store Matsubara indices

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or points is null * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_basis_get_default_matsus(b, positive_only, points)
    ccall((:spir_basis_get_default_matsus, libsparseir), StatusCode, (Ptr{spir_basis}, Bool, Ptr{Int64}), b, positive_only, points)
end

"""
    spir_basis_get_u(b, status)

Gets the basis functions in imaginary time (τ) domain

# Arguments * `b` - Pointer to the finite temperature basis object * `status` - Pointer to store the status code

# Returns Pointer to the basis functions object ([`spir_funcs`](@ref)), or NULL if creation fails

# Safety The caller must ensure that `b` is a valid pointer, and must call `[`spir_funcs_release`](@ref)()` on the returned pointer when done.
"""
function spir_basis_get_u(b, status)
    ccall((:spir_basis_get_u, libsparseir), Ptr{spir_funcs}, (Ptr{spir_basis}, Ptr{StatusCode}), b, status)
end

"""
    spir_basis_get_v(b, status)

Gets the basis functions in real frequency (ω) domain

# Arguments * `b` - Pointer to the finite temperature basis object * `status` - Pointer to store the status code

# Returns Pointer to the basis functions object ([`spir_funcs`](@ref)), or NULL if creation fails

# Safety The caller must ensure that `b` is a valid pointer, and must call `[`spir_funcs_release`](@ref)()` on the returned pointer when done.
"""
function spir_basis_get_v(b, status)
    ccall((:spir_basis_get_v, libsparseir), Ptr{spir_funcs}, (Ptr{spir_basis}, Ptr{StatusCode}), b, status)
end

"""
    spir_basis_get_n_default_ws(b, num_points)

Gets the number of default omega (real frequency) sampling points

# Arguments * `b` - Pointer to the finite temperature basis object * `num_points` - Pointer to store the number of sampling points

# Returns Status code ([`SPIR_COMPUTATION_SUCCESS`](@ref) on success)

# Safety The caller must ensure that `b` and `num_points` are valid pointers
"""
function spir_basis_get_n_default_ws(b, num_points)
    ccall((:spir_basis_get_n_default_ws, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cint}), b, num_points)
end

"""
    spir_basis_get_default_ws(b, points)

Gets the default omega (real frequency) sampling points

# Arguments * `b` - Pointer to the finite temperature basis object * `points` - Pre-allocated array to store the omega sampling points

# Returns Status code ([`SPIR_COMPUTATION_SUCCESS`](@ref) on success)

# Safety The caller must ensure that `points` has size >= `[`spir_basis_get_n_default_ws`](@ref)(b)`
"""
function spir_basis_get_default_ws(b, points)
    ccall((:spir_basis_get_default_ws, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cdouble}), b, points)
end

"""
    spir_basis_get_uhat(b, status)

Gets the basis functions in Matsubara frequency domain

# Arguments * `b` - Pointer to the finite temperature basis object * `status` - Pointer to store the status code

# Returns Pointer to the basis functions object ([`spir_funcs`](@ref)), or NULL if creation fails

# Safety The caller must ensure that `b` is a valid pointer, and must call `[`spir_funcs_release`](@ref)()` on the returned pointer when done.
"""
function spir_basis_get_uhat(b, status)
    ccall((:spir_basis_get_uhat, libsparseir), Ptr{spir_funcs}, (Ptr{spir_basis}, Ptr{StatusCode}), b, status)
end

"""
    spir_basis_get_uhat_full(b, status)

Gets the full (untruncated) Matsubara-frequency basis functions

This function returns an object representing all basis functions in the Matsubara-frequency domain, including those beyond the truncation threshold. Unlike [`spir_basis_get_uhat`](@ref), which returns only the truncated basis functions (up to `basis.size()`), this function returns all basis functions from the SVE result (up to `sve\\_result.s.size()`).

# Arguments * `b` - Pointer to the finite temperature basis object (must be an IR basis) * `status` - Pointer to store the status code

# Returns Pointer to the basis functions object, or NULL if creation fails

# Note The returned object must be freed using [`spir_funcs_release`](@ref) when no longer needed This function is only available for IR basis objects (not DLR) uhat\\_full.size() >= uhat.size() is always true The first uhat.size() functions in uhat\\_full are identical to uhat

# Safety The caller must ensure that `b` is a valid pointer, and must call `[`spir_funcs_release`](@ref)()` on the returned pointer when done.
"""
function spir_basis_get_uhat_full(b, status)
    ccall((:spir_basis_get_uhat_full, libsparseir), Ptr{spir_funcs}, (Ptr{spir_basis}, Ptr{StatusCode}), b, status)
end

"""
    spir_basis_get_default_taus_ext(b, n_points, points, n_points_returned)

Get default tau sampling points with custom limit (extended version)

# Arguments * `b` - Basis object * `n_points` - Maximum number of points requested * `points` - Pre-allocated array to store tau points (size >= n\\_points) * `n_points_returned` - Pointer to store actual number of points returned

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if any pointer is null or n\\_points < 0 * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs

# Note Returns min(n\\_points, actual\\_default\\_points) sampling points
"""
function spir_basis_get_default_taus_ext(b, n_points, points, n_points_returned)
    ccall((:spir_basis_get_default_taus_ext, libsparseir), StatusCode, (Ptr{spir_basis}, Cint, Ptr{Cdouble}, Ptr{Cint}), b, n_points, points, n_points_returned)
end

"""
    spir_basis_get_n_default_matsus_ext(b, positive_only, mitigate, L, num_points_returned)

Get number of default Matsubara sampling points with custom limit (extended version)

# Arguments * `b` - Basis object * `positive_only` - If true, return only positive frequencies * `mitigate` - If true, enable mitigation (fencing) to improve conditioning * `L` - Requested number of sampling points * `num_points_returned` - Pointer to store actual number of points

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if any pointer is null or L < 0 * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs

# Note Returns the actual number of points that will be returned by [`spir_basis_get_default_matsus_ext`](@ref) with the same parameters. When mitigate is true, may return more points than requested due to fencing.
"""
function spir_basis_get_n_default_matsus_ext(b, positive_only, mitigate, L, num_points_returned)
    ccall((:spir_basis_get_n_default_matsus_ext, libsparseir), StatusCode, (Ptr{spir_basis}, Bool, Bool, Cint, Ptr{Cint}), b, positive_only, mitigate, L, num_points_returned)
end

"""
    spir_basis_get_default_matsus_ext(b, positive_only, mitigate, n_points, points, n_points_returned)

Get default Matsubara sampling points with custom limit (extended version)

# Arguments * `b` - Basis object * `positive_only` - If true, return only positive frequencies * `mitigate` - If true, enable mitigation (fencing) to improve conditioning * `n_points` - Maximum number of points requested * `points` - Pre-allocated array to store Matsubara indices (size >= n\\_points) * `n_points_returned` - Pointer to store actual number of points returned

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if any pointer is null or n\\_points < 0 * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs

# Note Returns the actual number of sampling points (may be more than n\\_points when mitigate is true due to fencing). The caller should call [`spir_basis_get_n_default_matsus_ext`](@ref) with the same parameters first to determine the required buffer size.
"""
function spir_basis_get_default_matsus_ext(b, positive_only, mitigate, n_points, points, n_points_returned)
    ccall((:spir_basis_get_default_matsus_ext, libsparseir), StatusCode, (Ptr{spir_basis}, Bool, Bool, Cint, Ptr{Int64}, Ptr{Cint}), b, positive_only, mitigate, n_points, points, n_points_returned)
end

"""
    spir_dlr_new(b, status)

Creates a new DLR from an IR basis with default poles

# Arguments * `b` - Pointer to a finite temperature basis object * `status` - Pointer to store the status code

# Returns Pointer to the newly created DLR basis object, or NULL if creation fails

# Safety Caller must ensure `b` is a valid IR basis pointer
"""
function spir_dlr_new(b, status)
    ccall((:spir_dlr_new, libsparseir), Ptr{spir_basis}, (Ptr{spir_basis}, Ptr{StatusCode}), b, status)
end

"""
    spir_dlr_new_with_poles(b, npoles, poles, status)

Creates a new DLR with custom poles

# Arguments * `b` - Pointer to a finite temperature basis object * `npoles` - Number of poles to use * `poles` - Array of pole locations on the real-frequency axis * `status` - Pointer to store the status code

# Returns Pointer to the newly created DLR basis object, or NULL if creation fails

# Safety Caller must ensure `b` is valid and `poles` has `npoles` elements
"""
function spir_dlr_new_with_poles(b, npoles, poles, status)
    ccall((:spir_dlr_new_with_poles, libsparseir), Ptr{spir_basis}, (Ptr{spir_basis}, Cint, Ptr{Cdouble}, Ptr{StatusCode}), b, npoles, poles, status)
end

"""
    spir_dlr_get_npoles(dlr, num_poles)

Gets the number of poles in a DLR

# Arguments * `dlr` - Pointer to a DLR basis object * `num_poles` - Pointer to store the number of poles

# Returns Status code

# Safety Caller must ensure `dlr` is a valid DLR basis pointer
"""
function spir_dlr_get_npoles(dlr, num_poles)
    ccall((:spir_dlr_get_npoles, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cint}), dlr, num_poles)
end

"""
    spir_dlr_get_poles(dlr, poles)

Gets the pole locations in a DLR

# Arguments * `dlr` - Pointer to a DLR basis object * `poles` - Pre-allocated array to store pole locations

# Returns Status code

# Safety Caller must ensure `dlr` is valid and `poles` has sufficient size
"""
function spir_dlr_get_poles(dlr, poles)
    ccall((:spir_dlr_get_poles, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cdouble}), dlr, poles)
end

"""
    spir_ir2dlr_dd(dlr, backend, order, ndim, input_dims, target_dim, input, out)

Convert IR coefficients to DLR (real-valued)

# Arguments * `dlr` - Pointer to a DLR basis object * `order` - Memory layout order * `ndim` - Number of dimensions * `input_dims` - Array of input dimensions * `target_dim` - Dimension to transform * `input` - IR coefficients * `out` - Output DLR coefficients

# Returns Status code

# Safety Caller must ensure pointers are valid and arrays have correct sizes
"""
function spir_ir2dlr_dd(dlr, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_ir2dlr_dd, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), dlr, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_ir2dlr_zz(dlr, backend, order, ndim, input_dims, target_dim, input, out)

Convert IR coefficients to DLR (complex-valued)

# Arguments * `dlr` - Pointer to a DLR basis object * `order` - Memory layout order * `ndim` - Number of dimensions * `input_dims` - Array of input dimensions * `target_dim` - Dimension to transform * `input` - Complex IR coefficients * `out` - Output complex DLR coefficients

# Returns Status code

# Safety Caller must ensure pointers are valid and arrays have correct sizes
"""
function spir_ir2dlr_zz(dlr, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_ir2dlr_zz, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Complex64}, Ptr{Complex64}), dlr, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_dlr2ir_dd(dlr, backend, order, ndim, input_dims, target_dim, input, out)

Convert DLR coefficients to IR (real-valued)

# Arguments * `dlr` - Pointer to a DLR basis object * `order` - Memory layout order * `ndim` - Number of dimensions * `input_dims` - Array of input dimensions * `target_dim` - Dimension to transform * `input` - DLR coefficients * `out` - Output IR coefficients

# Returns Status code

# Safety Caller must ensure pointers are valid and arrays have correct sizes
"""
function spir_dlr2ir_dd(dlr, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_dlr2ir_dd, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), dlr, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_dlr2ir_zz(dlr, backend, order, ndim, input_dims, target_dim, input, out)

Convert DLR coefficients to IR (complex-valued)

# Arguments * `dlr` - Pointer to a DLR basis object * `order` - Memory layout order * `ndim` - Number of dimensions * `input_dims` - Array of input dimensions * `target_dim` - Dimension to transform * `input` - Complex DLR coefficients * `out` - Output complex IR coefficients

# Returns Status code

# Safety Caller must ensure pointers are valid and arrays have correct sizes
"""
function spir_dlr2ir_zz(dlr, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_dlr2ir_zz, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Complex64}, Ptr{Complex64}), dlr, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_funcs_release(funcs)

Manual release function (replaces macro-generated one)
"""
function spir_funcs_release(funcs)
    ccall((:spir_funcs_release, libsparseir), Cvoid, (Ptr{spir_funcs},), funcs)
end

"""
    spir_funcs_clone(src)

Manual clone function (replaces macro-generated one)
"""
function spir_funcs_clone(src)
    ccall((:spir_funcs_clone, libsparseir), Ptr{spir_funcs}, (Ptr{spir_funcs},), src)
end

"""
    spir_funcs_is_assigned(obj)

Manual is\\_assigned function (replaces macro-generated one)
"""
function spir_funcs_is_assigned(obj)
    ccall((:spir_funcs_is_assigned, libsparseir), Int32, (Ptr{spir_funcs},), obj)
end

"""
    spir_funcs_from_piecewise_legendre(segments, n_segments, coeffs, nfuncs, _order, status)

Create a [`spir_funcs`](@ref) object from piecewise Legendre polynomial coefficients

Constructs a continuous function object from segments and Legendre polynomial expansion coefficients. The coefficients are organized per segment, with each segment containing nfuncs coefficients (degrees 0 to nfuncs-1).

# Arguments * `segments` - Array of segment boundaries (n\\_segments+1 elements). Must be monotonically increasing. * `n_segments` - Number of segments (must be >= 1) * `coeffs` - Array of Legendre coefficients. Layout: contiguous per segment, coefficients for segment i are stored at indices [i*nfuncs, (i+1)*nfuncs). Each segment has nfuncs coefficients for Legendre degrees 0 to nfuncs-1. * `nfuncs` - Number of basis functions per segment (Legendre polynomial degrees 0 to nfuncs-1) * `order` - Order parameter (currently unused, reserved for future use) * `status` - Pointer to store the status code

# Returns Pointer to the newly created funcs object, or NULL if creation fails

# Note The function creates a single piecewise Legendre polynomial function. To create multiple functions, call this function multiple times.
"""
function spir_funcs_from_piecewise_legendre(segments, n_segments, coeffs, nfuncs, _order, status)
    ccall((:spir_funcs_from_piecewise_legendre, libsparseir), Ptr{spir_funcs}, (Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Ptr{StatusCode}), segments, n_segments, coeffs, nfuncs, _order, status)
end

"""
    spir_funcs_get_slice(funcs, nslice, indices, status)

Extract a subset of functions by indices

# Arguments * `funcs` - Pointer to the source funcs object * `nslice` - Number of functions to select (length of indices array) * `indices` - Array of indices specifying which functions to include * `status` - Pointer to store the status code

# Returns Pointer to a new funcs object containing only the selected functions, or null on error

# Safety The caller must ensure that `funcs` and `indices` are valid pointers. The returned pointer must be freed with `[`spir_funcs_release`](@ref)()`.
"""
function spir_funcs_get_slice(funcs, nslice, indices, status)
    ccall((:spir_funcs_get_slice, libsparseir), Ptr{spir_funcs}, (Ptr{spir_funcs}, Int32, Ptr{Int32}, Ptr{StatusCode}), funcs, nslice, indices, status)
end

"""
    spir_funcs_get_size(funcs, size)

Gets the number of basis functions

# Arguments * `funcs` - Pointer to the funcs object * `size` - Pointer to store the number of functions

# Returns Status code ([`SPIR_COMPUTATION_SUCCESS`](@ref) on success)
"""
function spir_funcs_get_size(funcs, size)
    ccall((:spir_funcs_get_size, libsparseir), StatusCode, (Ptr{spir_funcs}, Ptr{Cint}), funcs, size)
end

"""
    spir_funcs_get_n_knots(funcs, n_knots)

Gets the number of knots for continuous functions

# Arguments * `funcs` - Pointer to the funcs object * `n_knots` - Pointer to store the number of knots

# Returns Status code ([`SPIR_COMPUTATION_SUCCESS`](@ref) on success, [`SPIR_NOT_SUPPORTED`](@ref) if not continuous)
"""
function spir_funcs_get_n_knots(funcs, n_knots)
    ccall((:spir_funcs_get_n_knots, libsparseir), StatusCode, (Ptr{spir_funcs}, Ptr{Cint}), funcs, n_knots)
end

"""
    spir_funcs_get_knots(funcs, knots)

Gets the knot positions for continuous functions

# Arguments * `funcs` - Pointer to the funcs object * `knots` - Pre-allocated array to store knot positions

# Returns Status code ([`SPIR_COMPUTATION_SUCCESS`](@ref) on success, [`SPIR_NOT_SUPPORTED`](@ref) if not continuous)

# Safety The caller must ensure that `knots` has size >= `[`spir_funcs_get_n_knots`](@ref)(funcs)`
"""
function spir_funcs_get_knots(funcs, knots)
    ccall((:spir_funcs_get_knots, libsparseir), StatusCode, (Ptr{spir_funcs}, Ptr{Cdouble}), funcs, knots)
end

"""
    spir_funcs_eval(funcs, x, out)

Evaluate functions at a single point (continuous functions only)

# Arguments * `funcs` - Pointer to the funcs object * `x` - Point to evaluate at (tau coordinate in [-1, 1]) * `out` - Pre-allocated array to store function values

# Returns Status code ([`SPIR_COMPUTATION_SUCCESS`](@ref) on success, [`SPIR_NOT_SUPPORTED`](@ref) if not continuous)

# Safety The caller must ensure that `out` has size >= `[`spir_funcs_get_size`](@ref)(funcs)`
"""
function spir_funcs_eval(funcs, x, out)
    ccall((:spir_funcs_eval, libsparseir), StatusCode, (Ptr{spir_funcs}, Cdouble, Ptr{Cdouble}), funcs, x, out)
end

"""
    spir_funcs_eval_matsu(funcs, n, out)

Evaluate functions at a single Matsubara frequency

# Arguments * `funcs` - Pointer to the funcs object * `n` - Matsubara frequency index * `out` - Pre-allocated array to store complex function values

# Returns Status code ([`SPIR_COMPUTATION_SUCCESS`](@ref) on success, [`SPIR_NOT_SUPPORTED`](@ref) if not Matsubara type)

# Safety The caller must ensure that `out` has size >= `[`spir_funcs_get_size`](@ref)(funcs)` Complex numbers are laid out as [real, imag] pairs
"""
function spir_funcs_eval_matsu(funcs, n, out)
    ccall((:spir_funcs_eval_matsu, libsparseir), StatusCode, (Ptr{spir_funcs}, Int64, Ptr{Complex64}), funcs, n, out)
end

"""
    spir_funcs_batch_eval(funcs, order, num_points, xs, out)

Batch evaluate functions at multiple points (continuous functions only)

# Arguments * `funcs` - Pointer to the funcs object * `order` - Memory layout: 0 for row-major, 1 for column-major * `num_points` - Number of evaluation points * `xs` - Array of points to evaluate at * `out` - Pre-allocated array to store results

# Returns Status code ([`SPIR_COMPUTATION_SUCCESS`](@ref) on success, [`SPIR_NOT_SUPPORTED`](@ref) if not continuous)

# Safety - `xs` must have size >= `num_points` - `out` must have size >= `num\\_points * [`spir_funcs_get_size`](@ref)(funcs)` - Layout: row-major = out[point][func], column-major = out[func][point]
"""
function spir_funcs_batch_eval(funcs, order, num_points, xs, out)
    ccall((:spir_funcs_batch_eval, libsparseir), StatusCode, (Ptr{spir_funcs}, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}), funcs, order, num_points, xs, out)
end

"""
    spir_funcs_batch_eval_matsu(funcs, order, num_freqs, ns, out)

Batch evaluate functions at multiple Matsubara frequencies

# Arguments * `funcs` - Pointer to the funcs object * `order` - Memory layout: 0 for row-major, 1 for column-major * `num_freqs` - Number of Matsubara frequencies * `ns` - Array of Matsubara frequency indices * `out` - Pre-allocated array to store complex results

# Returns Status code ([`SPIR_COMPUTATION_SUCCESS`](@ref) on success, [`SPIR_NOT_SUPPORTED`](@ref) if not Matsubara type)

# Safety - `ns` must have size >= `num_freqs` - `out` must have size >= `num\\_freqs * [`spir_funcs_get_size`](@ref)(funcs)` - Complex numbers are laid out as [real, imag] pairs - Layout: row-major = out[freq][func], column-major = out[func][freq]
"""
function spir_funcs_batch_eval_matsu(funcs, order, num_freqs, ns, out)
    ccall((:spir_funcs_batch_eval_matsu, libsparseir), StatusCode, (Ptr{spir_funcs}, Cint, Cint, Ptr{Int64}, Ptr{Complex64}), funcs, order, num_freqs, ns, out)
end

"""
    spir_uhat_get_default_matsus(uhat, l, positive_only, mitigate, points, n_points_returned)

Get default Matsubara sampling points from a Matsubara-space [`spir_funcs`](@ref)

This function computes default sampling points in Matsubara frequencies (iωn) from a [`spir_funcs`](@ref) object that represents Matsubara-space basis functions (e.g., uhat or uhat\\_full). The statistics type (Fermionic/Bosonic) is automatically detected from the [`spir_funcs`](@ref) object type.

This extracts the PiecewiseLegendreFTVector from [`spir_funcs`](@ref) and calls `FiniteTempBasis::default\\_matsubara\\_sampling\\_points\\_impl` from `basis.rs` (lines 332-387) to compute default sampling points.

The implementation uses the same algorithm as defined in `sparseir-rust/src/basis.rs`, which selects sampling points based on sign changes or extrema of the Matsubara basis functions.

# Arguments * `uhat` - Pointer to a [`spir_funcs`](@ref) object representing Matsubara-space basis functions * `l` - Number of requested sampling points * `positive_only` - If true, only positive frequencies are used * `mitigate` - If true, enable mitigation (fencing) to improve conditioning by adding oversampling points * `points` - Pre-allocated array to store the sampling points. The size of the array must be sufficient for the returned points (may exceed L if mitigate is true). * `n_points_returned` - Pointer to store the number of sampling points returned (may exceed L if mitigate is true, or approximately L/2 when positive\\_only=true).

# Returns Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if uhat, points, or n\\_points\\_returned is null - [`SPIR_NOT_SUPPORTED`](@ref) if uhat is not a Matsubara-space function

# Note This function is only available for [`spir_funcs`](@ref) objects representing Matsubara-space basis functions The statistics type is automatically detected from the [`spir_funcs`](@ref) object type The default sampling points are chosen to provide near-optimal conditioning
"""
function spir_uhat_get_default_matsus(uhat, l, positive_only, mitigate, points, n_points_returned)
    ccall((:spir_uhat_get_default_matsus, libsparseir), StatusCode, (Ptr{spir_funcs}, Cint, Bool, Bool, Ptr{Int64}, Ptr{Cint}), uhat, l, positive_only, mitigate, points, n_points_returned)
end

"""
    spir_gemm_backend_new_from_fblas_lp64(dgemm, zgemm)

Create GEMM backend from Fortran BLAS function pointers (LP64)

Creates a new backend handle from Fortran BLAS function pointers.

# Arguments * `dgemm` - Function pointer to Fortran BLAS dgemm (double precision) * `zgemm` - Function pointer to Fortran BLAS zgemm (complex double precision)

# Returns * Pointer to [`spir_gemm_backend`](@ref) on success * `NULL` if function pointers are null

# Safety The provided function pointers must: - Be valid Fortran BLAS function pointers following the standard Fortran BLAS interface - Use 32-bit integers for all dimension parameters (LP64 interface) - Be thread-safe (will be called from multiple threads) - Remain valid for the entire lifetime of the backend handle

The returned pointer must be freed with `spir_gemm_backend_free` when no longer needed.
"""
function spir_gemm_backend_new_from_fblas_lp64(dgemm, zgemm)
    ccall((:spir_gemm_backend_new_from_fblas_lp64, libsparseir), Ptr{spir_gemm_backend}, (Ptr{Cvoid}, Ptr{Cvoid}), dgemm, zgemm)
end

"""
    spir_gemm_backend_new_from_fblas_ilp64(dgemm64, zgemm64)

Create GEMM backend from Fortran BLAS function pointers (ILP64)

Creates a new backend handle from Fortran BLAS function pointers with 64-bit integers.

# Arguments * `dgemm64` - Function pointer to Fortran BLAS dgemm (double precision, 64-bit integers) * `zgemm64` - Function pointer to Fortran BLAS zgemm (complex double precision, 64-bit integers)

# Returns * Pointer to [`spir_gemm_backend`](@ref) on success * `NULL` if function pointers are null

# Safety The provided function pointers must: - Be valid Fortran BLAS function pointers following the standard Fortran BLAS interface - Use 64-bit integers for all dimension parameters (ILP64 interface) - Be thread-safe (will be called from multiple threads) - Remain valid for the entire lifetime of the backend handle

The returned pointer must be freed with `spir_gemm_backend_free` when no longer needed.
"""
function spir_gemm_backend_new_from_fblas_ilp64(dgemm64, zgemm64)
    ccall((:spir_gemm_backend_new_from_fblas_ilp64, libsparseir), Ptr{spir_gemm_backend}, (Ptr{Cvoid}, Ptr{Cvoid}), dgemm64, zgemm64)
end

"""
    spir_gemm_backend_release(backend)

Release GEMM backend handle

Releases the memory associated with a backend handle.

# Arguments * `backend` - Pointer to backend handle (can be NULL)

# Safety The pointer must have been created by [`spir_gemm_backend_new_from_fblas_lp64`](@ref) or [`spir_gemm_backend_new_from_fblas_ilp64`](@ref). After calling this function, the pointer must not be used again.
"""
function spir_gemm_backend_release(backend)
    ccall((:spir_gemm_backend_release, libsparseir), Cvoid, (Ptr{spir_gemm_backend},), backend)
end

"""
    spir_logistic_kernel_new(lambda, status)

Create a new Logistic kernel

# Arguments * `lambda` - The kernel parameter Λ = β * ωmax (must be > 0) * `status` - Pointer to store the status code

# Returns * Pointer to the newly created kernel object, or NULL if creation fails

# Safety The caller must ensure `status` is a valid pointer.

# Example (C) ```c int status; [`spir_kernel`](@ref)* kernel = [`spir_logistic_kernel_new`](@ref)(10.0, &status); if (kernel != NULL) { // Use kernel... [`spir_kernel_release`](@ref)(kernel); } ```
"""
function spir_logistic_kernel_new(lambda, status)
    ccall((:spir_logistic_kernel_new, libsparseir), Ptr{spir_kernel}, (Cdouble, Ptr{StatusCode}), lambda, status)
end

"""
    spir_reg_bose_kernel_new(lambda, status)

Create a new RegularizedBose kernel

# Arguments * `lambda` - The kernel parameter Λ = β * ωmax (must be > 0) * `status` - Pointer to store the status code

# Returns * Pointer to the newly created kernel object, or NULL if creation fails
"""
function spir_reg_bose_kernel_new(lambda, status)
    ccall((:spir_reg_bose_kernel_new, libsparseir), Ptr{spir_kernel}, (Cdouble, Ptr{StatusCode}), lambda, status)
end

"""
    spir_kernel_get_lambda(kernel, lambda_out)

Get the lambda parameter of a kernel

# Arguments * `kernel` - Kernel object * `lambda_out` - Pointer to store the lambda value

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if kernel or lambda\\_out is null * [`SPIR_INTERNAL_ERROR`](@ref) if internal panic occurs
"""
function spir_kernel_get_lambda(kernel, lambda_out)
    ccall((:spir_kernel_get_lambda, libsparseir), StatusCode, (Ptr{spir_kernel}, Ptr{Cdouble}), kernel, lambda_out)
end

"""
    spir_kernel_compute(kernel, x, y, out)

Compute kernel value K(x, y)

# Arguments * `kernel` - Kernel object * `x` - First argument (typically in [-1, 1]) * `y` - Second argument (typically in [-1, 1]) * `out` - Pointer to store the result

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if kernel or out is null * [`SPIR_INTERNAL_ERROR`](@ref) if internal panic occurs
"""
function spir_kernel_compute(kernel, x, y, out)
    ccall((:spir_kernel_compute, libsparseir), StatusCode, (Ptr{spir_kernel}, Cdouble, Cdouble, Ptr{Cdouble}), kernel, x, y, out)
end

"""
    spir_kernel_release(kernel)

Manual release function (replaces macro-generated one)

# Safety This function drops the kernel. The inner KernelType data is automatically freed by the Drop implementation when the [`spir_kernel`](@ref) structure is dropped.
"""
function spir_kernel_release(kernel)
    ccall((:spir_kernel_release, libsparseir), Cvoid, (Ptr{spir_kernel},), kernel)
end

"""
    spir_kernel_clone(src)

Manual clone function (replaces macro-generated one)
"""
function spir_kernel_clone(src)
    ccall((:spir_kernel_clone, libsparseir), Ptr{spir_kernel}, (Ptr{spir_kernel},), src)
end

"""
    spir_kernel_is_assigned(obj)

Manual is\\_assigned function (replaces macro-generated one)
"""
function spir_kernel_is_assigned(obj)
    ccall((:spir_kernel_is_assigned, libsparseir), Int32, (Ptr{spir_kernel},), obj)
end

"""
    spir_kernel_get_domain(k, xmin, xmax, ymin, ymax)

Get kernel domain boundaries

# Arguments * `k` - Kernel object * `xmin` - Pointer to store minimum x value * `xmax` - Pointer to store maximum x value * `ymin` - Pointer to store minimum y value * `ymax` - Pointer to store maximum y value

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if any pointer is null * [`SPIR_INTERNAL_ERROR`](@ref) if internal panic occurs
"""
function spir_kernel_get_domain(k, xmin, xmax, ymin, ymax)
    ccall((:spir_kernel_get_domain, libsparseir), StatusCode, (Ptr{spir_kernel}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), k, xmin, xmax, ymin, ymax)
end

"""
    spir_kernel_get_sve_hints_segments_x(k, epsilon, segments, n_segments)

Get x-segments for SVE discretization hints from a kernel

This function should be called twice: 1. First call with segments=NULL: set n\\_segments to the required array size 2. Second call with segments allocated: fill segments[0..n\\_segments-1] with values

# Arguments * `k` - Kernel object * `epsilon` - Accuracy target for the basis * `segments` - Pointer to store segments array (NULL for first call) * `n_segments` - [IN/OUT] Input: ignored when segments is NULL. Output: number of segments

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if k or n\\_segments is null, or segments array is too small * [`SPIR_INTERNAL_ERROR`](@ref) if internal panic occurs
"""
function spir_kernel_get_sve_hints_segments_x(k, epsilon, segments, n_segments)
    ccall((:spir_kernel_get_sve_hints_segments_x, libsparseir), StatusCode, (Ptr{spir_kernel}, Cdouble, Ptr{Cdouble}, Ptr{Cint}), k, epsilon, segments, n_segments)
end

"""
    spir_kernel_get_sve_hints_segments_y(k, epsilon, segments, n_segments)

Get y-segments for SVE discretization hints from a kernel

This function should be called twice: 1. First call with segments=NULL: set n\\_segments to the required array size 2. Second call with segments allocated: fill segments[0..n\\_segments-1] with values

# Arguments * `k` - Kernel object * `epsilon` - Accuracy target for the basis * `segments` - Pointer to store segments array (NULL for first call) * `n_segments` - [IN/OUT] Input: ignored when segments is NULL. Output: number of segments

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if k or n\\_segments is null, or segments array is too small * [`SPIR_INTERNAL_ERROR`](@ref) if internal panic occurs
"""
function spir_kernel_get_sve_hints_segments_y(k, epsilon, segments, n_segments)
    ccall((:spir_kernel_get_sve_hints_segments_y, libsparseir), StatusCode, (Ptr{spir_kernel}, Cdouble, Ptr{Cdouble}, Ptr{Cint}), k, epsilon, segments, n_segments)
end

"""
    spir_kernel_get_sve_hints_nsvals(k, epsilon, nsvals)

Get the number of singular values hint from a kernel

# Arguments * `k` - Kernel object * `epsilon` - Accuracy target for the basis * `nsvals` - Pointer to store the number of singular values

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if k or nsvals is null * [`SPIR_INTERNAL_ERROR`](@ref) if internal panic occurs
"""
function spir_kernel_get_sve_hints_nsvals(k, epsilon, nsvals)
    ccall((:spir_kernel_get_sve_hints_nsvals, libsparseir), StatusCode, (Ptr{spir_kernel}, Cdouble, Ptr{Cint}), k, epsilon, nsvals)
end

"""
    spir_kernel_get_sve_hints_ngauss(k, epsilon, ngauss)

Get the number of Gauss points hint from a kernel

# Arguments * `k` - Kernel object * `epsilon` - Accuracy target for the basis * `ngauss` - Pointer to store the number of Gauss points

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if k or ngauss is null * [`SPIR_INTERNAL_ERROR`](@ref) if internal panic occurs
"""
function spir_kernel_get_sve_hints_ngauss(k, epsilon, ngauss)
    ccall((:spir_kernel_get_sve_hints_ngauss, libsparseir), StatusCode, (Ptr{spir_kernel}, Cdouble, Ptr{Cint}), k, epsilon, ngauss)
end

"""
    spir_sampling_release(sampling)

Manual release function (replaces macro-generated one)
"""
function spir_sampling_release(sampling)
    ccall((:spir_sampling_release, libsparseir), Cvoid, (Ptr{spir_sampling},), sampling)
end

"""
    spir_sampling_clone(src)

Manual clone function (replaces macro-generated one)
"""
function spir_sampling_clone(src)
    ccall((:spir_sampling_clone, libsparseir), Ptr{spir_sampling}, (Ptr{spir_sampling},), src)
end

"""
    spir_sampling_is_assigned(obj)

Manual is\\_assigned function (replaces macro-generated one)
"""
function spir_sampling_is_assigned(obj)
    ccall((:spir_sampling_is_assigned, libsparseir), Int32, (Ptr{spir_sampling},), obj)
end

"""
    spir_tau_sampling_new(b, num_points, points, status)

Creates a new tau sampling object for sparse sampling in imaginary time

# Arguments * `b` - Pointer to a finite temperature basis object * `num_points` - Number of sampling points * `points` - Array of sampling points in imaginary time (τ) * `status` - Pointer to store the status code

# Returns Pointer to the newly created sampling object, or NULL if creation fails

# Safety Caller must ensure `b` is valid and `points` has `num_points` elements
"""
function spir_tau_sampling_new(b, num_points, points, status)
    ccall((:spir_tau_sampling_new, libsparseir), Ptr{spir_sampling}, (Ptr{spir_basis}, Cint, Ptr{Cdouble}, Ptr{StatusCode}), b, num_points, points, status)
end

"""
    spir_matsu_sampling_new(b, positive_only, num_points, points, status)

Creates a new Matsubara sampling object for sparse sampling in Matsubara frequencies

# Arguments * `b` - Pointer to a finite temperature basis object * `positive_only` - If true, only positive frequencies are used * `num_points` - Number of sampling points * `points` - Array of Matsubara frequency indices (n) * `status` - Pointer to store the status code

# Returns Pointer to the newly created sampling object, or NULL if creation fails
"""
function spir_matsu_sampling_new(b, positive_only, num_points, points, status)
    ccall((:spir_matsu_sampling_new, libsparseir), Ptr{spir_sampling}, (Ptr{spir_basis}, Bool, Cint, Ptr{Int64}, Ptr{StatusCode}), b, positive_only, num_points, points, status)
end

"""
    spir_tau_sampling_new_with_matrix(order, statistics, basis_size, num_points, points, matrix, status)

Creates a new tau sampling object with custom sampling points and pre-computed matrix

# Arguments * `order` - Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `statistics` - Statistics type ([`SPIR_STATISTICS_FERMIONIC`](@ref) or [`SPIR_STATISTICS_BOSONIC`](@ref)) * `basis_size` - Basis size * `num_points` - Number of sampling points * `points` - Array of sampling points in imaginary time (τ) * `matrix` - Pre-computed matrix for the sampling points (num\\_points x basis\\_size) * `status` - Pointer to store the status code

# Returns Pointer to the newly created sampling object, or NULL if creation fails

# Safety Caller must ensure `points` and `matrix` have correct sizes
"""
function spir_tau_sampling_new_with_matrix(order, statistics, basis_size, num_points, points, matrix, status)
    ccall((:spir_tau_sampling_new_with_matrix, libsparseir), Ptr{spir_sampling}, (Cint, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{StatusCode}), order, statistics, basis_size, num_points, points, matrix, status)
end

"""
    spir_matsu_sampling_new_with_matrix(order, statistics, basis_size, positive_only, num_points, points, matrix, status)

Creates a new Matsubara sampling object with custom sampling points and pre-computed matrix

# Arguments * `order` - Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `statistics` - Statistics type ([`SPIR_STATISTICS_FERMIONIC`](@ref) or [`SPIR_STATISTICS_BOSONIC`](@ref)) * `basis_size` - Basis size * `positive_only` - If true, only positive frequencies are used * `num_points` - Number of sampling points * `points` - Array of Matsubara frequency indices (n) * `matrix` - Pre-computed complex matrix (num\\_points x basis\\_size) * `status` - Pointer to store the status code

# Returns Pointer to the newly created sampling object, or NULL if creation fails

# Safety Caller must ensure `points` and `matrix` have correct sizes
"""
function spir_matsu_sampling_new_with_matrix(order, statistics, basis_size, positive_only, num_points, points, matrix, status)
    ccall((:spir_matsu_sampling_new_with_matrix, libsparseir), Ptr{spir_sampling}, (Cint, Cint, Cint, Bool, Cint, Ptr{Int64}, Ptr{Complex64}, Ptr{StatusCode}), order, statistics, basis_size, positive_only, num_points, points, matrix, status)
end

"""
    spir_sampling_get_npoints(s, num_points)

Gets the number of sampling points in a sampling object.

This function returns the number of sampling points used in the specified sampling object. This number is needed to allocate arrays of the correct size when retrieving the actual sampling points.

# Arguments

* `s` - Pointer to the sampling object. * `num_points` - Pointer to store the number of sampling points.

# Returns

A status code: - `0` ([[`SPIR_COMPUTATION_SUCCESS`](@ref)]) on success - A non-zero error code on failure

# See also

- [[`spir_sampling_get_taus`](@ref)] - [[`spir_sampling_get_matsus`](@ref)]
"""
function spir_sampling_get_npoints(s, num_points)
    ccall((:spir_sampling_get_npoints, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{Cint}), s, num_points)
end

"""
    spir_sampling_get_taus(s, points)

Gets the imaginary time (τ) sampling points used in the specified sampling object.

This function fills the provided array with the imaginary time (τ) sampling points used in the specified sampling object. The array must be pre-allocated with sufficient size (use [[`spir_sampling_get_npoints`](@ref)] to determine the required size).

# Arguments

* `s` - Pointer to the sampling object. * `points` - Pre-allocated array to store the τ sampling points.

# Returns

An integer status code: - `0` ([[`SPIR_COMPUTATION_SUCCESS`](@ref)]) on success - A non-zero error code on failure

# Notes

The array must be pre-allocated with size >= [[`spir_sampling_get_npoints`](@ref)]([`spir_sampling_get_npoints`](@ref)).

# See also

- [[`spir_sampling_get_npoints`](@ref)]
"""
function spir_sampling_get_taus(s, points)
    ccall((:spir_sampling_get_taus, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{Cdouble}), s, points)
end

"""
    spir_sampling_get_matsus(s, points)

Gets the Matsubara frequency sampling points
"""
function spir_sampling_get_matsus(s, points)
    ccall((:spir_sampling_get_matsus, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{Int64}), s, points)
end

"""
    spir_sampling_get_cond_num(s, cond_num)

Gets the condition number of the sampling matrix.

This function returns the condition number of the sampling matrix used in the specified sampling object. The condition number is a measure of how well- conditioned the sampling matrix is.

# Parameters - `s`: Pointer to the sampling object. - `cond_num`: Pointer to store the condition number.

# Returns An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - Non-zero error code on failure

# Notes - A large condition number indicates that the sampling matrix is ill-conditioned, which may lead to numerical instability in transformations. - The condition number is the ratio of the largest to smallest singular value of the sampling matrix.
"""
function spir_sampling_get_cond_num(s, cond_num)
    ccall((:spir_sampling_get_cond_num, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{Cdouble}), s, cond_num)
end

"""
    spir_sampling_eval_dd(s, backend, order, ndim, input_dims, target_dim, input, out)

Evaluates basis coefficients at sampling points (double to double version).

Transforms basis coefficients to values at sampling points, where both input and output are real (double precision) values. The operation can be performed along any dimension of a multidimensional array.

# Arguments

* `s` - Pointer to the sampling object * `order` - Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `ndim` - Number of dimensions in the input/output arrays * `input_dims` - Array of dimension sizes * `target_dim` - Target dimension for the transformation (0-based) * `input` - Input array of basis coefficients * `out` - Output array for the evaluated values at sampling points

# Returns

An integer status code: - `0` ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure

# Notes

- For optimal performance, the target dimension should be either the first (`0`) or the last (`ndim-1`) dimension to avoid large temporary array allocations - The output array must be pre-allocated with the correct size - The input and output arrays must be contiguous in memory - The transformation is performed using a pre-computed sampling matrix that is factorized using SVD for efficiency

# See also - [[`spir_sampling_eval_dz`](@ref)] - [[`spir_sampling_eval_zz`](@ref)] # Note Supports both row-major and column-major order. Zero-copy implementation.
"""
function spir_sampling_eval_dd(s, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_dd, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), s, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_eval_dz(s, backend, order, ndim, input_dims, target_dim, input, out)

Evaluate basis coefficients at sampling points (double → complex)

For Matsubara sampling: transforms real IR coefficients to complex values. Zero-copy implementation.
"""
function spir_sampling_eval_dz(s, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_dz, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Complex64}), s, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_eval_zz(s, backend, order, ndim, input_dims, target_dim, input, out)

Evaluate basis coefficients at sampling points (complex → complex)

For Matsubara sampling: transforms complex coefficients to complex values. Zero-copy implementation.
"""
function spir_sampling_eval_zz(s, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_zz, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Complex64}, Ptr{Complex64}), s, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_fit_dd(s, backend, order, ndim, input_dims, target_dim, input, out)

Fits values at sampling points to basis coefficients (double to double version).

Transforms values at sampling points back to basis coefficients, where both input and output are real (double precision) values. The operation can be performed along any dimension of a multidimensional array.

# Arguments

* `s` - Pointer to the sampling object * `backend` - Pointer to the GEMM backend (can be null to use default) * `order` - Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `ndim` - Number of dimensions in the input/output arrays * `input_dims` - Array of dimension sizes * `target_dim` - Target dimension for the transformation (0-based) * `input` - Input array of values at sampling points * `out` - Output array for the fitted basis coefficients

# Returns

An integer status code: * `0` ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success * A non-zero error code on failure

# Notes

* The output array must be pre-allocated with the correct size * This function performs the inverse operation of [`spir_sampling_eval_dd`](@ref) * The transformation is performed using a pre-computed sampling matrix that is factorized using SVD for efficiency * Zero-copy implementation

# See also

* [[`spir_sampling_eval_dd`](@ref)] * [[`spir_sampling_fit_zz`](@ref)]
"""
function spir_sampling_fit_dd(s, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_fit_dd, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), s, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_fit_zz(s, backend, order, ndim, input_dims, target_dim, input, out)

Fits values at sampling points to basis coefficients (complex to complex version).

For more details, see [[`spir_sampling_fit_dd`](@ref)] Zero-copy implementation for Tau and Matsubara (full). MatsubaraPositiveOnly requires intermediate storage for real→complex conversion.
"""
function spir_sampling_fit_zz(s, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_fit_zz, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Complex64}, Ptr{Complex64}), s, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_fit_zd(s, backend, order, ndim, input_dims, target_dim, input, out)

Fit basis coefficients from Matsubara sampling points (complex input, real output)

This function fits basis coefficients from Matsubara sampling points using complex input and real output.

# Supported Sampling Types

- **Matsubara (full)**: ✅ Supported (takes real part of fitted complex coefficients) - **Matsubara (positive\\_only)**: ✅ Supported - **Tau**: ❌ Not supported (use [`spir_sampling_fit_dd`](@ref) instead)

# Notes

For full-range Matsubara sampling, this function fits complex coefficients internally and returns their real parts. This is physically correct for Green's functions where IR coefficients are guaranteed to be real by symmetry.

Zero-copy implementation.

# Arguments

* `s` - Pointer to the sampling object (must be Matsubara) * `backend` - Pointer to the GEMM backend (can be null to use default) * `order` - Memory layout order ([`SPIR_ORDER_COLUMN_MAJOR`](@ref) or [`SPIR_ORDER_ROW_MAJOR`](@ref)) * `ndim` - Number of dimensions in the input/output arrays * `input_dims` - Array of dimension sizes * `target_dim` - Target dimension for the transformation (0-based) * `input` - Input array (complex) * `out` - Output array (real)

# Returns

- [`SPIR_COMPUTATION_SUCCESS`](@ref) on success - [`SPIR_NOT_SUPPORTED`](@ref) if the sampling type doesn't support this operation - Other error codes on failure

# See also

* [[`spir_sampling_fit_zz`](@ref)] * [[`spir_sampling_fit_dd`](@ref)]
"""
function spir_sampling_fit_zd(s, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_fit_zd, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Complex64}, Ptr{Cdouble}), s, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sve_result_release(sve)

Manual release function (replaces macro-generated one)
"""
function spir_sve_result_release(sve)
    ccall((:spir_sve_result_release, libsparseir), Cvoid, (Ptr{spir_sve_result},), sve)
end

"""
    spir_sve_result_clone(src)

Manual clone function (replaces macro-generated one)
"""
function spir_sve_result_clone(src)
    ccall((:spir_sve_result_clone, libsparseir), Ptr{spir_sve_result}, (Ptr{spir_sve_result},), src)
end

"""
    spir_sve_result_is_assigned(obj)

Manual is\\_assigned function (replaces macro-generated one)
"""
function spir_sve_result_is_assigned(obj)
    ccall((:spir_sve_result_is_assigned, libsparseir), Int32, (Ptr{spir_sve_result},), obj)
end

"""
    spir_sve_result_new(k, epsilon, _lmax, _n_gauss, twork, status)

Compute Singular Value Expansion (SVE) of a kernel (libsparseir compatible)

# Arguments * `k` - Kernel object * `epsilon` - Accuracy target for the basis * `lmax` - Maximum number of Legendre polynomials (currently ignored, auto-determined) * `n_gauss` - Number of Gauss points for integration (currently ignored, auto-determined) * `Twork` - Working precision: 0=Float64, 1=Float64x2, -1=Auto * `status` - Pointer to store status code

# Returns * Pointer to SVE result, or NULL on failure

# Safety The caller must ensure `status` is a valid pointer.

# Note Parameters `lmax` and `n_gauss` are accepted for libsparseir compatibility but currently ignored. The Rust implementation automatically determines optimal values. The cutoff is automatically set to 2*sqrt(machine\\_epsilon) internally.
"""
function spir_sve_result_new(k, epsilon, _lmax, _n_gauss, twork, status)
    ccall((:spir_sve_result_new, libsparseir), Ptr{spir_sve_result}, (Ptr{spir_kernel}, Cdouble, Cint, Cint, Cint, Ptr{StatusCode}), k, epsilon, _lmax, _n_gauss, twork, status)
end

"""
    spir_sve_result_get_size(sve, size)

Get the number of singular values in an SVE result

# Arguments * `sve` - SVE result object * `size` - Pointer to store the size

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if sve or size is null * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_sve_result_get_size(sve, size)
    ccall((:spir_sve_result_get_size, libsparseir), StatusCode, (Ptr{spir_sve_result}, Ptr{Cint}), sve, size)
end

"""
    spir_sve_result_truncate(sve, epsilon, max_size, status)

Truncate an SVE result based on epsilon and max\\_size

This function creates a new SVE result containing only the singular values that are larger than `epsilon * s[0]`, where `s[0]` is the largest singular value. The result can also be limited to a maximum size.

# Arguments * `sve` - Source SVE result object * `epsilon` - Relative threshold for truncation (singular values < epsilon * s[0] are removed) * `max_size` - Maximum number of singular values to keep (-1 for no limit) * `status` - Pointer to store status code

# Returns * Pointer to new truncated SVE result, or NULL on failure * Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if sve or status is null, or epsilon is invalid - [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs

# Safety The caller must ensure `status` is a valid pointer. The returned pointer must be freed with `[`spir_sve_result_release`](@ref)()`.

# Example (C) ```c [`spir_sve_result`](@ref)* sve = [`spir_sve_result_new`](@ref)(kernel, 1e-10, 0, 0, -1, &status);

// Truncate to keep only singular values > 1e-8 * s[0], max 50 values [`spir_sve_result`](@ref)* sve\\_truncated = [`spir_sve_result_truncate`](@ref)(sve, 1e-8, 50, &status);

// Use truncated result...

[`spir_sve_result_release`](@ref)(sve\\_truncated); [`spir_sve_result_release`](@ref)(sve); ```
"""
function spir_sve_result_truncate(sve, epsilon, max_size, status)
    ccall((:spir_sve_result_truncate, libsparseir), Ptr{spir_sve_result}, (Ptr{spir_sve_result}, Cdouble, Cint, Ptr{StatusCode}), sve, epsilon, max_size, status)
end

"""
    spir_sve_result_get_svals(sve, svals)

Get singular values from an SVE result

# Arguments * `sve` - SVE result object * `svals` - Pre-allocated array to store singular values (size must be >= result size)

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if sve or svals is null * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_sve_result_get_svals(sve, svals)
    ccall((:spir_sve_result_get_svals, libsparseir), StatusCode, (Ptr{spir_sve_result}, Ptr{Cdouble}), sve, svals)
end

"""
    spir_sve_result_from_matrix(K_high, K_low, nx, ny, order, segments_x, n_segments_x, segments_y, n_segments_y, n_gauss, epsilon, status)

Create a SVE result from a discretized kernel matrix

This function performs singular value expansion (SVE) on a discretized kernel matrix K. The matrix K should already be in the appropriate form (no weight application needed). The function supports both double and DDouble precision based on whether K\\_low is provided.

# Arguments * `K_high` - High part of the kernel matrix (required, size: nx * ny) * `K_low` - Low part of the kernel matrix (optional, nullptr for double precision) * `nx` - Number of rows in the matrix * `ny` - Number of columns in the matrix * `order` - Memory layout ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `segments_x` - X-direction segments (array of boundary points, size: n\\_segments\\_x + 1) * `n_segments_x` - Number of segments in x direction (boundary points - 1) * `segments_y` - Y-direction segments (array of boundary points, size: n\\_segments\\_y + 1) * `n_segments_y` - Number of segments in y direction (boundary points - 1) * `n_gauss` - Number of Gauss points per segment * `epsilon` - Target accuracy * `status` - Pointer to store status code

# Returns Pointer to SVE result on success, nullptr on failure
"""
function spir_sve_result_from_matrix(K_high, K_low, nx, ny, order, segments_x, n_segments_x, segments_y, n_segments_y, n_gauss, epsilon, status)
    ccall((:spir_sve_result_from_matrix, libsparseir), Ptr{spir_sve_result}, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Cdouble, Ptr{StatusCode}), K_high, K_low, nx, ny, order, segments_x, n_segments_x, segments_y, n_segments_y, n_gauss, epsilon, status)
end

"""
    spir_sve_result_from_matrix_centrosymmetric(K_even_high, K_even_low, K_odd_high, K_odd_low, nx, ny, order, segments_x, n_segments_x, segments_y, n_segments_y, n_gauss, epsilon, status)

Create a SVE result from centrosymmetric discretized kernel matrices

This function performs singular value expansion (SVE) on centrosymmetric discretized kernel matrices using even/odd symmetry decomposition. The matrices K\\_even and K\\_odd should already be in the appropriate form (no weight application needed). The function supports both double and DDouble precision based on whether K\\_low is provided.

# Arguments * `K_even_high` - High part of the even-symmetry kernel matrix (required, size: nx * ny) * `K_even_low` - Low part of the even-symmetry kernel matrix (optional, nullptr for double precision) * `K_odd_high` - High part of the odd-symmetry kernel matrix (required, size: nx * ny) * `K_odd_low` - Low part of the odd-symmetry kernel matrix (optional, nullptr for double precision) * `nx` - Number of rows in the matrix * `ny` - Number of columns in the matrix * `order` - Memory layout ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `segments_x` - X-direction segments (array of boundary points, size: n\\_segments\\_x + 1) * `n_segments_x` - Number of segments in x direction (boundary points - 1) * `segments_y` - Y-direction segments (array of boundary points, size: n\\_segments\\_y + 1) * `n_segments_y` - Number of segments in y direction (boundary points - 1) * `n_gauss` - Number of Gauss points per segment * `epsilon` - Target accuracy * `status` - Pointer to store status code

# Returns Pointer to SVE result on success, nullptr on failure
"""
function spir_sve_result_from_matrix_centrosymmetric(K_even_high, K_even_low, K_odd_high, K_odd_low, nx, ny, order, segments_x, n_segments_x, segments_y, n_segments_y, n_gauss, epsilon, status)
    ccall((:spir_sve_result_from_matrix_centrosymmetric, libsparseir), Ptr{spir_sve_result}, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Cdouble, Ptr{StatusCode}), K_even_high, K_even_low, K_odd_high, K_odd_low, nx, ny, order, segments_x, n_segments_x, segments_y, n_segments_y, n_gauss, epsilon, status)
end

"""
    spir_choose_working_type(epsilon)

Choose the working type (Twork) based on epsilon value

This function determines the appropriate working precision type based on the target accuracy epsilon. It follows the same logic as [`SPIR_TWORK_AUTO`](@ref): - Returns [`SPIR_TWORK_FLOAT64X2`](@ref) if epsilon < 1e-8 or epsilon is NaN - Returns [`SPIR_TWORK_FLOAT64`](@ref) otherwise

# Arguments * `epsilon` - Target accuracy (must be non-negative, or NaN for auto-selection)

# Returns Working type constant: - [`SPIR_TWORK_FLOAT64`](@ref) (0): Use double precision (64-bit) - [`SPIR_TWORK_FLOAT64X2`](@ref) (1): Use extended precision (128-bit)
"""
function spir_choose_working_type(epsilon)
    ccall((:spir_choose_working_type, libsparseir), Cint, (Cdouble,), epsilon)
end

"""
    spir_gauss_legendre_rule_piecewise_double(n, segments, n_segments, x, w, status)

Compute piecewise Gauss-Legendre quadrature rule (double precision)

Generates a piecewise Gauss-Legendre quadrature rule with n points per segment. The rule is concatenated across all segments, with points and weights properly scaled for each segment interval.

# Arguments * `n` - Number of Gauss points per segment (must be >= 1) * `segments` - Array of segment boundaries (n\\_segments + 1 elements). Must be monotonically increasing. * `n_segments` - Number of segments (must be >= 1) * `x` - Output array for Gauss points (size n * n\\_segments). Must be pre-allocated. * `w` - Output array for Gauss weights (size n * n\\_segments). Must be pre-allocated. * `status` - Pointer to store the status code

# Returns Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - Non-zero error code on failure
"""
function spir_gauss_legendre_rule_piecewise_double(n, segments, n_segments, x, w, status)
    ccall((:spir_gauss_legendre_rule_piecewise_double, libsparseir), StatusCode, (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{StatusCode}), n, segments, n_segments, x, w, status)
end

"""
    spir_gauss_legendre_rule_piecewise_ddouble(n, segments, n_segments, x_high, x_low, w_high, w_low, status)

Compute piecewise Gauss-Legendre quadrature rule (DDouble precision)

Generates a piecewise Gauss-Legendre quadrature rule with n points per segment, computed using extended precision (DDouble). Returns high and low parts separately for maximum precision.

# Arguments * `n` - Number of Gauss points per segment (must be >= 1) * `segments` - Array of segment boundaries (n\\_segments + 1 elements). Must be monotonically increasing. * `n_segments` - Number of segments (must be >= 1) * `x_high` - Output array for high part of Gauss points (size n * n\\_segments). Must be pre-allocated. * `x_low` - Output array for low part of Gauss points (size n * n\\_segments). Must be pre-allocated. * `w_high` - Output array for high part of Gauss weights (size n * n\\_segments). Must be pre-allocated. * `w_low` - Output array for low part of Gauss weights (size n * n\\_segments). Must be pre-allocated. * `status` - Pointer to store the status code

# Returns Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - Non-zero error code on failure
"""
function spir_gauss_legendre_rule_piecewise_ddouble(n, segments, n_segments, x_high, x_low, w_high, w_low, status)
    ccall((:spir_gauss_legendre_rule_piecewise_ddouble, libsparseir), StatusCode, (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{StatusCode}), n, segments, n_segments, x_high, x_low, w_high, w_low, status)
end

const SPIR_ORDER_ROW_MAJOR = 0

const SPIR_ORDER_COLUMN_MAJOR = 1

const SPIR_STATISTICS_BOSONIC = 0

const SPIR_STATISTICS_FERMIONIC = 1

const SPIR_TWORK_FLOAT64 = 0

const SPIR_TWORK_FLOAT64X2 = 1

const SPIR_TWORK_AUTO = -1

const SPIR_SVDSTRAT_FAST = 0

const SPIR_SVDSTRAT_ACCURATE = 1

const SPIR_SVDSTRAT_AUTO = -1

const SPIR_COMPUTATION_SUCCESS = 0

const SPIR_GET_IMPL_FAILED = -1

const SPIR_INVALID_DIMENSION = -2

const SPIR_INPUT_DIMENSION_MISMATCH = -3

const SPIR_OUTPUT_DIMENSION_MISMATCH = -4

const SPIR_NOT_SUPPORTED = -5

const SPIR_INVALID_ARGUMENT = -6

const SPIR_INTERNAL_ERROR = -7

# exports
const PREFIXES = ["spir_", "SPIR_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
