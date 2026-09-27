module C_API

using CEnum: CEnum, @cenum

using Libdl: Libdl
using libsparseir_jll: libsparseir_jll

function get_libsparseir()
    deps_dir = joinpath(dirname(@__DIR__), "deps")
    local_libsparseir_path = joinpath(deps_dir, "libsparse_ir_capi.$(Libdl.dlext)")
    if isfile(local_libsparseir_path)
        @info "Using local libsparseir: $local_libsparseir_path"
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

Note: Named [`spir_sve_result`](@ref) to match libsparseir C++ API exactly. The internal structure is hidden using a void pointer to prevent exposing `Arc<SVEResult>` to C.
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

Check if the basis pointer is non-null.

Note: This only performs a null check. It cannot detect dangling pointers; dereferencing an arbitrary non-null pointer would be undefined behaviour that `catch_unwind` cannot reliably catch.

# Returns 1 if the pointer is non-null, 0 otherwise
"""
function spir_basis_is_assigned(obj)
    ccall((:spir_basis_is_assigned, libsparseir), Int32, (Ptr{spir_basis},), obj)
end

"""
    spir_basis_new(statistics, beta, omega_max, epsilon, k, sve, max_size, status)

Create a finite temperature basis (libsparseir compatible)

# Arguments * `statistics` - 0 for Bosonic, 1 for Fermionic * `beta` - Inverse temperature (must be > 0) * `omega_max` - Frequency cutoff (must be > 0) * `epsilon` - Accuracy target (must be > 0) * `k` - Kernel object (required; its Λ must equal beta * omega\\_max) * `sve` - Pre-computed SVE result (can be NULL, will compute if needed) * `max_size` - Maximum basis size (-1 for no limit). It truncates the basis, not the SVE (also when `sve` is NULL and the SVE is computed here): the default sampling points and [`spir_basis_get_uhat_full`](@ref) use the SVE functions beyond the basis * `status` - Pointer to store status code

# Returns * Pointer to basis object, or NULL on failure * Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if `k` is NULL, `statistics` is invalid, `beta`, `omega_max` or `epsilon` is not positive and finite, `epsilon` is 1 or more, `max_size` is 0, the lambda of `k` differs from `beta * omega\\_max` by more than 1e-10, `sve` is not an SVE on [-1, 1] × [-1, 1] (e.g. from [`spir_sve_result_from_matrix`](@ref) with other segments), or `sve` is NULL and the discretized kernel has a non-finite entry (e.g. a `RegularizedBoseKernel` with a tiny lambda) - [`SPIR_NOT_SUPPORTED`](@ref) (-5) if `k` is a `RegularizedBoseKernel` and `statistics` is fermionic: that kernel supports bosonic statistics only - [`SPIR_INTERNAL_ERROR`](@ref) (-7) if the SVE cannot be computed (an SVD does not converge) or an internal panic occurs

# Safety The caller must ensure `status` is a valid pointer.
"""
function spir_basis_new(statistics, beta, omega_max, epsilon, k, sve, max_size, status)
    ccall((:spir_basis_new, libsparseir), Ptr{spir_basis}, (Cint, Cdouble, Cdouble, Cdouble, Ptr{spir_kernel}, Ptr{spir_sve_result}, Cint, Ptr{StatusCode}), statistics, beta, omega_max, epsilon, k, sve, max_size, status)
end

"""
    spir_basis_new_from_sve_and_regularizer(statistics, beta, omega_max, epsilon, lambda, ypower, _conv_radius, sve, regularizer_funcs, max_size, status)

Create a finite temperature basis from SVE result and custom regularizer function

This function creates a basis from a pre-computed SVE result and a custom regularizer function. The regularizer function is used to scale the basis functions in the frequency domain.

# Arguments * `statistics` - 0 for Bosonic, 1 for Fermionic * `beta` - Inverse temperature (must be > 0) * `omega_max` - Frequency cutoff (must be > 0) * `epsilon` - Accuracy target (must be > 0) * `lambda` - Kernel parameter Λ = β * ωmax (must be > 0) * `ypower` - Power of y in kernel: 0 for `LogisticKernel`, 1 for `RegularizedBoseKernel`. Other values return [`SPIR_INVALID_ARGUMENT`](@ref). * `conv_radius` - Convergence radius for Fourier transform (currently unused) * `sve` - Pre-computed SVE result (must not be NULL) * `regularizer_funcs` - Custom regularizer function (must not be NULL) * `max_size` - Maximum basis size (-1 for no limit) * `status` - Pointer to store status code

# Returns * Pointer to basis object, or NULL on failure * Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if `sve` or `regularizer_funcs` is NULL, `statistics` or `ypower` is invalid, `beta`, `omega_max`, `epsilon` or `lambda` is not positive and finite, `epsilon` is 1 or more, `max_size` is 0, `lambda` differs from `beta * omega\\_max` by more than 1e-10, `sve` is not an SVE on [-1, 1] × [-1, 1], or `regularizer_funcs` holds τ or ω functions that are not defined at `omega\\_max / 2`, the point at which they are evaluated for validity (e.g. the u of a basis whose β is less than `omega\\_max / 2`) - [`SPIR_NOT_SUPPORTED`](@ref) (-5) if `ypower` is 1 (`RegularizedBoseKernel`) and `statistics` is fermionic: that kernel supports bosonic statistics only - [`SPIR_INTERNAL_ERROR`](@ref) (-7) if an internal panic occurs

# Note The kernel type is determined by `ypower`: 0 selects `LogisticKernel`, 1 selects `RegularizedBoseKernel`. The regularizer function is evaluated for validity but the custom weight is not yet fully integrated into basis construction.

# Safety The caller must ensure `status` is a valid pointer.
"""
function spir_basis_new_from_sve_and_regularizer(statistics, beta, omega_max, epsilon, lambda, ypower, _conv_radius, sve, regularizer_funcs, max_size, status)
    ccall((:spir_basis_new_from_sve_and_regularizer, libsparseir), Ptr{spir_basis}, (Cint, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cdouble, Ptr{spir_sve_result}, Ptr{spir_funcs}, Cint, Ptr{StatusCode}), statistics, beta, omega_max, epsilon, lambda, ypower, _conv_radius, sve, regularizer_funcs, max_size, status)
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

A DLR has no default τ sampling points: for a DLR this returns [`SPIR_COMPUTATION_SUCCESS`](@ref) with 0 points.

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or num\\_points is null * [`SPIR_NOT_SUPPORTED`](@ref) (-5) if the default points are not defined for `b`: its SVE has so few singular functions that the last one has no extrema to stand in for the roots of the missing one (e.g. an SVE from [`spir_sve_result_truncate`](@ref) with `max\\_size = 2`). Nothing is written. * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_basis_get_n_default_taus(b, num_points)
    ccall((:spir_basis_get_n_default_taus, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cint}), b, num_points)
end

"""
    spir_basis_get_default_taus(b, points)

Get default tau sampling points

The points are the roots of the first discarded basis function u\\_L (the extrema of u\\_{L-1} when u\\_L is not available), sorted and in (-β/2, β/2]; a negative point τ stands for τ + β with the sign of the statistics (see [`spir_funcs_eval`](@ref)).

# Arguments * `b` - Basis object * `points` - Pre-allocated array to store tau points

A DLR has no default τ sampling points: for a DLR this returns [`SPIR_COMPUTATION_SUCCESS`](@ref) with 0 points and writes nothing.

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or points is null * [`SPIR_NOT_SUPPORTED`](@ref) (-5) if the default points are not defined for `b`: its SVE has so few singular functions that the last one has no extrema to stand in for the roots of the missing one (e.g. an SVE from [`spir_sve_result_truncate`](@ref) with `max\\_size = 2`). Nothing is written. * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_basis_get_default_taus(b, points)
    ccall((:spir_basis_get_default_taus, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cdouble}), b, points)
end

"""
    spir_basis_get_n_default_matsus(b, positive_only, num_points)

Get the number of default Matsubara sampling points

# Arguments * `b` - Basis object * `positive_only` - If true, return only non-negative frequencies (n ≥ 0; bosonic sets include n = 0) * `num_points` - Pointer to store the number of points

A DLR has no default Matsubara sampling points: for a DLR this returns [`SPIR_COMPUTATION_SUCCESS`](@ref) with 0 points.

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or num\\_points is null * [`SPIR_NOT_SUPPORTED`](@ref) (-5) if the basis functions have no definite parity, as for a basis built on an SVE from [`spir_sve_result_from_matrix`](@ref) (not centrosymmetric): the default Matsubara sampling points are chosen by the parity of a basis function and are not defined then. Nothing is written. * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
"""
function spir_basis_get_n_default_matsus(b, positive_only, num_points)
    ccall((:spir_basis_get_n_default_matsus, libsparseir), StatusCode, (Ptr{spir_basis}, Bool, Ptr{Cint}), b, positive_only, num_points)
end

"""
    spir_basis_get_default_matsus(b, positive_only, points)

Get default Matsubara sampling points

# Arguments * `b` - Basis object * `positive_only` - If true, return only non-negative frequencies (n ≥ 0; bosonic sets include n = 0) * `points` - Pre-allocated array to store the reduced Matsubara frequencies n (iν = iπn/β)

A DLR has no default Matsubara sampling points: for a DLR this returns [`SPIR_COMPUTATION_SUCCESS`](@ref) with 0 points and writes nothing.

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or points is null * [`SPIR_NOT_SUPPORTED`](@ref) (-5) if the basis functions have no definite parity, as for a basis built on an SVE from [`spir_sve_result_from_matrix`](@ref) (not centrosymmetric): the default Matsubara sampling points are chosen by the parity of a basis function and are not defined then. Nothing is written. * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs
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

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or num\\_points is null * [`SPIR_NOT_SUPPORTED`](@ref) (-5) if the default points are not defined for `b`: its SVE has so few singular functions that the last v has no extrema to stand in for the roots of the missing one. Nothing is written. * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs

# Safety The caller must ensure that `b` and `num_points` are valid pointers
"""
function spir_basis_get_n_default_ws(b, num_points)
    ccall((:spir_basis_get_n_default_ws, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{Cint}), b, num_points)
end

"""
    spir_basis_get_default_ws(b, points)

Gets the default omega (real frequency) sampling points

# Arguments * `b` - Pointer to the finite temperature basis object * `points` - Pre-allocated array to store the omega sampling points

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if b or points is null * [`SPIR_NOT_SUPPORTED`](@ref) (-5) if the default points are not defined for `b`: its SVE has so few singular functions that the last v has no extrema to stand in for the roots of the missing one. Nothing is written. * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs

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

A DLR has no default τ sampling points: for a DLR this returns [`SPIR_COMPUTATION_SUCCESS`](@ref) with 0 points and writes nothing.

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if any pointer is null or n\\_points < 0 * [`SPIR_NOT_SUPPORTED`](@ref) (-5) if `n_points` is at least the number of singular functions of the SVE of `b` and the default points are not defined for it: the last singular function has no extrema to stand in for the roots of the missing one (e.g. an SVE from [`spir_sve_result_truncate`](@ref) with `max\\_size = 2`). Nothing is written. * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs

# Note Returns min(n\\_points, actual\\_default\\_points) sampling points
"""
function spir_basis_get_default_taus_ext(b, n_points, points, n_points_returned)
    ccall((:spir_basis_get_default_taus_ext, libsparseir), StatusCode, (Ptr{spir_basis}, Cint, Ptr{Cdouble}, Ptr{Cint}), b, n_points, points, n_points_returned)
end

"""
    spir_basis_get_n_default_matsus_ext(b, positive_only, fence, basis_size, n_points_total)

Get the number of default Matsubara sampling points for a given basis size (extended version)

# Arguments * `b` - Basis object * `positive_only` - If true, return only non-negative frequencies * `fence` - If true, add fencing points to improve conditioning * `basis_size` - Size of the basis the points are chosen for. When sampling an augmented basis, pass the augmented size; it may differ from the size of `b`. * `n_points_total` - Pointer to store the number of sampling points

A DLR has no default Matsubara sampling points: for a DLR this returns [`SPIR_COMPUTATION_SUCCESS`](@ref) with 0 points.

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if `b` or `n_points_total` is null, or `basis\\_size < 0` * [`SPIR_NOT_SUPPORTED`](@ref) (-5) if the basis functions have no definite parity, as for a basis built on an SVE from [`spir_sve_result_from_matrix`](@ref) (not centrosymmetric): the default Matsubara sampling points are chosen by the parity of a basis function and are not defined then. Nothing is written. * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs

# Note The count generally differs from `basis_size`: the parity adjustment, `fence` and `positive_only` all change it. Pass it as `points_capacity` to [`spir_basis_get_default_matsus_ext`](@ref) with the same `positive_only`, `fence` and `basis_size`.
"""
function spir_basis_get_n_default_matsus_ext(b, positive_only, fence, basis_size, n_points_total)
    ccall((:spir_basis_get_n_default_matsus_ext, libsparseir), StatusCode, (Ptr{spir_basis}, Bool, Bool, Cint, Ptr{Cint}), b, positive_only, fence, basis_size, n_points_total)
end

"""
    spir_basis_get_default_matsus_ext(b, positive_only, fence, basis_size, points_capacity, points, n_points_total)

Get default Matsubara sampling points for a given basis size (extended version)

# Arguments * `b` - Basis object * `positive_only` - If true, return only non-negative frequencies * `fence` - If true, add fencing points to improve conditioning * `basis_size` - Size of the basis the points are chosen for. When sampling an augmented basis, pass the augmented size; it may differ from the size of `b`. * `points_capacity` - Number of elements `points` can hold * `points` - Buffer for the Matsubara indices, or NULL to query the number of points only * `n_points_total` - Pointer to store the number of sampling points

A DLR has no default Matsubara sampling points: for a DLR this returns [`SPIR_COMPUTATION_SUCCESS`](@ref) with 0 points and writes nothing.

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success, including a count query (`points` is NULL, `points_capacity` is ignored) * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if `b` or `n_points_total` is null, or `basis_size` or `points_capacity` is negative; nothing is written * [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if `points_capacity` is smaller than the number of points; `points` is left untouched and `*n\\_points\\_total` is set to the required number * [`SPIR_NOT_SUPPORTED`](@ref) (-5) if the basis functions have no definite parity, as for a basis built on an SVE from [`spir_sve_result_from_matrix`](@ref) (not centrosymmetric): the default Matsubara sampling points are chosen by the parity of a basis function and are not defined then. Nothing is written. * [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs

# Note The point set depends only on `positive_only`, `fence` and `basis_size`. `points_capacity` never changes it, and the output is never truncated.
"""
function spir_basis_get_default_matsus_ext(b, positive_only, fence, basis_size, points_capacity, points, n_points_total)
    ccall((:spir_basis_get_default_matsus_ext, libsparseir), StatusCode, (Ptr{spir_basis}, Bool, Bool, Cint, Cint, Ptr{Int64}, Ptr{Cint}), b, positive_only, fence, basis_size, points_capacity, points, n_points_total)
end

"""
    spir_dlr_new(b, status)

Creates a new DLR from an IR basis with default poles

The default poles are the default real-frequency sampling points of `b` (see [`spir_basis_get_default_ws`](@ref)).

# Arguments * `b` - Pointer to a finite temperature (IR) basis object * `status` - Pointer to store the status code (may be NULL, in which case no status is written)

# Returns * Pointer to the newly created DLR basis object, or NULL on failure. The caller owns it and must release it with [`spir_basis_release`](@ref). * Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if `b` is NULL or already a DLR, or if `b` has fewer default poles than basis functions ([`spir_basis_get_n_default_ws`](@ref) < [`spir_basis_get_size`](@ref)). Root finding can lose poles, e.g. for `RegularizedBoseKernel` at large lambda; pass the poles explicitly with [`spir_dlr_new_with_poles`](@ref) instead. - [`SPIR_NOT_SUPPORTED`](@ref) (-5) if the kernel of `b` does not support its statistics (`RegularizedBoseKernel` with fermionic statistics; the basis constructors already reject this combination), or the default poles of `b` are not defined (see [`spir_basis_get_default_ws`](@ref)) - [`SPIR_INTERNAL_ERROR`](@ref) (-7) if an internal panic occurs

# Safety Caller must ensure `b` is a valid IR basis pointer
"""
function spir_dlr_new(b, status)
    ccall((:spir_dlr_new, libsparseir), Ptr{spir_basis}, (Ptr{spir_basis}, Ptr{StatusCode}), b, status)
end

"""
    spir_dlr_new_with_poles(b, npoles, poles, status)

Creates a new DLR with custom poles

# Arguments * `b` - Pointer to a finite temperature (IR) basis object * `npoles` - Number of poles to use (must be > 0) * `poles` - Array of `npoles` pole locations in [-omega\\_max, omega\\_max] of `b` * `status` - Pointer to store the status code (may be NULL, in which case no status is written)

# Returns * Pointer to the newly created DLR basis object, or NULL on failure. The caller owns it and must release it with [`spir_basis_release`](@ref). * Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if `b` or `poles` is NULL, `npoles <= 0`, or `b` is already a DLR, or a pole is outside [-omega\\_max, omega\\_max] of `b` or not finite - [`SPIR_NOT_SUPPORTED`](@ref) (-5) if the kernel of `b` does not support its statistics (`RegularizedBoseKernel` with fermionic statistics). The basis constructors already reject this combination. - [`SPIR_INTERNAL_ERROR`](@ref) (-7) if an internal panic occurs

Duplicate poles are accepted. They make [`spir_ir2dlr_dd`](@ref) and [`spir_ir2dlr_zz`](@ref) ill-conditioned: the coefficients of equal poles are not unique, although the round trip through [`spir_dlr2ir_dd`](@ref) / [`spir_dlr2ir_zz`](@ref) still recovers the IR coefficients.

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

# Arguments * `dlr` - Pointer to a DLR basis object * `order` - Memory layout order * `ndim` - Number of dimensions * `input_dims` - Array of `ndim` input dimensions, each of which must be positive * `target_dim` - Dimension to transform * `input` - IR coefficients * `out` - Output DLR coefficients

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if `dlr`, `input_dims`, `input` or `out` is null, `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)` * [`SPIR_INVALID_DIMENSION`](@ref) if an element of `input_dims` is zero or negative, or the input or output array is too large to be addressed * [`SPIR_INPUT_DIMENSION_MISMATCH`](@ref) if `input\\_dims[target\\_dim]` is not the size of the IR basis of `dlr` * [`SPIR_NOT_SUPPORTED`](@ref) if `dlr` is not a DLR basis * [`SPIR_INTERNAL_ERROR`](@ref) if an internal panic occurs

`input_dims` is validated before `input` or `out` is accessed.

# Safety Caller must ensure pointers are valid and arrays have correct sizes
"""
function spir_ir2dlr_dd(dlr, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_ir2dlr_dd, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), dlr, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_ir2dlr_zz(dlr, backend, order, ndim, input_dims, target_dim, input, out)

Convert IR coefficients to DLR (complex-valued)

# Arguments * `dlr` - Pointer to a DLR basis object * `order` - Memory layout order * `ndim` - Number of dimensions * `input_dims` - Array of `ndim` input dimensions, each of which must be positive * `target_dim` - Dimension to transform * `input` - Complex IR coefficients * `out` - Output complex DLR coefficients

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if `dlr`, `input_dims`, `input` or `out` is null, `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)` * [`SPIR_INVALID_DIMENSION`](@ref) if an element of `input_dims` is zero or negative, or the input or output array is too large to be addressed * [`SPIR_INPUT_DIMENSION_MISMATCH`](@ref) if `input\\_dims[target\\_dim]` is not the size of the IR basis of `dlr` * [`SPIR_NOT_SUPPORTED`](@ref) if `dlr` is not a DLR basis * [`SPIR_INTERNAL_ERROR`](@ref) if an internal panic occurs

`input_dims` is validated before `input` or `out` is accessed.

# Safety Caller must ensure pointers are valid and arrays have correct sizes
"""
function spir_ir2dlr_zz(dlr, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_ir2dlr_zz, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Complex64}, Ptr{Complex64}), dlr, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_dlr2ir_dd(dlr, backend, order, ndim, input_dims, target_dim, input, out)

Convert DLR coefficients to IR (real-valued)

# Arguments * `dlr` - Pointer to a DLR basis object * `order` - Memory layout order * `ndim` - Number of dimensions * `input_dims` - Array of `ndim` input dimensions, each of which must be positive * `target_dim` - Dimension to transform * `input` - DLR coefficients * `out` - Output IR coefficients

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if `dlr`, `input_dims`, `input` or `out` is null, `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)` * [`SPIR_INVALID_DIMENSION`](@ref) if an element of `input_dims` is zero or negative, or the input or output array is too large to be addressed * [`SPIR_INPUT_DIMENSION_MISMATCH`](@ref) if `input\\_dims[target\\_dim]` is not the number of poles of `dlr` ([`spir_dlr_get_npoles`](@ref)) * [`SPIR_NOT_SUPPORTED`](@ref) if `dlr` is not a DLR basis * [`SPIR_INTERNAL_ERROR`](@ref) if an internal panic occurs

`input_dims` is validated before `input` or `out` is accessed.

# Safety Caller must ensure pointers are valid and arrays have correct sizes
"""
function spir_dlr2ir_dd(dlr, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_dlr2ir_dd, libsparseir), StatusCode, (Ptr{spir_basis}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), dlr, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_dlr2ir_zz(dlr, backend, order, ndim, input_dims, target_dim, input, out)

Convert DLR coefficients to IR (complex-valued)

# Arguments * `dlr` - Pointer to a DLR basis object * `order` - Memory layout order * `ndim` - Number of dimensions * `input_dims` - Array of `ndim` input dimensions, each of which must be positive * `target_dim` - Dimension to transform * `input` - Complex DLR coefficients * `out` - Output complex IR coefficients

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if `dlr`, `input_dims`, `input` or `out` is null, `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)` * [`SPIR_INVALID_DIMENSION`](@ref) if an element of `input_dims` is zero or negative, or the input or output array is too large to be addressed * [`SPIR_INPUT_DIMENSION_MISMATCH`](@ref) if `input\\_dims[target\\_dim]` is not the number of poles of `dlr` ([`spir_dlr_get_npoles`](@ref)) * [`SPIR_NOT_SUPPORTED`](@ref) if `dlr` is not a DLR basis * [`SPIR_INTERNAL_ERROR`](@ref) if an internal panic occurs

`input_dims` is validated before `input` or `out` is accessed.

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

Check if the funcs pointer is non-null.

Note: This only performs a null check. It cannot detect dangling pointers; dereferencing an arbitrary non-null pointer would be undefined behaviour that `catch_unwind` cannot reliably catch.

# Returns 1 if the pointer is non-null, 0 otherwise
"""
function spir_funcs_is_assigned(obj)
    ccall((:spir_funcs_is_assigned, libsparseir), Int32, (Ptr{spir_funcs},), obj)
end

"""
    spir_funcs_deriv(funcs, n, status)

Compute the n-th derivative of basis functions

Creates a new funcs object representing the n-th derivative of the input functions. For n=0, returns a clone of the input. For n=1, returns the first derivative, etc.

# Arguments * `funcs` - Pointer to the input funcs object * `n` - Order of derivative (0 = no derivative, 1 = first derivative, etc.) * `status` - Pointer to store the status code

# Returns Pointer to the newly created derivative funcs object, or NULL if computation fails

# Safety Caller must ensure `funcs` is a valid pointer and `status` is non-null
"""
function spir_funcs_deriv(funcs, n, status)
    ccall((:spir_funcs_deriv, libsparseir), Ptr{spir_funcs}, (Ptr{spir_funcs}, Cint, Ptr{StatusCode}), funcs, n, status)
end

"""
    spir_funcs_from_piecewise_legendre(segments, n_segments, coeffs, nfuncs, _order, status)

Create a [`spir_funcs`](@ref) object from piecewise Legendre polynomial coefficients

Constructs a continuous function object from segments and Legendre polynomial expansion coefficients. The coefficients are organized per segment, with each segment containing nfuncs coefficients (degrees 0 to nfuncs-1).

# Arguments * `segments` - Array of the `n\\_segments + 1` segment boundaries: finite and strictly increasing, with every segment length `segments[i + 1] - segments[i]` a normal double (finite and at least `DBL_MIN`) * `n_segments` - Number of segments (must be >= 1) * `coeffs` - Array of Legendre coefficients. Layout: contiguous per segment, coefficients for segment i are stored at indices [i*nfuncs, (i+1)*nfuncs). Each segment has nfuncs coefficients for Legendre degrees 0 to nfuncs-1. * `nfuncs` - Number of basis functions per segment (Legendre polynomial degrees 0 to nfuncs-1) * `order` - Order parameter (currently unused, reserved for future use) * `status` - Pointer to store the status code

# Returns Pointer to the newly created funcs object, or NULL if creation fails. If `status` is non-NULL, `*status` is set to: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `segments` or `coeffs` is NULL, `n_segments` or `nfuncs` < 1, or `segments` does not meet the conditions above - [`SPIR_INVALID_DIMENSION`](@ref) if `n_segments` is `INT_MAX` (the number of knots, `n\\_segments + 1`, must fit in an `int`), or `segments` or `coeffs` is too large to be addressed - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

Nothing is written when `status` is NULL. The sizes are validated before `segments` or `coeffs` is read.

# Note The function creates a single piecewise Legendre polynomial function. To create multiple functions, call this function multiple times.
"""
function spir_funcs_from_piecewise_legendre(segments, n_segments, coeffs, nfuncs, _order, status)
    ccall((:spir_funcs_from_piecewise_legendre, libsparseir), Ptr{spir_funcs}, (Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Ptr{StatusCode}), segments, n_segments, coeffs, nfuncs, _order, status)
end

"""
    spir_funcs_get_slice(funcs, nslice, indices, status)

Extract a subset of functions by indices

The new object holds the selected functions in the order given by `indices`, for every function type (τ, ω, Matsubara and DLR functions).

# Arguments * `funcs` - Pointer to the source funcs object * `nslice` - Number of functions to select (length of `indices`), at least 1 * `indices` - Array of `nslice` distinct 0-based indices, each in `[0, size)` where `size` is given by `[`spir_funcs_get_size`](@ref)(funcs)` * `status` - Pointer to store the status code

# Returns Pointer to a new funcs object containing only the selected functions, or NULL on error. `*status` is set to: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `funcs` or `indices` is NULL, if `nslice` < 1 (an empty selection is rejected for every function type), or if an index is negative, not less than `size`, or repeated - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

Nothing is written when `status` is NULL.

# Safety The caller must ensure that `funcs` and `indices` are valid pointers and that `indices` holds at least `nslice` elements. The returned pointer must be freed with `[`spir_funcs_release`](@ref)()`.
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

The valid points depend on the functions: - τ functions (`u` of an IR or DLR basis, and their slices and derivatives): τ ∈ [-β, β]. A negative τ is folded onto [0, β] by the (anti)periodicity: u(τ) = -u(τ + β) for fermions and u(τ) = u(τ + β) for bosons. τ = +0.0 is read as 0⁺, τ = β as β⁻, τ = -β as (-β)⁺ (folded onto 0⁺), and τ = -0.0 as 0⁻ (folded onto β⁻). - ω functions (`v` of an IR basis, and functions from [`spir_funcs_from_piecewise_legendre`](@ref)): ω from the first to the last knot (see [`spir_funcs_get_knots`](@ref)), i.e. ω ∈ [-ωmax, ωmax] for `v`.

# Arguments * `funcs` - Pointer to the funcs object * `x` - Point to evaluate at: τ or ω in the domain above * `out` - Pre-allocated array to store function values

# Returns Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `funcs` or `out` is NULL, or `x` is NaN, infinite or outside the domain; `out` is not written - [`SPIR_NOT_SUPPORTED`](@ref) if `funcs` holds Matsubara-frequency functions - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

# Safety The caller must ensure that `out` has size >= `[`spir_funcs_get_size`](@ref)(funcs)`
"""
function spir_funcs_eval(funcs, x, out)
    ccall((:spir_funcs_eval, libsparseir), StatusCode, (Ptr{spir_funcs}, Cdouble, Ptr{Cdouble}), funcs, x, out)
end

"""
    spir_funcs_eval_matsu(funcs, n, out)

Evaluate functions at a single Matsubara frequency

# Arguments * `funcs` - Pointer to the funcs object * `n` - Reduced Matsubara frequency n (iν = iπn/β): odd for fermionic, even for bosonic functions * `out` - Pre-allocated array to store complex function values

# Returns Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `funcs` or `out` is NULL, or `n` has the wrong parity for the statistics of `funcs`; `out` is not written - [`SPIR_NOT_SUPPORTED`](@ref) if `funcs` does not hold Matsubara-frequency functions - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

# Safety The caller must ensure that `out` has size >= `[`spir_funcs_get_size`](@ref)(funcs)` Complex numbers are laid out as [real, imag] pairs
"""
function spir_funcs_eval_matsu(funcs, n, out)
    ccall((:spir_funcs_eval_matsu, libsparseir), StatusCode, (Ptr{spir_funcs}, Int64, Ptr{Complex64}), funcs, n, out)
end

"""
    spir_funcs_batch_eval(funcs, order, num_points, xs, out)

Batch evaluate functions at multiple points (continuous functions only)

Every point must lie in the domain described for [`spir_funcs_eval`](@ref).

# Arguments * `funcs` - Pointer to the funcs object * `order` - Memory layout of `out`: [`SPIR_ORDER_ROW_MAJOR`](@ref) (0) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref) (1) * `num_points` - Number of evaluation points * `xs` - Array of points to evaluate at, in the units and domain of `x` in [`spir_funcs_eval`](@ref) * `out` - Pre-allocated array to store results

# Returns Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `funcs`, `xs` or `out` is NULL, `num_points` <= 0, `order` is not one of the constants above, or any point is NaN, infinite or outside the domain; `out` is not written - [`SPIR_NOT_SUPPORTED`](@ref) if `funcs` holds Matsubara-frequency functions - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

# Safety - `xs` must have size >= `num_points` - `out` must have size >= `num\\_points * [`spir_funcs_get_size`](@ref)(funcs)` - Layout: row-major = out\\[point\\]\\[func\\], column-major = out\\[func\\]\\[point\\]
"""
function spir_funcs_batch_eval(funcs, order, num_points, xs, out)
    ccall((:spir_funcs_batch_eval, libsparseir), StatusCode, (Ptr{spir_funcs}, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}), funcs, order, num_points, xs, out)
end

"""
    spir_funcs_batch_eval_matsu(funcs, order, num_freqs, ns, out)

Batch evaluate functions at multiple Matsubara frequencies

# Arguments * `funcs` - Pointer to the funcs object * `order` - Memory layout of `out`: [`SPIR_ORDER_ROW_MAJOR`](@ref) (0) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref) (1) * `num_freqs` - Number of Matsubara frequencies * `ns` - Array of reduced Matsubara frequencies n (iν = iπn/β): odd for fermionic, even for bosonic functions * `out` - Pre-allocated array to store complex results

# Returns Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `funcs`, `ns` or `out` is NULL, `num_freqs` <= 0, `order` is not one of the constants above, or any index has the wrong parity for the statistics of `funcs`; `out` is not written - [`SPIR_NOT_SUPPORTED`](@ref) if `funcs` does not hold Matsubara-frequency functions - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

# Safety - `ns` must have size >= `num_freqs` - `out` must have size >= `num\\_freqs * [`spir_funcs_get_size`](@ref)(funcs)` - Complex numbers are laid out as [real, imag] pairs - Layout: row-major = out\\[freq\\]\\[func\\], column-major = out\\[func\\]\\[freq\\]
"""
function spir_funcs_batch_eval_matsu(funcs, order, num_freqs, ns, out)
    ccall((:spir_funcs_batch_eval_matsu, libsparseir), StatusCode, (Ptr{spir_funcs}, Cint, Cint, Ptr{Int64}, Ptr{Complex64}), funcs, order, num_freqs, ns, out)
end

"""
    spir_uhat_get_default_matsus(uhat, positive_only, fence, basis_size, points_capacity, points, n_points_total)

Get default Matsubara sampling points from a Matsubara-space [`spir_funcs`](@ref)

This function computes default sampling points in Matsubara frequencies (iν) from a [`spir_funcs`](@ref) object that represents Matsubara-space basis functions (e.g., uhat or uhat\\_full). The statistics type (Fermionic/Bosonic) is automatically detected from the [`spir_funcs`](@ref) object type.

This extracts the PiecewiseLegendreFTVector from [`spir_funcs`](@ref) and calls `sparse\\_ir::basis::default\\_matsubara\\_sampling\\_points\\_from\\_uhat` to compute default sampling points: the sign changes of the first discarded Matsubara basis function (its extrema when that function is not available); bosonic sets always include n = 0.

# Arguments * `uhat` - Pointer to a [`spir_funcs`](@ref) object representing Matsubara-space basis functions * `positive_only` - If true, return only non-negative frequencies * `fence` - If true, add fencing points to improve conditioning * `basis_size` - Size of the basis the points are chosen for * `points_capacity` - Number of elements `points` can hold * `points` - Buffer for the Matsubara indices, or NULL to query the number of points only * `n_points_total` - Pointer to store the number of sampling points

# Returns Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success, including a count query (`points` is NULL, `points_capacity` is ignored) - [`SPIR_INVALID_ARGUMENT`](@ref) if uhat or n\\_points\\_total is null, or basis\\_size or points\\_capacity is negative; nothing is written - [`SPIR_INVALID_ARGUMENT`](@ref) if points\\_capacity is smaller than the number of points; `points` is left untouched and `*n\\_points\\_total` is set to the required number - [`SPIR_NOT_SUPPORTED`](@ref) if uhat is not a Matsubara-space function, or if its functions have no definite parity, as those of a basis built on an SVE from [`spir_sve_result_from_matrix`](@ref) (not centrosymmetric): the default points are chosen by the parity of a basis function and are not defined then. Nothing is written.

# Note This function is only available for [`spir_funcs`](@ref) objects representing Matsubara-space basis functions The statistics type is automatically detected from the [`spir_funcs`](@ref) object type The default sampling points are chosen to provide near-optimal conditioning The number of points generally differs from `basis_size`: the parity adjustment, `fence` and `positive_only` all change it. The point set never depends on `points_capacity`, and the output is never truncated.
"""
function spir_uhat_get_default_matsus(uhat, positive_only, fence, basis_size, points_capacity, points, n_points_total)
    ccall((:spir_uhat_get_default_matsus, libsparseir), StatusCode, (Ptr{spir_funcs}, Bool, Bool, Cint, Cint, Ptr{Int64}, Ptr{Cint}), uhat, positive_only, fence, basis_size, points_capacity, points, n_points_total)
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

# Deprecated Use [`spir_logistic_kernel_new`](@ref), the default kernel for both statistics. `RegularizedBoseKernel` will be removed in a future release (https://github.com/SpM-lab/sparse-ir-rs/issues/273).

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

Check if the kernel pointer is non-null.

Note: This only performs a null check. It cannot detect dangling pointers; dereferencing an arbitrary non-null pointer would be undefined behaviour that `catch_unwind` cannot reliably catch.

# Returns 1 if the pointer is non-null, 0 otherwise
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

This function should be called twice: 1. First call with segments=NULL: sets `*n\\_segments` to the number of segment intervals. 2. Second call with segments allocated: fills `segments[0..n\\_segments]` with boundary points (`n\\_segments + 1` values total). The caller must allocate at least `n\\_segments + 1` elements.

# Arguments * `k` - Kernel object * `epsilon` - Accuracy target for the basis * `segments` - Pointer to store segments array (NULL for first call) * `n_segments` - [IN/OUT] Input: ignored when segments is NULL. Output: number of segment intervals

# Returns * [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if k or n\\_segments is null, or segments array is too small * [`SPIR_INTERNAL_ERROR`](@ref) if internal panic occurs
"""
function spir_kernel_get_sve_hints_segments_x(k, epsilon, segments, n_segments)
    ccall((:spir_kernel_get_sve_hints_segments_x, libsparseir), StatusCode, (Ptr{spir_kernel}, Cdouble, Ptr{Cdouble}, Ptr{Cint}), k, epsilon, segments, n_segments)
end

"""
    spir_kernel_get_sve_hints_segments_y(k, epsilon, segments, n_segments)

Get y-segments for SVE discretization hints from a kernel

This function should be called twice: 1. First call with segments=NULL: sets `*n\\_segments` to the number of segment intervals. 2. Second call with segments allocated: fills `segments[0..n\\_segments]` with boundary points (`n\\_segments + 1` values total). The caller must allocate at least `n\\_segments + 1` elements.

# Arguments * `k` - Kernel object * `epsilon` - Accuracy target for the basis * `segments` - Pointer to store segments array (NULL for first call) * `n_segments` - [IN/OUT] Input: ignored when segments is NULL. Output: number of segment intervals

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

Check if the sampling pointer is non-null.

Note: This only performs a null check. It cannot detect dangling pointers; dereferencing an arbitrary non-null pointer would be undefined behaviour that `catch_unwind` cannot reliably catch.

# Returns 1 if the pointer is non-null, 0 otherwise
"""
function spir_sampling_is_assigned(obj)
    ccall((:spir_sampling_is_assigned, libsparseir), Int32, (Ptr{spir_sampling},), obj)
end

"""
    spir_tau_sampling_new(b, num_points, points, status)

Creates a new tau sampling object for sparse sampling in imaginary time

# Arguments * `b` - Pointer to a finite temperature basis object * `num_points` - Number of sampling points * `points` - Array of `num_points` sampling points τ ∈ [-β, β], with β the inverse temperature of `b`; a negative τ is folded onto [0, β] as in [`spir_funcs_eval`](@ref) * `status` - Pointer to store the status code

# Returns Pointer to the newly created sampling object, or NULL if creation fails. If `status` is non-NULL, `*status` is set to: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `b` or `points` is NULL, `num_points` <= 0, or a point is NaN, infinite or outside [-β, β] - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

The points may be in any order, and the sampling object keeps it: [`spir_sampling_get_taus`](@ref) returns `points` unchanged, and index i along the sampling-point axis of the evaluate and fit functions refers to `points[i]`.

# Safety Caller must ensure `b` is valid and `points` has `num_points` elements
"""
function spir_tau_sampling_new(b, num_points, points, status)
    ccall((:spir_tau_sampling_new, libsparseir), Ptr{spir_sampling}, (Ptr{spir_basis}, Cint, Ptr{Cdouble}, Ptr{StatusCode}), b, num_points, points, status)
end

"""
    spir_matsu_sampling_new(b, positive_only, num_points, points, status)

Creates a new Matsubara sampling object for sparse sampling in Matsubara frequencies

# Arguments * `b` - Pointer to a finite temperature basis object * `positive_only` - If true, only non-negative frequencies are used; the IR coefficients are then real, i.e. G(-iν) = conj(G(iν)) * `num_points` - Number of sampling points * `points` - Array of `num_points` reduced Matsubara frequencies n (iν = iπn/β): odd for a fermionic basis, even for a bosonic basis, and non-negative when `positive_only` is true * `status` - Pointer to store the status code

# Returns Pointer to the newly created sampling object, or NULL if creation fails. If `status` is non-NULL, `*status` is set to: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `b` or `points` is NULL, `num_points` <= 0, an index has the wrong parity for the statistics of `b`, or `positive_only` is true and an index is negative - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

The points may be in any order, and the sampling object keeps it: they are not sorted, [`spir_sampling_get_matsus`](@ref) returns `points` unchanged, and index i along the sampling-point axis of the evaluate and fit functions refers to `points[i]`.
"""
function spir_matsu_sampling_new(b, positive_only, num_points, points, status)
    ccall((:spir_matsu_sampling_new, libsparseir), Ptr{spir_sampling}, (Ptr{spir_basis}, Bool, Cint, Ptr{Int64}, Ptr{StatusCode}), b, positive_only, num_points, points, status)
end

"""
    spir_tau_sampling_new_with_matrix(order, statistics, basis_size, num_points, points, matrix, status)

Creates a new tau sampling object with custom sampling points and pre-computed matrix

# Arguments * `order` - Memory layout of `matrix` ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `statistics` - Statistics type ([`SPIR_STATISTICS_FERMIONIC`](@ref) or [`SPIR_STATISTICS_BOSONIC`](@ref)) * `basis_size` - Basis size (the number of columns of `matrix`) * `num_points` - Number of sampling points (the number of rows of `matrix`) * `points` - Array of `num_points` finite sampling points in imaginary time (τ), one per row of `matrix`. Without β the domain [-β, β] of [`spir_tau_sampling_new`](@ref) cannot be checked here: any finite value is accepted and reported back by [`spir_sampling_get_taus`](@ref) * `matrix` - Pre-computed `num\\_points × basis\\_size` sampling matrix in `order`, with finite entries * `status` - Pointer to store the status code

# Returns Pointer to the newly created sampling object, or NULL if creation fails. If `status` is non-NULL, `*status` is set to: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `points` or `matrix` is NULL, `num_points` or `basis_size` <= 0, `order` or `statistics` is not one of the constants above, a point is NaN or infinite, or an entry of `matrix` is NaN or infinite - [`SPIR_INVALID_DIMENSION`](@ref) if the matrix is too large to be addressed - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

The scalar arguments are validated before `points` or `matrix` is read.

The points may be in any order, and the sampling object keeps it: [`spir_sampling_get_taus`](@ref) returns `points` unchanged, and index i along the sampling-point axis of the evaluate and fit functions refers to `points[i]`, the point of row i of `matrix`.

# Safety Caller must ensure `points` and `matrix` have correct sizes
"""
function spir_tau_sampling_new_with_matrix(order, statistics, basis_size, num_points, points, matrix, status)
    ccall((:spir_tau_sampling_new_with_matrix, libsparseir), Ptr{spir_sampling}, (Cint, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{StatusCode}), order, statistics, basis_size, num_points, points, matrix, status)
end

"""
    spir_matsu_sampling_new_with_matrix(order, statistics, basis_size, positive_only, num_points, points, matrix, status)

Creates a new Matsubara sampling object with custom sampling points and pre-computed matrix

# Arguments * `order` - Memory layout of `matrix` ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `statistics` - Statistics type ([`SPIR_STATISTICS_FERMIONIC`](@ref) or [`SPIR_STATISTICS_BOSONIC`](@ref)) * `basis_size` - Basis size (the number of columns of `matrix`) * `positive_only` - If true, only non-negative frequencies are used; the IR coefficients are then real, i.e. G(-iν) = conj(G(iν)) * `num_points` - Number of sampling points (the number of rows of `matrix`) * `points` - Array of `num_points` reduced Matsubara frequencies n (iν = iπn/β): odd for fermionic, even for bosonic `statistics`, and non-negative when `positive_only` is true * `matrix` - Pre-computed complex `num\\_points × basis\\_size` sampling matrix in `order`, with finite real and imaginary parts * `status` - Pointer to store the status code

# Returns Pointer to the newly created sampling object, or NULL if creation fails. If `status` is non-NULL, `*status` is set to: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `points` or `matrix` is NULL, `num_points` or `basis_size` <= 0, `order` or `statistics` is not one of the constants above, an index has the wrong parity for `statistics`, `positive_only` is true and an index is negative, or an entry of `matrix` has a NaN or infinite part - [`SPIR_INVALID_DIMENSION`](@ref) if the matrix is too large to be addressed - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

The scalar arguments are validated before `points` or `matrix` is read, and the indices before `matrix` is read.

The points may be in any order, and the sampling object keeps it: [`spir_sampling_get_matsus`](@ref) returns `points` unchanged, and index i along the sampling-point axis of the evaluate and fit functions refers to `points[i]`, the point of row i of `matrix`.

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

Gets the condition number of the least-squares problem that fitting solves.

Stores in `*cond\\_num` the ratio σ\\_max / σ\\_min of the largest to the smallest of the min(rows, columns) singular values of the matrix that the fit functions ([`spir_sampling_fit_dd`](@ref), [`spir_sampling_fit_zz`](@ref), [`spir_sampling_fit_zd`](@ref)) solve with. Let `A` be the `n\\_points × basis\\_size` sampling matrix: `A[i, l]` is basis function `l` at sampling point `i`, or `A` is the matrix passed to [`spir_tau_sampling_new_with_matrix`](@ref) or [`spir_matsu_sampling_new_with_matrix`](@ref). The matrix the fit solves with is:

- τ sampling: the real matrix `A`. - Matsubara sampling with `positive\\_only = false`: the complex matrix `A`. - Matsubara sampling with `positive\\_only = true`: the real `2 n\\_points × basis\\_size` matrix `[Re A; Im A]` of the real least-squares problem `[Re A; Im A] x = [Re g; Im g]` that the fit solves for real coefficients `x`. This is not the condition number of the complex matrix `A`: with `n\\_points ≈ basis\\_size / 2`, `A` is wide, and its condition number can understate the error amplification of the fit by orders of magnitude.

The value bounds how much fitting can amplify relative errors in the values.

# Parameters - `s`: Pointer to the sampling object. - `cond_num`: Pointer to store the condition number.

# Returns An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `s` or `cond_num` is null; `*cond\\_num` is not written - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs

# Notes - A large condition number indicates that the sampling problem is ill-conditioned, which may lead to numerical instability in fitting. - `+inf` is stored if the smallest singular value is below 1e-15 (numerically singular matrix). - The singular value decomposition is the one the fit functions use: it is computed once per sampling object (shared with its clones), by the first call to this function or to a fit function, and then reused.
"""
function spir_sampling_get_cond_num(s, cond_num)
    ccall((:spir_sampling_get_cond_num, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{Cdouble}), s, cond_num)
end

"""
    spir_sampling_eval_dd(s, backend, order, ndim, input_dims, target_dim, input, out)

Evaluates basis coefficients at sampling points (double to double version).

Transforms basis coefficients to values at sampling points, where both input and output are real (double precision) values. The operation can be performed along any dimension of a multidimensional array.

# Arguments

* `s` - Pointer to the sampling object * `order` - Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `ndim` - Number of dimensions in the input/output arrays * `input_dims` - Array of `ndim` dimension sizes, each of which must be positive * `target_dim` - Target dimension for the transformation (0-based) * `input` - Input array of basis coefficients * `out` - Output array for the evaluated values at sampling points

# Returns

An integer status code: - `0` ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `s`, `input_dims`, `input` or `out` is null, `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)` - [`SPIR_INVALID_DIMENSION`](@ref) if an element of `input_dims` is zero or negative, or the input or output array is too large to be addressed - [`SPIR_INPUT_DIMENSION_MISMATCH`](@ref) if `input\\_dims[target\\_dim]` is not the basis size - [`SPIR_NOT_SUPPORTED`](@ref) if the sampling type does not support this operation - [`SPIR_INTERNAL_ERROR`](@ref) if an internal panic occurs

All shape arguments are validated before `input` or `out` is accessed.

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

For Matsubara sampling: transforms real IR coefficients to complex values. Zero-copy implementation. Arguments are as for [[`spir_sampling_eval_dd`](@ref)].

# Returns

- [`SPIR_COMPUTATION_SUCCESS`](@ref) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `s`, `input_dims`, `input` or `out` is null, `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)` - [`SPIR_INVALID_DIMENSION`](@ref) if an element of `input_dims` is zero or negative, or the input or output array is too large to be addressed - [`SPIR_INPUT_DIMENSION_MISMATCH`](@ref) if `input\\_dims[target\\_dim]` is not the basis size - [`SPIR_NOT_SUPPORTED`](@ref) if the sampling type does not support this operation - [`SPIR_INTERNAL_ERROR`](@ref) if an internal panic occurs

All shape arguments are validated before `input` or `out` is accessed.
"""
function spir_sampling_eval_dz(s, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_dz, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Complex64}), s, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_eval_zz(s, backend, order, ndim, input_dims, target_dim, input, out)

Evaluate basis coefficients at sampling points (complex → complex)

For Matsubara sampling: transforms complex coefficients to complex values. Zero-copy implementation. Arguments are as for [[`spir_sampling_eval_dd`](@ref)].

# Returns

- [`SPIR_COMPUTATION_SUCCESS`](@ref) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `s`, `input_dims`, `input` or `out` is null, `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)` - [`SPIR_INVALID_DIMENSION`](@ref) if an element of `input_dims` is zero or negative, or the input or output array is too large to be addressed - [`SPIR_INPUT_DIMENSION_MISMATCH`](@ref) if `input\\_dims[target\\_dim]` is not the basis size - [`SPIR_NOT_SUPPORTED`](@ref) if the sampling type does not support this operation - [`SPIR_INTERNAL_ERROR`](@ref) if an internal panic occurs

All shape arguments are validated before `input` or `out` is accessed.
"""
function spir_sampling_eval_zz(s, backend, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_zz, libsparseir), StatusCode, (Ptr{spir_sampling}, Ptr{spir_gemm_backend}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Complex64}, Ptr{Complex64}), s, backend, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_fit_dd(s, backend, order, ndim, input_dims, target_dim, input, out)

Fits values at sampling points to basis coefficients (double to double version).

Transforms values at sampling points back to basis coefficients, where both input and output are real (double precision) values. The operation can be performed along any dimension of a multidimensional array.

# Arguments

* `s` - Pointer to the sampling object * `backend` - Pointer to the GEMM backend (can be null to use default) * `order` - Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `ndim` - Number of dimensions in the input/output arrays * `input_dims` - Array of `ndim` dimension sizes, each of which must be positive * `target_dim` - Target dimension for the transformation (0-based) * `input` - Input array of values at sampling points * `out` - Output array for the fitted basis coefficients

# Returns

An integer status code: * `0` ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if `s`, `input_dims`, `input` or `out` is null, `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)` * [`SPIR_INVALID_DIMENSION`](@ref) if an element of `input_dims` is zero or negative, or the input or output array is too large to be addressed * [`SPIR_INPUT_DIMENSION_MISMATCH`](@ref) if `input\\_dims[target\\_dim]` is not the number of sampling points * [`SPIR_NOT_SUPPORTED`](@ref) if the sampling type does not support this operation * [`SPIR_INTERNAL_ERROR`](@ref) if an internal panic occurs

All shape arguments are validated before `input` or `out` is accessed.

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

# Returns

* [`SPIR_COMPUTATION_SUCCESS`](@ref) on success * [`SPIR_INVALID_ARGUMENT`](@ref) if `s`, `input_dims`, `input` or `out` is null, `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)` * [`SPIR_INVALID_DIMENSION`](@ref) if an element of `input_dims` is zero or negative, or the input or output array is too large to be addressed * [`SPIR_INPUT_DIMENSION_MISMATCH`](@ref) if `input\\_dims[target\\_dim]` is not the number of sampling points * [`SPIR_NOT_SUPPORTED`](@ref) if the sampling type does not support this operation * [`SPIR_INTERNAL_ERROR`](@ref) if an internal panic occurs

All shape arguments are validated before `input` or `out` is accessed.
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

* `s` - Pointer to the sampling object (must be Matsubara) * `backend` - Pointer to the GEMM backend (can be null to use default) * `order` - Memory layout order ([`SPIR_ORDER_COLUMN_MAJOR`](@ref) or [`SPIR_ORDER_ROW_MAJOR`](@ref)) * `ndim` - Number of dimensions in the input/output arrays * `input_dims` - Array of `ndim` dimension sizes, each of which must be positive * `target_dim` - Target dimension for the transformation (0-based) * `input` - Input array (complex) * `out` - Output array (real)

# Returns

- [`SPIR_COMPUTATION_SUCCESS`](@ref) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `s`, `input_dims`, `input` or `out` is null, `order` is invalid, `ndim < 1`, or `target_dim` is not in `[0, ndim)` - [`SPIR_INVALID_DIMENSION`](@ref) if an element of `input_dims` is zero or negative, or the input or output array is too large to be addressed - [`SPIR_INPUT_DIMENSION_MISMATCH`](@ref) if `input\\_dims[target\\_dim]` is not the number of sampling points - [`SPIR_NOT_SUPPORTED`](@ref) if the sampling type doesn't support this operation - [`SPIR_INTERNAL_ERROR`](@ref) if an internal panic occurs

All shape arguments are validated before `input` or `out` is accessed.

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

Check if the SVE result pointer is non-null.

Note: This only performs a null check. It cannot detect dangling pointers; dereferencing an arbitrary non-null pointer would be undefined behaviour that `catch_unwind` cannot reliably catch.

# Returns 1 if the pointer is non-null, 0 otherwise
"""
function spir_sve_result_is_assigned(obj)
    ccall((:spir_sve_result_is_assigned, libsparseir), Int32, (Ptr{spir_sve_result},), obj)
end

"""
    spir_sve_result_new(k, epsilon, _lmax, _n_gauss, twork, status)

Compute Singular Value Expansion (SVE) of a kernel (libsparseir compatible)

# Arguments * `k` - Kernel object * `epsilon` - Accuracy target for the basis * `lmax` - Maximum number of Legendre polynomials (currently ignored, auto-determined) * `n_gauss` - Number of Gauss points for integration (currently ignored, auto-determined) * `Twork` - Working precision: 0=Float64, 1=Float64x2, -1=Auto * `status` - Pointer to store status code

# Returns * Pointer to SVE result, or NULL on failure * Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if `k` is NULL, `epsilon` is not positive and finite or is 1 or more, `Twork` is invalid, or the discretized kernel has a NaN or infinite entry (e.g. a `RegularizedBoseKernel` whose lambda is so small that 1/lambda overflows) - [`SPIR_INTERNAL_ERROR`](@ref) (-7) if an SVD does not converge or an internal panic occurs

# Safety The caller must ensure `status` is a valid pointer.

# Note Parameters `lmax` and `n_gauss` are accepted for libsparseir compatibility but currently ignored. The Rust implementation automatically determines optimal values. The singular value truncation cutoff is automatically set to 2 * machine epsilon of the working precision (about 4.44e-16 for Float64 and 4.93e-32 for Float64x2), as in libsparseir: singular values smaller than this cutoff times the largest singular value are discarded.
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

This function creates a new SVE result containing only the singular values that are larger than `epsilon * s\\[0\\]`, where `s\\[0\\]` is the largest singular value. The result can also be limited to a maximum size.

# Arguments * `sve` - Source SVE result object * `epsilon` - Relative threshold for truncation (singular values < epsilon * s\\[0\\] are removed) * `max_size` - Maximum number of singular values to keep (-1 for no limit) * `status` - Pointer to store status code

# Returns * Pointer to new truncated SVE result, or NULL on failure * Status code: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) (-6) if `sve` is NULL, `epsilon` is not finite, negative or 1 or more (0 keeps every singular value), or `max_size` is 0 - [`SPIR_INTERNAL_ERROR`](@ref) (-7) if internal panic occurs

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

# Arguments * `K_high` - High part of the kernel matrix (required, size: nx * ny, finite entries) * `K_low` - Low part of the kernel matrix (optional, nullptr for double precision; finite entries) * `nx` - Number of rows in the matrix (must be `n\\_segments\\_x * n\\_gauss`, the number of Gauss points of the segments) * `ny` - Number of columns in the matrix (must be `n\\_segments\\_y * n\\_gauss`, the number of Gauss points of the segments) * `order` - Memory layout ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `segments_x` - X-direction segments (array of boundary points, size: n\\_segments\\_x + 1, finite and strictly increasing) * `n_segments_x` - Number of segments in x direction (boundary points - 1) * `segments_y` - Y-direction segments (array of boundary points, size: n\\_segments\\_y + 1, finite and strictly increasing) * `n_segments_y` - Number of segments in y direction (boundary points - 1) * `n_gauss` - Number of Gauss points per segment * `epsilon` - Target accuracy * `status` - Pointer to store status code

# Returns Pointer to SVE result on success, nullptr on failure. If `status` is non-NULL, `*status` is set to: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `K_high`, `segments_x` or `segments_y` is NULL, a size is less than 1, `epsilon` is not positive and finite or is 1 or more, an entry of `K_high` or `K_low` is NaN or infinite, the segments are not finite and strictly increasing, a segment length is not finite or is subnormal, the sum of the ends of a segment overflows, `nx` is not `n\\_segments\\_x * n\\_gauss` or `ny` is not `n\\_segments\\_y * n\\_gauss`, or the matrix has rank 0 - [`SPIR_INVALID_DIMENSION`](@ref) if the matrix is too large to be addressed - [`SPIR_INTERNAL_ERROR`](@ref) if the SVD fails (e.g. the QR of the matrix overflows) or an internal error occurs

The arrays are validated before the SVE is computed.
"""
function spir_sve_result_from_matrix(K_high, K_low, nx, ny, order, segments_x, n_segments_x, segments_y, n_segments_y, n_gauss, epsilon, status)
    ccall((:spir_sve_result_from_matrix, libsparseir), Ptr{spir_sve_result}, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cint, Cdouble, Ptr{StatusCode}), K_high, K_low, nx, ny, order, segments_x, n_segments_x, segments_y, n_segments_y, n_gauss, epsilon, status)
end

"""
    spir_sve_result_from_matrix_centrosymmetric(K_even_high, K_even_low, K_odd_high, K_odd_low, nx, ny, order, segments_x, n_segments_x, segments_y, n_segments_y, n_gauss, epsilon, status)

Create a SVE result from centrosymmetric discretized kernel matrices

This function performs singular value expansion (SVE) on centrosymmetric discretized kernel matrices using even/odd symmetry decomposition. The matrices K\\_even and K\\_odd should already be in the appropriate form (no weight application needed). The function supports both double and DDouble precision based on whether K\\_low is provided.

# Arguments * `K_even_high` - High part of the even-symmetry kernel matrix (required, size: nx * ny, finite entries) * `K_even_low` - Low part of the even-symmetry kernel matrix (optional, nullptr for double precision; finite entries) * `K_odd_high` - High part of the odd-symmetry kernel matrix (required, size: nx * ny, finite entries) * `K_odd_low` - Low part of the odd-symmetry kernel matrix (optional, nullptr for double precision; finite entries) * `nx` - Number of rows in the matrix (must be `n\\_segments\\_x * n\\_gauss`, the number of Gauss points of the segments on [0, xmax]) * `ny` - Number of columns in the matrix (must be `n\\_segments\\_y * n\\_gauss`, the number of Gauss points of the segments on [0, ymax]) * `order` - Memory layout ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref)) * `segments_x` - X-direction segments on the half domain (array of boundary points, size: n\\_segments\\_x + 1, finite and strictly increasing, starting at 0) * `n_segments_x` - Number of segments in x direction (boundary points - 1) * `segments_y` - Y-direction segments on the half domain (array of boundary points, size: n\\_segments\\_y + 1, finite and strictly increasing, starting at 0) * `n_segments_y` - Number of segments in y direction (boundary points - 1) * `n_gauss` - Number of Gauss points per segment * `epsilon` - Target accuracy * `status` - Pointer to store status code

# Returns Pointer to SVE result on success, nullptr on failure. If `status` is non-NULL, `*status` is set to: - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if `K_even_high`, `K_odd_high`, `segments_x` or `segments_y` is NULL, a size is less than 1, `epsilon` is not positive and finite or is 1 or more, an entry of a matrix that is read is NaN or infinite, the segments are not finite and strictly increasing, the segments do not start at 0, a segment length is not finite or is subnormal, the sum of the ends of a segment overflows, `nx` is not `n\\_segments\\_x * n\\_gauss` or `ny` is not `n\\_segments\\_y * n\\_gauss`, or both matrices have rank 0 - [`SPIR_INVALID_DIMENSION`](@ref) if the matrices are too large to be addressed - [`SPIR_INTERNAL_ERROR`](@ref) if an SVD fails (e.g. the QR of a matrix overflows) or an internal error occurs

The low parts are read only if both are non-NULL. The arrays are validated before the SVE is computed.
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

# Arguments * `n` - Number of Gauss points per segment (must be >= 1) * `segments` - Array of segment boundaries (n\\_segments + 1 elements): finite and strictly increasing, with finite segment lengths and a finite sum of the ends of each segment * `n_segments` - Number of segments (must be >= 1) * `x` - Output array for Gauss points (size n * n\\_segments). Must be pre-allocated. * `w` - Output array for Gauss weights (size n * n\\_segments). Must be pre-allocated. * `status` - Pointer to store the status code

# Returns Status code (also written to `*status`): - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if a pointer is NULL, `n` or `n_segments` < 1, or `segments` does not meet the conditions above - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs
"""
function spir_gauss_legendre_rule_piecewise_double(n, segments, n_segments, x, w, status)
    ccall((:spir_gauss_legendre_rule_piecewise_double, libsparseir), StatusCode, (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{StatusCode}), n, segments, n_segments, x, w, status)
end

"""
    spir_gauss_legendre_rule_piecewise_ddouble(n, segments, n_segments, x_high, x_low, w_high, w_low, status)

Compute piecewise Gauss-Legendre quadrature rule (DDouble precision)

Generates a piecewise Gauss-Legendre quadrature rule with n points per segment, computed using extended precision (DDouble). Returns high and low parts separately for maximum precision.

# Arguments * `n` - Number of Gauss points per segment (must be >= 1) * `segments` - Array of segment boundaries (n\\_segments + 1 elements): finite and strictly increasing, with finite segment lengths and a finite sum of the ends of each segment * `n_segments` - Number of segments (must be >= 1) * `x_high` - Output array for high part of Gauss points (size n * n\\_segments). Must be pre-allocated. * `x_low` - Output array for low part of Gauss points (size n * n\\_segments). Must be pre-allocated. * `w_high` - Output array for high part of Gauss weights (size n * n\\_segments). Must be pre-allocated. * `w_low` - Output array for low part of Gauss weights (size n * n\\_segments). Must be pre-allocated. * `status` - Pointer to store the status code

# Returns Status code (also written to `*status`): - [`SPIR_COMPUTATION_SUCCESS`](@ref) (0) on success - [`SPIR_INVALID_ARGUMENT`](@ref) if a pointer is NULL, `n` or `n_segments` < 1, or `segments` does not meet the conditions above - [`SPIR_INTERNAL_ERROR`](@ref) if an internal error occurs
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
