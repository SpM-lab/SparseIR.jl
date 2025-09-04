module C_API

using CEnum

using Libdl: dlext

libsparseir = expanduser("~/opt/libsparseir/lib/libsparseir.$(dlext)")


const c_complex = ComplexF64

mutable struct _spir_kernel end

const spir_kernel = _spir_kernel

function spir_kernel_release(obj)
    ccall((:spir_kernel_release, libsparseir), Cvoid, (Ptr{spir_kernel},), obj)
end

function spir_kernel_clone(src)
    ccall((:spir_kernel_clone, libsparseir), Ptr{spir_kernel}, (Ptr{spir_kernel},), src)
end

function spir_kernel_is_assigned(obj)
    ccall((:spir_kernel_is_assigned, libsparseir), Cint, (Ptr{spir_kernel},), obj)
end

function _spir_kernel_get_raw_ptr(obj)
    ccall((:_spir_kernel_get_raw_ptr, libsparseir), Ptr{Cvoid}, (Ptr{spir_kernel},), obj)
end

mutable struct _spir_funcs end

const spir_funcs = _spir_funcs

function spir_funcs_release(obj)
    ccall((:spir_funcs_release, libsparseir), Cvoid, (Ptr{spir_funcs},), obj)
end

function spir_funcs_clone(src)
    ccall((:spir_funcs_clone, libsparseir), Ptr{spir_funcs}, (Ptr{spir_funcs},), src)
end

function spir_funcs_is_assigned(obj)
    ccall((:spir_funcs_is_assigned, libsparseir), Cint, (Ptr{spir_funcs},), obj)
end

function _spir_funcs_get_raw_ptr(obj)
    ccall((:_spir_funcs_get_raw_ptr, libsparseir), Ptr{Cvoid}, (Ptr{spir_funcs},), obj)
end

mutable struct _spir_basis end

const spir_basis = _spir_basis

function spir_basis_release(obj)
    ccall((:spir_basis_release, libsparseir), Cvoid, (Ptr{spir_basis},), obj)
end

function spir_basis_clone(src)
    ccall((:spir_basis_clone, libsparseir), Ptr{spir_basis}, (Ptr{spir_basis},), src)
end

function spir_basis_is_assigned(obj)
    ccall((:spir_basis_is_assigned, libsparseir), Cint, (Ptr{spir_basis},), obj)
end

function _spir_basis_get_raw_ptr(obj)
    ccall((:_spir_basis_get_raw_ptr, libsparseir), Ptr{Cvoid}, (Ptr{spir_basis},), obj)
end

mutable struct _spir_sampling end

const spir_sampling = _spir_sampling

function spir_sampling_release(obj)
    ccall((:spir_sampling_release, libsparseir), Cvoid, (Ptr{spir_sampling},), obj)
end

function spir_sampling_clone(src)
    ccall((:spir_sampling_clone, libsparseir), Ptr{spir_sampling}, (Ptr{spir_sampling},), src)
end

function spir_sampling_is_assigned(obj)
    ccall((:spir_sampling_is_assigned, libsparseir), Cint, (Ptr{spir_sampling},), obj)
end

function _spir_sampling_get_raw_ptr(obj)
    ccall((:_spir_sampling_get_raw_ptr, libsparseir), Ptr{Cvoid}, (Ptr{spir_sampling},), obj)
end

mutable struct _spir_sve_result end

const spir_sve_result = _spir_sve_result

function spir_sve_result_release(obj)
    ccall((:spir_sve_result_release, libsparseir), Cvoid, (Ptr{spir_sve_result},), obj)
end

function spir_sve_result_clone(src)
    ccall((:spir_sve_result_clone, libsparseir), Ptr{spir_sve_result}, (Ptr{spir_sve_result},), src)
end

function spir_sve_result_is_assigned(obj)
    ccall((:spir_sve_result_is_assigned, libsparseir), Cint, (Ptr{spir_sve_result},), obj)
end

function _spir_sve_result_get_raw_ptr(obj)
    ccall((:_spir_sve_result_get_raw_ptr, libsparseir), Ptr{Cvoid}, (Ptr{spir_sve_result},), obj)
end

"""
    spir_logistic_kernel_new(lambda, status)

Creates a new logistic kernel for fermionic/bosonic analytical continuation.

In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is a function on [-1, 1] × [-1, 1]:

K(x, y) = exp(-Λy(x + 1)/2)/(1 + exp(-Λy))

While LogisticKernel is primarily a fermionic analytic continuation kernel, it can also model the τ dependence of a bosonic correlation function as:

∫ [exp(-Λy(x + 1)/2)/(1 - exp(-Λy))] ρ(y) dy = ∫ K(x, y) ρ'(y) dy

where ρ'(y) = w(y)ρ(y) and the weight function w(y) = 1/tanh(Λy/2)

!!! note

    The kernel is implemented using piecewise Legendre polynomial expansion for numerical stability and accuracy.

# Arguments
* `lambda`: The cutoff parameter Λ (must be non-negative)
* `status`: Pointer to store the status code
# Returns
Pointer to the newly created kernel object, or NULL if creation fails
"""
function spir_logistic_kernel_new(lambda, status)
    ccall((:spir_logistic_kernel_new, libsparseir), Ptr{spir_kernel}, (Cdouble, Ptr{Cint}), lambda, status)
end

"""
    spir_reg_bose_kernel_new(lambda, status)

Creates a new regularized bosonic kernel for analytical continuation.

In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is a function on [-1, 1] × [-1, 1]:

K(x, y) = y * exp(-Λy(x + 1)/2)/(exp(-Λy) - 1)

Special care is taken in evaluating this expression around y = 0 to handle the singularity. The kernel is specifically designed for bosonic functions and includes proper regularization to handle numerical stability issues.

!!! note

    This kernel is specifically designed for bosonic correlation functions and should not be used for fermionic cases.

!!! note

    The kernel is implemented using piecewise Legendre polynomial expansion for numerical stability and accuracy.

# Arguments
* `lambda`: The cutoff parameter Λ (must be non-negative)
* `status`: Pointer to store the status code
# Returns
Pointer to the newly created kernel object, or NULL if creation fails
"""
function spir_reg_bose_kernel_new(lambda, status)
    ccall((:spir_reg_bose_kernel_new, libsparseir), Ptr{spir_kernel}, (Cdouble, Ptr{Cint}), lambda, status)
end

"""
    spir_kernel_domain(k, xmin, xmax, ymin, ymax)

Retrieves the domain boundaries of a kernel function.

This function obtains the domain boundaries (ranges) for both the x and y variables of the specified kernel function. The kernel domain is typically defined as a rectangle in the (x,y) plane.

!!! note

    For the logistic and regularized bosonic kernels, the domain is typically [-1, 1] × [-1, 1] in dimensionless variables.

# Arguments
* `k`: Pointer to the kernel object whose domain is to be retrieved.
* `xmin`: Pointer to store the minimum value of the x-range.
* `xmax`: Pointer to store the maximum value of the x-range.
* `ymin`: Pointer to store the minimum value of the y-range.
* `ymax`: Pointer to store the maximum value of the y-range.
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
"""
function spir_kernel_domain(k, xmin, xmax, ymin, ymax)
    ccall((:spir_kernel_domain, libsparseir), Cint, (Ptr{spir_kernel}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), k, xmin, xmax, ymin, ymax)
end

"""
    spir_sve_result_new(k, epsilon, cutoff, lmax, n_gauss, work_dtype, status)

Perform truncated singular value expansion (SVE) of a kernel.

Computes a truncated singular value expansion of an integral kernel K: [xmin, xmax] × [ymin, ymax] → ℝ in the form:

K(x, y) = ∑ s[l] * u[l](x) * v[l](y) for l = 1, 2, 3, ...

where: - s[l] are singular values in non-increasing order - u[l](x) are left singular functions, forming an orthonormal system on [xmin, xmax] - v[l](y) are right singular functions, forming an orthonormal system on [ymin, ymax]

The SVE is computed by mapping it onto a singular value decomposition (SVD) of a matrix using piecewise Legendre polynomial expansion. The accuracy of the computation is controlled by the epsilon parameter, which determines: - The relative magnitude of included singular values - The accuracy of computed singular values and vectors

!!! note

    The computation automatically uses optimized strategies: - For centrosymmetric kernels, specialized algorithms are employed - The working precision is adjusted to meet accuracy requirements - If epsilon is below √ε (where ε is machine epsilon), a warning is issued and higher precision arithmetic is used

!!! note

    The returned object must be freed using spir\\_release\\_sve\\_result when no longer needed

# Arguments
* `k`: Pointer to the kernel object for which to compute SVE
* `epsilon`: Accuracy target for the basis. Determines: - The relative magnitude of included singular values - The accuracy of computed singular values and vectors
* `status`: Pointer to store the status code
# Returns
Pointer to the newly created SVE result, or NULL if creation fails
# See also
spir\\_release\\_sve\\_result
"""
function spir_sve_result_new(k, epsilon, cutoff, lmax, n_gauss, work_dtype, status)
    ccall((:spir_sve_result_new, libsparseir), Ptr{spir_sve_result}, (Ptr{spir_kernel}, Cdouble, Cdouble, Cint, Cint, Cint, Ptr{Cint}), k, epsilon, cutoff, lmax, n_gauss, work_dtype, status)
end

"""
    spir_sve_result_get_size(sve, size)

Gets the number of singular values/vectors in an SVE result.

This function returns the number of singular values and corresponding singular vectors contained in the specified SVE result object. This number is needed to allocate arrays of the correct size when retrieving singular values or evaluating singular vectors.

# Arguments
* `sve`: Pointer to the SVE result object
* `size`: Pointer to store the number of singular values/vectors
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
"""
function spir_sve_result_get_size(sve, size)
    ccall((:spir_sve_result_get_size, libsparseir), Cint, (Ptr{spir_sve_result}, Ptr{Cint}), sve, size)
end

"""
    spir_sve_result_get_svals(sve, svals)

Gets the singular values from an SVE result.

This function retrieves all singular values from the specified SVE result object. The singular values are stored in descending order in the output array.

# Arguments
* `sve`: Pointer to the SVE result object
* `svals`: Pre-allocated array to store the singular values. Must have size at least equal to the value returned by [`spir_sve_result_get_size`](@ref)()
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_sve_result_get_size`](@ref)
"""
function spir_sve_result_get_svals(sve, svals)
    ccall((:spir_sve_result_get_svals, libsparseir), Cint, (Ptr{spir_sve_result}, Ptr{Cdouble}), sve, svals)
end

"""
    spir_funcs_get_size(funcs, size)

Gets the number of functions in a functions object.

This function returns the number of functions contained in the specified functions object. This number is needed to allocate arrays of the correct size when evaluating the functions.

# Arguments
* `funcs`: Pointer to the functions object
* `size`: Pointer to store the number of functions
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
"""
function spir_funcs_get_size(funcs, size)
    ccall((:spir_funcs_get_size, libsparseir), Cint, (Ptr{spir_funcs}, Ptr{Cint}), funcs, size)
end

"""
    spir_funcs_get_slice(funcs, nslice, indices, status)

Creates a new function object containing a subset of functions from the input.

This function creates a new function object that contains only the functions specified by the indices array. The indices must be valid (within range and no duplicates).

!!! note

    The caller is responsible for freeing the returned object using spir\\_funcs\\_free

!!! note

    If status is non-zero, the returned pointer will be NULL

# Arguments
* `funcs`: Pointer to the source function object
* `nslice`: Number of functions to select (length of indices array)
* `indices`: Array of indices specifying which functions to include in the slice
* `status`: Pointer to store the status code (0 for success, non-zero for error)
# Returns
Pointer to the new function object containing the selected functions, or NULL on error
"""
function spir_funcs_get_slice(funcs, nslice, indices, status)
    ccall((:spir_funcs_get_slice, libsparseir), Ptr{spir_funcs}, (Ptr{spir_funcs}, Cint, Ptr{Cint}, Ptr{Cint}), funcs, nslice, indices, status)
end

"""
    spir_funcs_eval(funcs, x, out)

Evaluates functions at a single point in the imaginary-time domain or the real frequency domain.

This function evaluates all functions at a specified point x. The values of each basis function at x are stored in the output array. The output array out[j] contains the value of the j-th function evaluated at x.

!!! note

    The output array must be pre-allocated with sufficient size to store all function values

# Arguments
* `funcs`: Pointer to a functions object
* `x`: Point at which to evaluate the functions
* `out`: Pre-allocated array to store the evaluation results.
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
"""
function spir_funcs_eval(funcs, x, out)
    ccall((:spir_funcs_eval, libsparseir), Cint, (Ptr{spir_funcs}, Cdouble, Ptr{Cdouble}), funcs, x, out)
end

"""
    spir_funcs_eval_matsu(funcs, x, out)

Evaluate a funcs object at a single Matsubara frequency

This function evaluates the basis functions at a single Matsubara frequency index. The output array will contain the values of all basis functions at the specified frequency.

# Arguments
* `funcs`: Pointer to the funcs object to evaluate
* `x`: The Matsubara frequency index (integer)
* `out`: Pointer to the output array where the results will be stored. The array must have enough space to store all basis function values. The values are stored in the order of basis functions.
# Returns
int [`SPIR_COMPUTATION_SUCCESS`](@ref) on success, or an error code on failure
"""
function spir_funcs_eval_matsu(funcs, x, out)
    ccall((:spir_funcs_eval_matsu, libsparseir), Cint, (Ptr{spir_funcs}, Int64, Ptr{c_complex}), funcs, x, out)
end

"""
    spir_funcs_batch_eval(funcs, order, num_points, xs, out)

Evaluate a funcs object at multiple points in the imaginary-time domain or the real frequency domain

This function evaluates the basis functions at multiple points. The points can be either in the imaginary-time domain or the real frequency domain, depending on the type of the funcs object (u or v basis functions).

The output array can be stored in either row-major or column-major order, specified by the order parameter. In row-major order, the output is stored as (num\\_points, nfuncs), while in column-major order, it is stored as (nfuncs, num\\_points).

# Arguments
* `funcs`: Pointer to the funcs object to evaluate
* `order`: Memory layout of the output array: - [`SPIR_ORDER_ROW_MAJOR`](@ref): (num\\_points, nfuncs) - [`SPIR_ORDER_COLUMN_MAJOR`](@ref): (nfuncs, num\\_points)
* `num_points`: Number of points to evaluate
* `xs`: Array of points to evaluate at. The points should be in the appropriate domain (imaginary time for u basis, real frequency for v basis)
* `out`: Pointer to the output array where the results will be stored. The array must have enough space to store num\\_points * nfuncs values, where nfuncs is the number of basis functions.
# Returns
int [`SPIR_COMPUTATION_SUCCESS`](@ref) on success, or an error code on failure
"""
function spir_funcs_batch_eval(funcs, order, num_points, xs, out)
    ccall((:spir_funcs_batch_eval, libsparseir), Cint, (Ptr{spir_funcs}, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}), funcs, order, num_points, xs, out)
end

"""
    spir_funcs_batch_eval_matsu(funcs, order, num_freqs, matsubara_freq_indices, out)

Evaluates basis functions at multiple Matsubara frequencies.

This function evaluates all functions contained in a functions object at the specified Matsubara frequency indices. The values of each function at each frequency are stored in the output array.

!!! note

    The output array must be pre-allocated with sufficient size to store all function values at all requested frequencies. Indices n correspond to ωn = nπ/β, where n are odd for fermionic frequencies and even for bosonic frequencies.

# Arguments
* `funcs`: Pointer to the functions object
* `order`: Specifies the memory layout of the output array: [`SPIR_ORDER_ROW_MAJOR`](@ref) for row-major order (frequency index varies fastest), [`SPIR_ORDER_COLUMN_MAJOR`](@ref) for column-major order (function index varies fastest)
* `num_freqs`: Number of Matsubara frequencies at which to evaluate
* `matsubara_freq_indices`: Array of Matsubara frequency indices
* `out`: Pre-allocated array to store the evaluation results. The results are stored as a 2D array of size num\\_freqs x n\\_funcs.
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
"""
function spir_funcs_batch_eval_matsu(funcs, order, num_freqs, matsubara_freq_indices, out)
    ccall((:spir_funcs_batch_eval_matsu, libsparseir), Cint, (Ptr{spir_funcs}, Cint, Cint, Ptr{Int64}, Ptr{c_complex}), funcs, order, num_freqs, matsubara_freq_indices, out)
end

"""
    spir_funcs_get_n_roots(funcs, n_roots)

Gets the number of roots of a funcs object.

This function returns the number of roots of the specified funcs object. This function is only available for continuous functions.

# Arguments
* `funcs`: Pointer to the funcs object
* `n_roots`: Pointer to store the number of roots
# Returns
An integer status code:
"""
function spir_funcs_get_n_roots(funcs, n_roots)
    ccall((:spir_funcs_get_n_roots, libsparseir), Cint, (Ptr{spir_funcs}, Ptr{Cint}), funcs, n_roots)
end

"""
    spir_funcs_get_roots(funcs, roots)

Gets the roots of a funcs object.

This function returns the roots of the specified funcs object in the non-ascending order. If the size of the funcs object is greater than 1, the roots for all the functions are returned. This function is only available for continuous functions.

# Arguments
* `funcs`: Pointer to the funcs object
* `n_roots`: Pointer to store the number of roots
* `roots`: Pointer to store the roots
# Returns
An integer status code:
"""
function spir_funcs_get_roots(funcs, roots)
    ccall((:spir_funcs_get_roots, libsparseir), Cint, (Ptr{spir_funcs}, Ptr{Cdouble}), funcs, roots)
end

"""
    spir_basis_new(statistics, beta, omega_max, k, sve, max_size, status)

Creates a new finite temperature IR basis using a pre-computed SVE result.

This function creates a intermediate representation (IR) basis using a pre-computed singular value expansion (SVE) result. This allows for reusing an existing SVE computation, which can be more efficient than recomputing it.

!!! note

    Using a pre-computed SVE can significantly improve performance when creating multiple basis objects with the same kernel

# Arguments
* `statistics`: Statistics type ([`SPIR_STATISTICS_FERMIONIC`](@ref) or [`SPIR_STATISTICS_BOSONIC`](@ref))
* `beta`: Inverse temperature β (must be positive)
* `omega_max`: Frequency cutoff ωmax (must be non-negative)
* `k`: Pointer to the kernel object used for the basis construction
* `sve`: Pointer to a pre-computed SVE result for the kernel
* `max_size`: Maximum number of basis functions to include. If -1, all
* `status`: Pointer to store the status code
# Returns
Pointer to the newly created basis object, or NULL if creation fails
# See also
[`spir_sve_result_new`](@ref), spir\\_release\\_finite\\_temp\\_basis
"""
function spir_basis_new(statistics, beta, omega_max, k, sve, max_size, status)
    ccall((:spir_basis_new, libsparseir), Ptr{spir_basis}, (Cint, Cdouble, Cdouble, Ptr{spir_kernel}, Ptr{spir_sve_result}, Cint, Ptr{Cint}), statistics, beta, omega_max, k, sve, max_size, status)
end

"""
    spir_basis_get_size(b, size)

Gets the size (number of basis functions) of a finite temperature basis.

This function returns the number of basis functions in the specified finite temperature basis object. This size determines the dimensionality of the basis and is needed when allocating arrays for basis function evaluations.

!!! note

    For an IR basis, the size is determined automatically during basis construction based on the specified parameters (β, ωmax, ε) and the kernel's singular value expansion.

!!! note

    For a DLR basis, the size is the number of poles.

# Arguments
* `b`: Pointer to the finite temperature basis object
* `size`: Pointer to store the number of basis functions
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
"""
function spir_basis_get_size(b, size)
    ccall((:spir_basis_get_size, libsparseir), Cint, (Ptr{spir_basis}, Ptr{Cint}), b, size)
end

"""
    spir_basis_get_svals(b, svals)

Gets the singular values of a finite temperature basis.

This function returns the singular values of the specified finite temperature basis object. The singular values are the square roots of the eigenvalues of the covariance matrix of the basis functions.

!!! note

    The singular values are ordered in descending order

!!! note

    The number of singular values is equal to the basis size

# Arguments
* `sve`: Pointer to the finite temperature basis object
* `svals`: Pointer to store the singular values
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_basis_get_size`](@ref)
"""
function spir_basis_get_svals(b, svals)
    ccall((:spir_basis_get_svals, libsparseir), Cint, (Ptr{spir_basis}, Ptr{Cdouble}), b, svals)
end

"""
    spir_basis_get_stats(b, statistics)

Gets the statistics type (Fermionic or Bosonic) of a finite temperature basis.

This function returns the statistics type of the specified finite temperature basis object. The statistics type determines whether the basis is for fermionic or bosonic Green's functions.

!!! note

    The statistics type is determined during basis construction and cannot be changed

!!! note

    The statistics type affects the form of the basis functions and the sampling points used for evaluation.

# Arguments
* `b`: Pointer to the finite temperature basis object
* `statistics`: Pointer to store the statistics type ([`SPIR_STATISTICS_FERMIONIC`](@ref) or [`SPIR_STATISTICS_BOSONIC`](@ref))
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
"""
function spir_basis_get_stats(b, statistics)
    ccall((:spir_basis_get_stats, libsparseir), Cint, (Ptr{spir_basis}, Ptr{Cint}), b, statistics)
end

"""
    spir_basis_get_singular_values(b, svals)

Gets the singular values of a finite temperature basis.

This function returns the singular values of the specified finite temperature basis object. The singular values are the square roots of the eigenvalues of the covariance matrix of the basis functions.
"""
function spir_basis_get_singular_values(b, svals)
    ccall((:spir_basis_get_singular_values, libsparseir), Cint, (Ptr{spir_basis}, Ptr{Cdouble}), b, svals)
end

"""
    spir_basis_get_u(b, status)

Gets the basis functions of a finite temperature basis.

This function returns an object representing the basis functions in the imaginary-time domain of the specified finite temperature basis.

!!! note

    The returned object must be freed using spir\\_release\\_funcs when no longer needed

# Arguments
* `b`: Pointer to the finite temperature basis object
* `status`: Pointer to store the status code
# Returns
Pointer to the basis functions object, or NULL if creation fails
# See also
spir\\_release\\_funcs
"""
function spir_basis_get_u(b, status)
    ccall((:spir_basis_get_u, libsparseir), Ptr{spir_funcs}, (Ptr{spir_basis}, Ptr{Cint}), b, status)
end

"""
    spir_basis_get_v(b, status)

Gets the basis functions of a finite temperature basis.

This function returns an object representing the basis functions in the real-frequency domain of the specified finite temperature basis.

!!! note

    The returned object must be freed using spir\\_release\\_funcs when no longer needed

# Arguments
* `b`: Pointer to the finite temperature basis object
* `status`: Pointer to store the status code
# Returns
Pointer to the basis functions object, or NULL if creation fails
# See also
spir\\_release\\_funcs
"""
function spir_basis_get_v(b, status)
    ccall((:spir_basis_get_v, libsparseir), Ptr{spir_funcs}, (Ptr{spir_basis}, Ptr{Cint}), b, status)
end

"""
    spir_basis_get_uhat(b, status)

Gets the basis functions in Matsubara frequency domain.

This function returns an object representing the basis functions in the Matsubara-frequency domain of the specified finite temperature basis.

!!! note

    The returned object must be freed using spir\\_release\\_funcs when no longer needed

# Arguments
* `b`: Pointer to the finite temperature basis object
* `status`: Pointer to store the status code
# Returns
Pointer to the basis functions object, or NULL if creation fails
# See also
spir\\_release\\_funcs
"""
function spir_basis_get_uhat(b, status)
    ccall((:spir_basis_get_uhat, libsparseir), Ptr{spir_funcs}, (Ptr{spir_basis}, Ptr{Cint}), b, status)
end

"""
    spir_basis_get_n_default_taus(b, num_points)

Gets the number of default tau sampling points for an IR basis.

This function returns the number of default sampling points in imaginary time (τ) that are automatically chosen for optimal conditioning of the sampling matrix. These points are the extrema of the highest-order basis function in imaginary time.

!!! note

    This function is only available for IR basis objects

!!! note

    The default sampling points are chosen to provide near-optimal conditioning for the given basis size

# Arguments
* `b`: Pointer to a finite temperature basis object (must be an IR basis)
* `num_points`: Pointer to store the number of sampling points
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_basis_get_default_taus`](@ref)
"""
function spir_basis_get_n_default_taus(b, num_points)
    ccall((:spir_basis_get_n_default_taus, libsparseir), Cint, (Ptr{spir_basis}, Ptr{Cint}), b, num_points)
end

"""
    spir_basis_get_default_taus(b, points)

Gets the default tau sampling points for an IR basis.

This function fills the provided array with the default sampling points in imaginary time (τ) that are automatically chosen for optimal conditioning of the sampling matrix. These points are the extrema of the highest-order basis function in imaginary time.

!!! note

    This function is only available for IR basis objects

!!! note

    The array must be pre-allocated with size >= [`spir_basis_get_n_default_taus`](@ref)(b)

!!! note

    The default sampling points are chosen to provide near-optimal conditioning for the given basis size

# Arguments
* `b`: Pointer to a finite temperature basis object (must be an IR basis)
* `points`: Pre-allocated array to store the τ sampling points
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_basis_get_n_default_taus`](@ref)
"""
function spir_basis_get_default_taus(b, points)
    ccall((:spir_basis_get_default_taus, libsparseir), Cint, (Ptr{spir_basis}, Ptr{Cdouble}), b, points)
end

"""
    spir_basis_get_n_default_ws(b, num_points)

Gets the number of default omega sampling points for an IR basis.

This function returns the number of default sampling points in real frequency (ω) that are automatically chosen for optimal conditioning of the sampling matrix. These points are the extrema of the highest-order basis function in real frequency.

!!! note

    This function is only available for IR basis objects

!!! note

    The default sampling points are chosen to provide near-optimal conditioning for the given basis size

# Arguments
* `b`: Pointer to a finite temperature basis object (must be an IR basis)
* `num_points`: Pointer to store the number of sampling points
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_basis_get_default_ws`](@ref)
"""
function spir_basis_get_n_default_ws(b, num_points)
    ccall((:spir_basis_get_n_default_ws, libsparseir), Cint, (Ptr{spir_basis}, Ptr{Cint}), b, num_points)
end

"""
    spir_basis_get_default_ws(b, points)

Gets the default omega sampling points for an IR basis.

This function fills the provided array with the default sampling points in real frequency (ω) that are automatically chosen for optimal conditioning of the sampling matrix. These points are the extrema of the highest-order basis function in real frequency.

!!! note

    This function is only available for IR basis objects

!!! note

    The array must be pre-allocated with size >= [`spir_basis_get_n_default_ws`](@ref)(b)

!!! note

    The default sampling points are chosen to provide near-optimal conditioning for the given basis size

# Arguments
* `b`: Pointer to a finite temperature basis object (must be an IR basis)
* `points`: Pre-allocated array to store the ω sampling points
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_basis_get_n_default_ws`](@ref)
"""
function spir_basis_get_default_ws(b, points)
    ccall((:spir_basis_get_default_ws, libsparseir), Cint, (Ptr{spir_basis}, Ptr{Cdouble}), b, points)
end

"""
    spir_basis_get_default_taus_ext(b, n_points, points, n_points_returned)

*

Gets the default tau sampling points for ann IR basis.

This function returns default tau sampling points for an IR basis object.

# Arguments
* `b`: Pointer to the basis object
* `n_points`: Number of requested sampling points.
* `points`: Pre-allocated array to store the sampling points. The size of the array must be at least n\\_points.
* `n_points_returned`: Number of sampling points returned.
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success
"""
function spir_basis_get_default_taus_ext(b, n_points, points, n_points_returned)
    ccall((:spir_basis_get_default_taus_ext, libsparseir), Cint, (Ptr{spir_basis}, Cint, Ptr{Cdouble}, Ptr{Cint}), b, n_points, points, n_points_returned)
end

"""
    spir_basis_get_n_default_matsus(b, positive_only, num_points)

Gets the number of default Matsubara sampling points for an IR basis.

This function returns the number of default sampling points in Matsubara frequencies (iωn) that are automatically chosen for optimal conditioning of the sampling matrix. These points are the extrema of the highest-order basis function in Matsubara frequencies.

!!! note

    This function is only available for IR basis objects

!!! note

    The default sampling points are chosen to provide near-optimal conditioning for the given basis size

# Arguments
* `b`: Pointer to a finite temperature basis object (must be an IR basis)
* `positive_only`: If true, only positive frequencies are used
* `num_points`: Pointer to store the number of sampling points
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_basis_get_default_matsus`](@ref)
"""
function spir_basis_get_n_default_matsus(b, positive_only, num_points)
    ccall((:spir_basis_get_n_default_matsus, libsparseir), Cint, (Ptr{spir_basis}, Bool, Ptr{Cint}), b, positive_only, num_points)
end

"""
    spir_basis_get_default_matsus(b, positive_only, points)

Gets the default Matsubara sampling points for an IR basis.

This function fills the provided array with the default sampling points in Matsubara frequencies (iωn) that are automatically chosen for optimal conditioning of the sampling matrix. These points are the extrema of the highest-order basis function in Matsubara frequencies.

!!! note

    This function is only available for IR basis objects

!!! note

    The array must be pre-allocated with size >= [`spir_basis_get_n_default_matsus`](@ref)(b)

!!! note

    The default sampling points are chosen to provide near-optimal conditioning for the given basis size

!!! note

    For fermionic case, the indices n give frequencies ωn = (2n + 1)π/β

!!! note

    For bosonic case, the indices n give frequencies ωn = 2nπ/β

# Arguments
* `b`: Pointer to a finite temperature basis object (must be an IR basis)
* `positive_only`: If true, only positive frequencies are used
* `points`: Pre-allocated array to store the Matsubara frequency indices
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_basis_get_n_default_matsus`](@ref)
"""
function spir_basis_get_default_matsus(b, positive_only, points)
    ccall((:spir_basis_get_default_matsus, libsparseir), Cint, (Ptr{spir_basis}, Bool, Ptr{Int64}), b, positive_only, points)
end

"""
    spir_basis_get_n_default_matsus_ext(b, positive_only, L, num_points_returned)

Gets the number of default Matsubara sampling points for an IR basis.

This function returns the number of default sampling points in Matsubara frequencies (iωn) that are automatically chosen for optimal conditioning of the sampling matrix. These points are the extrema of the highest-order basis function in Matsubara frequencies.

!!! note

    This function is only available for IR basis objects

!!! note

    The default sampling points are chosen to provide near-optimal conditioning for the given basis size

# Arguments
* `b`: Pointer to a finite temperature basis object (must be an IR basis)
* `positive_only`: If true, only positive frequencies are used
* `L`: Number of requested sampling points.
* `num_points_returned`: Pointer to store the number of sampling points returned.
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_basis_get_default_matsus`](@ref)
"""
function spir_basis_get_n_default_matsus_ext(b, positive_only, L, num_points_returned)
    ccall((:spir_basis_get_n_default_matsus_ext, libsparseir), Cint, (Ptr{spir_basis}, Bool, Cint, Ptr{Cint}), b, positive_only, L, num_points_returned)
end

"""
    spir_basis_get_default_matsus_ext(b, positive_only, n_points, points, n_points_returned)

Gets the default Matsubara sampling points for an IR basis.

This function fills the provided array with the default sampling points in Matsubara frequencies (iωn) that are automatically chosen for optimal conditioning of the sampling matrix. These points are the extrema of the highest-order basis function in Matsubara frequencies.

# Arguments
* `b`: Pointer to a finite temperature basis object (must be an IR basis)
* `positive_only`: If true, only positive frequencies are used
* `n_points`: Number of requested sampling points.
* `points`: Pre-allocated array to store the sampling points. The size of the array must be at least n\\_points.
* `n_points_returned`: Number of sampling points returned.
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success
"""
function spir_basis_get_default_matsus_ext(b, positive_only, n_points, points, n_points_returned)
    ccall((:spir_basis_get_default_matsus_ext, libsparseir), Cint, (Ptr{spir_basis}, Bool, Cint, Ptr{Int64}, Ptr{Cint}), b, positive_only, n_points, points, n_points_returned)
end

"""
    spir_dlr_new(b, status)

Creates a new Discrete Lehmann Representation (DLR) basis.

This function implements a variant of the discrete Lehmann representation (DLR). Unlike the IR which uses truncated singular value expansion of the analytic continuation kernel K, the DLR is based on a "sketching" of K. The resulting basis is a linear combination of discrete set of poles on the real-frequency axis, continued to the imaginary-frequency axis:

G(iν) = ∑ a[i] * reg[i] / (iν - w[i]) for i = 1, 2, ..., L

where: - a[i] are the expansion coefficients - w[i] are the poles on the real axis - reg[i] are the regularization factors, which are 1 for fermionic frequencies. For bosonic frequencies, we take reg[i] = tanh(βω[i]/2) (logistic kernel), reg[i] = w[i] (regularized bosonic kernel). The DLR basis functions are given by u[i](iν) = reg[i] / (iν - w[i]) in the imaginary-frequency domain. In the imaginary-time domain, the basis functions are given by u[i](τ) = reg[i] * exp(-w[i]τ) / (1 + exp(-w[i]β)) for fermionic frequencies, u[i](τ) = reg[i] * exp(-w[i]τ) / (1 - exp(-w[i]β)) for bosonic frequencies. - iν are Matsubara frequencies

# Arguments
* `b`: Pointer to a finite temperature basis object
* `status`: Pointer to store the status code
# Returns
Pointer to the newly created DLR object, or NULL if creation fails
"""
function spir_dlr_new(b, status)
    ccall((:spir_dlr_new, libsparseir), Ptr{spir_basis}, (Ptr{spir_basis}, Ptr{Cint}), b, status)
end

"""
    spir_dlr_new_with_poles(b, npoles, poles, status)

Creates a new Discrete Lehmann Representation (DLR) with custom poles.

This function creates a DLR basis with user-specified pole locations on the real-frequency axis. This allows for more control over the pole selection compared to the automatic pole selection in [`spir_dlr_new`](@ref).

# Arguments
* `b`: Pointer to a finite temperature basis object
* `npoles`: Number of poles to use in the representation
* `poles`: Array of pole locations on the real-frequency axis
* `status`: Pointer to store the status code
# Returns
Pointer to the newly created DLR object, or NULL if creation fails
"""
function spir_dlr_new_with_poles(b, npoles, poles, status)
    ccall((:spir_dlr_new_with_poles, libsparseir), Ptr{spir_basis}, (Ptr{spir_basis}, Cint, Ptr{Cdouble}, Ptr{Cint}), b, npoles, poles, status)
end

"""
    spir_dlr_get_npoles(dlr, num_poles)

Gets the number of poles in a DLR.

This function returns the number of poles in the specified DLR object.

# Arguments
* `dlr`: Pointer to the DLR object
* `num_poles`: Pointer to store the number of poles
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_dlr_get_poles`](@ref)
"""
function spir_dlr_get_npoles(dlr, num_poles)
    ccall((:spir_dlr_get_npoles, libsparseir), Cint, (Ptr{spir_basis}, Ptr{Cint}), dlr, num_poles)
end

"""
    spir_dlr_get_poles(dlr, poles)

Gets the poles in a DLR.

This function returns the poles in the specified DLR object.

# Arguments
* `dlr`: Pointer to the DLR object
* `poles`: Pointer to store the poles
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_dlr_get_npoles`](@ref)
"""
function spir_dlr_get_poles(dlr, poles)
    ccall((:spir_dlr_get_poles, libsparseir), Cint, (Ptr{spir_basis}, Ptr{Cdouble}), dlr, poles)
end

"""
    spir_ir2dlr_dd(dlr, order, ndim, input_dims, target_dim, input, out)

Transforms a given input array from the Intermediate Representation (IR) to the Discrete Lehmann Representation (DLR) using the specified DLR object. This version handles real (double precision) input and output arrays.

!!! note

    The input and output arrays must be allocated with sufficient memory. The size of the input and output arrays should match the dimensions specified. The order type determines the memory layout of the input and output arrays. The function assumes that the input array is in the specified order type. The output array will be in the specified order type.

# Arguments
* `dlr`: Pointer to the DLR basis object
* `order`: Order type (C or Fortran)
* `ndim`: Number of dimensions of input/output arrays
* `input_dims`: Array of dimensions
* `target_dim`: Target dimension for the transformation (0-based)
* `input`: Input coefficients array in IR
* `out`: Output array in DLR
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
spir\\_ir2dlr, [`spir_dlr2ir_dd`](@ref)
"""
function spir_ir2dlr_dd(dlr, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_ir2dlr_dd, libsparseir), Cint, (Ptr{spir_basis}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), dlr, order, ndim, input_dims, target_dim, input, out)
end

function spir_ir2dlr_zz(dlr, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_ir2dlr_zz, libsparseir), Cint, (Ptr{spir_basis}, Cint, Cint, Ptr{Cint}, Cint, Ptr{c_complex}, Ptr{c_complex}), dlr, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_dlr2ir_dd(dlr, order, ndim, input_dims, target_dim, input, out)

Transforms coefficients from DLR basis to IR representation.

This function converts expansion coefficients from the Discrete Lehmann Representation (DLR) basis to the Intermediate Representation (IR) basis. The transformation is performed using the fitting matrix:

g\\_IR = fitmat * g\\_DLR

where: - g\\_IR are the coefficients in the IR basis - g\\_DLR are the coefficients in the DLR basis - fitmat is the transformation matrix

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    The input and output arrays must be contiguous in memory

!!! note

    The transformation is a direct matrix multiplication, which is typically faster than the inverse transformation

# Arguments
* `dlr`: Pointer to the DLR object
* `order`: Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref))
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `target_dim`: Target dimension for the transformation (0-based)
* `input`: Input array of DLR coefficients (double precision)
* `out`: Output array for the IR coefficients (double precision)
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
spir\\_ir2dlr
"""
function spir_dlr2ir_dd(dlr, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_dlr2ir_dd, libsparseir), Cint, (Ptr{spir_basis}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), dlr, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_dlr2ir_zz(dlr, order, ndim, input_dims, target_dim, input, out)

Transforms coefficients from DLR basis to IR representation. This version handles complex input and output arrays.

This function converts expansion coefficients from the Discrete Lehmann Representation (DLR) basis to the Intermediate Representation (IR) basis. The transformation is performed using the fitting matrix:

g\\_IR = fitmat * g\\_DLR

where: - g\\_IR are the coefficients in the IR basis - g\\_DLR are the coefficients in the DLR basis - fitmat is the transformation matrix

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    The input and output arrays must be contiguous in memory

!!! note

    The transformation is a direct matrix multiplication, which is typically faster than the inverse transformation

# Arguments
* `dlr`: Pointer to the DLR object
* `order`: Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref))
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `target_dim`: Target dimension for the transformation (0-based)
* `input`: Input array of DLR coefficients (complex)
* `out`: Output array for the IR coefficients (complex)
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_ir2dlr_zz`](@ref), [`spir_dlr2ir_dd`](@ref)
"""
function spir_dlr2ir_zz(dlr, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_dlr2ir_zz, libsparseir), Cint, (Ptr{spir_basis}, Cint, Cint, Ptr{Cint}, Cint, Ptr{c_complex}, Ptr{c_complex}), dlr, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_tau_sampling_new(b, num_points, points, status)

Creates a new tau sampling object for sparse sampling in imaginary time with custom sampling points.

Constructs a sampling object that allows transformation between the IR basis and a user-specified set of sampling points in imaginary time (τ). The sampling points are provided by the user, allowing for custom sampling strategies.

!!! note

    The sampling points should be chosen to ensure numerical stability and accuracy for the given basis

!!! note

    The sampling matrix is automatically factorized using SVD for efficient transformations

!!! note

    The returned object must be freed using spir\\_release\\_sampling when no longer needed

# Arguments
* `b`: Pointer to a finite temperature basis object
* `num_points`: Number of sampling points
* `points`: Array of sampling points in imaginary time (τ)
* `status`: Pointer to store the status code
# Returns
Pointer to the newly created sampling object, or NULL if creation fails
# See also
spir\\_release\\_sampling
"""
function spir_tau_sampling_new(b, num_points, points, status)
    ccall((:spir_tau_sampling_new, libsparseir), Ptr{spir_sampling}, (Ptr{spir_basis}, Cint, Ptr{Cdouble}, Ptr{Cint}), b, num_points, points, status)
end

"""
    spir_tau_sampling_new_with_matrix(order, statistics, basis_size, num_points, points, matrix, status)

Creates a new tau sampling object for sparse sampling in imaginary time with custom sampling points and a pre-computed matrix.

This function creates a sampling object that allows transformation between the IR basis and a user-specified set of sampling points in imaginary time (τ). The sampling points are provided by the user, allowing for custom sampling strategies.

# Arguments
* `order`: Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref))
* `statistics`: Statistics type ([`SPIR_STATISTICS_FERMIONIC`](@ref) or [`SPIR_STATISTICS_BOSONIC`](@ref))
* `basis_size`: Basis size
* `num_points`: Number of sampling points
* `points`: Array of sampling points in imaginary time (τ)
* `matrix`: Pre-computed matrix for the sampling points (num\\_points x basis\\_size). For Matsubara sampling, this should be a complex matrix.
* `status`: Pointer to store the status code
# Returns
Pointer to the newly created sampling object, or NULL if creation fails
"""
function spir_tau_sampling_new_with_matrix(order, statistics, basis_size, num_points, points, matrix, status)
    ccall((:spir_tau_sampling_new_with_matrix, libsparseir), Ptr{spir_sampling}, (Cint, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}), order, statistics, basis_size, num_points, points, matrix, status)
end

"""
    spir_matsu_sampling_new(b, positive_only, num_points, points, status)

Creates a new Matsubara sampling object for sparse sampling in Matsubara frequencies with custom sampling points.

Constructs a sampling object that allows transformation between the IR basis and a user-specified set of sampling points in Matsubara frequencies (iωn). The sampling points are provided by the user, allowing for custom sampling strategies.

# Arguments
* `b`: Pointer to a finite temperature basis object
* `positive_only`: If true, only positive frequencies are used
* `num_points`: Number of sampling points
* `points`: Array of Matsubara frequency indices (n) for the sampling points
* `status`: Pointer to store the status code
# Returns
Pointer to the newly created sampling object, or NULL if creation fails
"""
function spir_matsu_sampling_new(b, positive_only, num_points, points, status)
    ccall((:spir_matsu_sampling_new, libsparseir), Ptr{spir_sampling}, (Ptr{spir_basis}, Bool, Cint, Ptr{Int64}, Ptr{Cint}), b, positive_only, num_points, points, status)
end

"""
    spir_matsu_sampling_new_with_matrix(order, statistics, basis_size, positive_only, num_points, points, matrix, status)

Creates a new Matsubara sampling object for sparse sampling in Matsubara frequencies with custom sampling points and a pre-computed evaluation matrix.

This function creates a sampling object that can be used to evaluate and fit functions at specific Matsubara frequencies. The sampling points and evaluation matrix are provided directly, allowing for custom sampling configurations.

# Arguments
* `order`: Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref))
* `statistics`: Statistics type ([`SPIR_STATISTICS_FERMIONIC`](@ref) or [`SPIR_STATISTICS_BOSONIC`](@ref))
* `basis_size`: Basis size
* `positive_only`: If true, only positive Matsubara frequencies are used
* `num_points`: Number of sampling points
* `points`: Array of Matsubara frequencies (integer indices)
* `matrix`: Pre-computed evaluation matrix of size (num\\_points × basis\\_size)
* `status`: Pointer to store the status code
# Returns
Pointer to the new sampling object, or NULL if creation fails
# See also
[`spir_matsu_sampling_new`](@ref)
"""
function spir_matsu_sampling_new_with_matrix(order, statistics, basis_size, positive_only, num_points, points, matrix, status)
    ccall((:spir_matsu_sampling_new_with_matrix, libsparseir), Ptr{spir_sampling}, (Cint, Cint, Cint, Bool, Cint, Ptr{Int64}, Ptr{c_complex}, Ptr{Cint}), order, statistics, basis_size, positive_only, num_points, points, matrix, status)
end

"""
    spir_sampling_get_npoints(s, num_points)

Gets the number of sampling points in a sampling object.

This function returns the number of sampling points used in the specified sampling object. This number is needed to allocate arrays of the correct size when retrieving the actual sampling points.

# Arguments
* `s`: Pointer to the sampling object
* `num_points`: Pointer to store the number of sampling points
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_sampling_get_taus`](@ref), [`spir_sampling_get_matsus`](@ref)
"""
function spir_sampling_get_npoints(s, num_points)
    ccall((:spir_sampling_get_npoints, libsparseir), Cint, (Ptr{spir_sampling}, Ptr{Cint}), s, num_points)
end

"""
    spir_sampling_get_taus(s, points)

Gets the imaginary time sampling points.

This function fills the provided array with the imaginary time (τ) sampling points used in the specified sampling object. The array must be pre-allocated with sufficient size (use [`spir_sampling_get_npoints`](@ref) to determine the required size).

!!! note

    The array must be pre-allocated with size >= [`spir_sampling_get_npoints`](@ref)(s)

# Arguments
* `s`: Pointer to the sampling object
* `points`: Pre-allocated array to store the τ sampling points
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_sampling_get_npoints`](@ref)
"""
function spir_sampling_get_taus(s, points)
    ccall((:spir_sampling_get_taus, libsparseir), Cint, (Ptr{spir_sampling}, Ptr{Cdouble}), s, points)
end

"""
    spir_sampling_get_matsus(s, points)

Gets the Matsubara frequency sampling points.

This function fills the provided array with the Matsubara frequency indices (n) used in the specified sampling object. The actual Matsubara frequencies are ωn = (2n + 1)π/β for fermionic case and ωn = 2nπ/β for bosonic case. The array must be pre-allocated with sufficient size (use [`spir_sampling_get_npoints`](@ref) to determine the required size).

!!! note

    The array must be pre-allocated with size >= [`spir_sampling_get_npoints`](@ref)(s)

!!! note

    For fermionic case, the indices n give frequencies ωn = (2n + 1)π/β

!!! note

    For bosonic case, the indices n give frequencies ωn = 2nπ/β

# Arguments
* `s`: Pointer to the sampling object
* `points`: Pre-allocated array to store the Matsubara frequency indices
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_sampling_get_npoints`](@ref)
"""
function spir_sampling_get_matsus(s, points)
    ccall((:spir_sampling_get_matsus, libsparseir), Cint, (Ptr{spir_sampling}, Ptr{Int64}), s, points)
end

"""
    spir_sampling_get_cond_num(s, cond_num)

Gets the condition number of the sampling matrix.

This function returns the condition number of the sampling matrix used in the specified sampling object. The condition number is a measure of how well- conditioned the sampling matrix is.

!!! note

    A large condition number indicates that the sampling matrix is ill-conditioned, which may lead to numerical instability in transformations

!!! note

    The condition number is the ratio of the largest to smallest singular value of the sampling matrix

# Arguments
* `s`: Pointer to the sampling object
* `cond_num`: Pointer to store the condition number
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
"""
function spir_sampling_get_cond_num(s, cond_num)
    ccall((:spir_sampling_get_cond_num, libsparseir), Cint, (Ptr{spir_sampling}, Ptr{Cdouble}), s, cond_num)
end

"""
    spir_sampling_eval_dd(s, order, ndim, input_dims, target_dim, input, out)

Evaluates basis coefficients at sampling points (double to double version).

Transforms basis coefficients to values at sampling points, where both input and output are real (double precision) values. The operation can be performed along any dimension of a multidimensional array.

!!! note

    For optimal performance, the target dimension should be either the first (0) or the last (ndim-1) dimension to avoid large temporary array allocations

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    The input and output arrays must be contiguous in memory

!!! note

    The transformation is performed using a pre-computed sampling matrix that is factorized using SVD for efficiency

# Arguments
* `s`: Pointer to the sampling object
* `order`: Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref))
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `target_dim`: Target dimension for the transformation (0-based)
* `input`: Input array of basis coefficients
* `out`: Output array for the evaluated values at sampling points
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_sampling_eval_dz`](@ref), [`spir_sampling_eval_zz`](@ref)
"""
function spir_sampling_eval_dd(s, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_dd, libsparseir), Cint, (Ptr{spir_sampling}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), s, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_eval_dz(s, order, ndim, input_dims, target_dim, input, out)

Evaluates basis coefficients at sampling points (double to complex version).

For more details, see [`spir_sampling_eval_dd`](@ref)

# See also
[`spir_sampling_eval_dd`](@ref)
"""
function spir_sampling_eval_dz(s, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_dz, libsparseir), Cint, (Ptr{spir_sampling}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{c_complex}), s, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_eval_zz(s, order, ndim, input_dims, target_dim, input, out)

Evaluates basis coefficients at sampling points (complex to complex version).

For more details, see [`spir_sampling_eval_dd`](@ref)

# See also
[`spir_sampling_eval_dd`](@ref)
"""
function spir_sampling_eval_zz(s, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_zz, libsparseir), Cint, (Ptr{spir_sampling}, Cint, Cint, Ptr{Cint}, Cint, Ptr{c_complex}, Ptr{c_complex}), s, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_fit_dd(s, order, ndim, input_dims, target_dim, input, out)

Fits values at sampling points to basis coefficients (double to double version).

Transforms values at sampling points back to basis coefficients, where both input and output are real (double precision) values. The operation can be performed along any dimension of a multidimensional array.

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    This function performs the inverse operation of [`spir_sampling_eval_dd`](@ref)

!!! note

    The transformation is performed using a pre-computed sampling matrix that is factorized using SVD for efficiency

# Arguments
* `s`: Pointer to the sampling object
* `order`: Memory layout order ([`SPIR_ORDER_ROW_MAJOR`](@ref) or [`SPIR_ORDER_COLUMN_MAJOR`](@ref))
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `target_dim`: Target dimension for the transformation (0-based)
* `input`: Input array of values at sampling points
* `out`: Output array for the fitted basis coefficients
# Returns
An integer status code: - 0 ([`SPIR_COMPUTATION_SUCCESS`](@ref)) on success - A non-zero error code on failure
# See also
[`spir_sampling_eval_dd`](@ref), [`spir_sampling_fit_zz`](@ref)
"""
function spir_sampling_fit_dd(s, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_fit_dd, libsparseir), Cint, (Ptr{spir_sampling}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), s, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_fit_zz(s, order, ndim, input_dims, target_dim, input, out)

Fits values at sampling points to basis coefficients (complex to complex version).

For more details, see [`spir_sampling_fit_dd`](@ref)

# See also
[`spir_sampling_fit_dd`](@ref)
"""
function spir_sampling_fit_zz(s, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_fit_zz, libsparseir), Cint, (Ptr{spir_sampling}, Cint, Cint, Ptr{Cint}, Cint, Ptr{c_complex}, Ptr{c_complex}), s, order, ndim, input_dims, target_dim, input, out)
end

const SPIR_COMPUTATION_SUCCESS = 0

const SPIR_GET_IMPL_FAILED = -1

const SPIR_INVALID_DIMENSION = -2

const SPIR_INPUT_DIMENSION_MISMATCH = -3

const SPIR_OUTPUT_DIMENSION_MISMATCH = -4

const SPIR_NOT_SUPPORTED = -5

const SPIR_INVALID_ARGUMENT = -6

const SPIR_INTERNAL_ERROR = -7

const SPIR_STATISTICS_FERMIONIC = 1

const SPIR_STATISTICS_BOSONIC = 0

const SPIR_ORDER_COLUMN_MAJOR = 1

const SPIR_ORDER_ROW_MAJOR = 0

const SPIR_TWORK_FLOAT64 = 0

const SPIR_TWORK_FLOAT64X2 = 1

const SPARSEIR_VERSION_MAJOR = 0

const SPARSEIR_VERSION_MINOR = 4

const SPARSEIR_VERSION_PATCH = 0

# exports
const PREFIXES = ["spir_", "SPIR_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
