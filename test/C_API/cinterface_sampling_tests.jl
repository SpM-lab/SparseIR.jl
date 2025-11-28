# Tests corresponding to test/cpp/cinterface_sampling.cxx
# Comprehensive sampling functionality tests including TauSampling and MatsubaraSampling

@testitem "TauSampling" tags=[:cinterface] begin
    using SparseIR

    # Helper function to create tau sampling (corresponds to C++ create_tau_sampling)
    function create_tau_sampling(basis::Ptr{SparseIR.spir_basis})
        status = Ref{Int32}(0)

        n_tau_points_ref = Ref{Int32}(0)
        points_status = SparseIR.spir_basis_get_n_default_taus(basis, n_tau_points_ref)
        @test points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_tau_points = n_tau_points_ref[]

        tau_points = Vector{Float64}(undef, n_tau_points)
        get_points_status = SparseIR.spir_basis_get_default_taus(basis, tau_points)
        @test get_points_status == SparseIR.SPIR_COMPUTATION_SUCCESS

        sampling = SparseIR.spir_tau_sampling_new(basis, n_tau_points, tau_points, status)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test sampling != C_NULL

        return sampling
    end

    # Test tau sampling constructor (corresponds to C++ test_tau_sampling template)
    function test_tau_sampling(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-15

        # Create basis using kernel and SVE (equivalent to _spir_basis_new)
        kernel_status = Ref{Int32}(0)
        kernel = SparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = SparseIR.spir_sve_result_new(kernel, epsilon, typemax(Int32), -1, SparseIR.SPIR_TWORK_AUTO, sve_status)
        @test sve_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = SparseIR.spir_basis_new(statistics, beta, wmax, epsilon, kernel, sve, -1, basis_status)
        @test basis_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis != C_NULL

        # Get tau points
        n_tau_points_ref = Ref{Int32}(0)
        status = SparseIR.spir_basis_get_n_default_taus(basis, n_tau_points_ref)
        @test status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_tau_points = n_tau_points_ref[]
        @test n_tau_points > 0

        tau_points_org = Vector{Float64}(undef, n_tau_points)
        tau_status = SparseIR.spir_basis_get_default_taus(basis, tau_points_org)
        @test tau_status == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Create sampling
        sampling_status = Ref{Int32}(0)
        sampling = SparseIR.spir_tau_sampling_new(
            basis, n_tau_points, tau_points_org, sampling_status)
        @test sampling_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test sampling != C_NULL

        # Test getting number of sampling points
        n_points_ref = Ref{Int32}(0)
        points_status = SparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]
        @test n_points > 0

        # Test getting sampling points
        tau_points = Vector{Float64}(undef, n_points)
        get_tau_status = SparseIR.spir_sampling_get_taus(sampling, tau_points)
        @test get_tau_status == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Compare tau_points and tau_points_org (corresponds to C++ comparison)
        for i in 1:n_points
            @test tau_points[i]≈tau_points_org[i] atol=1e-14
        end

        # Test that matsubara points are not supported for tau sampling
        matsubara_points = Vector{Int64}(undef, n_points)
        matsubara_status = SparseIR.spir_sampling_get_matsus(sampling, matsubara_points)
        @test matsubara_status == SparseIR.SPIR_NOT_SUPPORTED

        # Clean up
        SparseIR.spir_sampling_release(sampling)
        SparseIR.spir_basis_release(basis)
        SparseIR.spir_sve_result_release(sve)
        SparseIR.spir_kernel_release(kernel)
    end

    # Test 1D evaluation (corresponds to C++ test_tau_sampling_evaluation_1d_column_major)
    function test_tau_sampling_evaluation_1d(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-10

        # Create basis
        kernel_status = Ref{Int32}(0)
        kernel = SparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = SparseIR.spir_sve_result_new(kernel, epsilon, typemax(Int32), -1, SparseIR.SPIR_TWORK_AUTO, sve_status)
        @test sve_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = SparseIR.spir_basis_new(statistics, beta, wmax, epsilon, kernel, sve, -1, basis_status)
        @test basis_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Create sampling
        sampling = create_tau_sampling(basis)

        # Get basis and sampling sizes
        basis_size_ref = Ref{Int32}(0)
        size_status = SparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test size_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        basis_size = basis_size_ref[]

        n_points_ref = Ref{Int32}(0)
        points_status = SparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]

        # Set up parameters for evaluation
        ndim = 1
        dims = Int32[basis_size]
        target_dim = 0

        # Create test coefficients
        coeffs = rand(Float64, basis_size) .- 0.5

        # Test evaluation
        evaluate_output = Vector{Float64}(undef, n_points)
        evaluate_status = SparseIR.spir_sampling_eval_dd(
            sampling, SparseIR.SPIR_ORDER_COLUMN_MAJOR, ndim,
            dims, target_dim, coeffs, evaluate_output)
        @test evaluate_status == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Test fitting
        fit_output = Vector{Float64}(undef, basis_size)
        fit_status = SparseIR.spir_sampling_fit_dd(
            sampling, SparseIR.SPIR_ORDER_COLUMN_MAJOR, ndim, dims,
            target_dim, evaluate_output, fit_output)
        @test fit_status == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Check round-trip accuracy
        for i in 1:basis_size
            @test fit_output[i]≈coeffs[i] atol=1e-10
        end

        # Clean up
        SparseIR.spir_sampling_release(sampling)
        SparseIR.spir_basis_release(basis)
        SparseIR.spir_sve_result_release(sve)
        SparseIR.spir_kernel_release(kernel)
    end

    # Test 4D evaluation with row-major layout (corresponds to C++ test_tau_sampling_evaluation_4d_row_major)
    function test_tau_sampling_evaluation_4d_row_major(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-10

        # Create basis
        kernel_status = Ref{Int32}(0)
        kernel = SparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = SparseIR.spir_sve_result_new(kernel, epsilon, typemax(Int32), -1, SparseIR.SPIR_TWORK_AUTO, sve_status)
        @test sve_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = SparseIR.spir_basis_new(statistics, beta, wmax, epsilon, kernel, sve, -1, basis_status)
        @test basis_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Create sampling
        sampling = create_tau_sampling(basis)

        # Get basis and sampling sizes
        basis_size_ref = Ref{Int32}(0)
        size_status = SparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test size_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        basis_size = basis_size_ref[]

        n_points_ref = Ref{Int32}(0)
        points_status = SparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]

        # Set up 4D tensor dimensions
        d1, d2, d3 = 2, 3, 4
        ndim = 4

        # Test evaluation and fitting along each dimension
        for dim in 0:3
            # Create dimension arrays for different target dimensions
            if dim == 0
                dims = Int32[basis_size, d1, d2, d3]
                output_dims = Int32[n_points, d1, d2, d3]
            elseif dim == 1
                dims = Int32[d1, basis_size, d2, d3]
                output_dims = Int32[d1, n_points, d2, d3]
            elseif dim == 2
                dims = Int32[d1, d2, basis_size, d3]
                output_dims = Int32[d1, d2, n_points, d3]
            else # dim == 3
                dims = Int32[d1, d2, d3, basis_size]
                output_dims = Int32[d1, d2, d3, n_points]
            end

            target_dim = dim
            total_size = prod(dims)
            output_total_size = prod(output_dims)

            # Create random test data (row-major layout)
            coeffs = rand(Float64, total_size) .- 0.5

            # Test evaluation
            evaluate_output = Vector{Float64}(undef, output_total_size)
            evaluate_status = SparseIR.spir_sampling_eval_dd(
                sampling, SparseIR.SPIR_ORDER_ROW_MAJOR, ndim,
                dims, target_dim, coeffs, evaluate_output)
            @test evaluate_status == SparseIR.SPIR_COMPUTATION_SUCCESS

            # Test fitting
            fit_output = Vector{Float64}(undef, total_size)
            fit_status = SparseIR.spir_sampling_fit_dd(
                sampling, SparseIR.SPIR_ORDER_ROW_MAJOR, ndim, output_dims,
                target_dim, evaluate_output, fit_output)
            @test fit_status == SparseIR.SPIR_COMPUTATION_SUCCESS

            # Check round-trip accuracy
            for i in 1:total_size
                @test fit_output[i]≈coeffs[i] atol=1e-10
            end
        end

        # Clean up
        SparseIR.spir_sampling_release(sampling)
        SparseIR.spir_basis_release(basis)
        SparseIR.spir_sve_result_release(sve)
        SparseIR.spir_kernel_release(kernel)
    end

    # Test 4D evaluation with column-major layout (corresponds to C++ test_tau_sampling_evaluation_4d_column_major)
    function test_tau_sampling_evaluation_4d_column_major(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-10

        # Create basis
        kernel_status = Ref{Int32}(0)
        kernel = SparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = SparseIR.spir_sve_result_new(kernel, epsilon, typemax(Int32), -1, SparseIR.SPIR_TWORK_AUTO, sve_status)
        @test sve_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = SparseIR.spir_basis_new(statistics, beta, wmax, epsilon, kernel, sve, -1, basis_status)
        @test basis_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Create sampling
        sampling = create_tau_sampling(basis)

        # Get basis and sampling sizes
        basis_size_ref = Ref{Int32}(0)
        size_status = SparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test size_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        basis_size = basis_size_ref[]

        n_points_ref = Ref{Int32}(0)
        points_status = SparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]

        # Set up 4D tensor dimensions
        d1, d2, d3 = 2, 3, 4
        ndim = 4

        # Test evaluation and fitting along each dimension
        for dim in 0:3
            # Create dimension arrays for different target dimensions
            if dim == 0
                dims = Int32[basis_size, d1, d2, d3]
                output_dims = Int32[n_points, d1, d2, d3]
            elseif dim == 1
                dims = Int32[d1, basis_size, d2, d3]
                output_dims = Int32[d1, n_points, d2, d3]
            elseif dim == 2
                dims = Int32[d1, d2, basis_size, d3]
                output_dims = Int32[d1, d2, n_points, d3]
            else # dim == 3
                dims = Int32[d1, d2, d3, basis_size]
                output_dims = Int32[d1, d2, d3, n_points]
            end

            target_dim = dim
            total_size = prod(dims)
            output_total_size = prod(output_dims)

            # Create random test data (column-major layout)
            coeffs = rand(Float64, total_size) .- 0.5

            # Test evaluation
            evaluate_output = Vector{Float64}(undef, output_total_size)
            evaluate_status = SparseIR.spir_sampling_eval_dd(
                sampling, SparseIR.SPIR_ORDER_COLUMN_MAJOR, ndim,
                dims, target_dim, coeffs, evaluate_output)
            @test evaluate_status == SparseIR.SPIR_COMPUTATION_SUCCESS

            # Test fitting
            fit_output = Vector{Float64}(undef, total_size)
            fit_status = SparseIR.spir_sampling_fit_dd(
                sampling, SparseIR.SPIR_ORDER_COLUMN_MAJOR, ndim, output_dims,
                target_dim, evaluate_output, fit_output)
            @test fit_status == SparseIR.SPIR_COMPUTATION_SUCCESS

            # Check round-trip accuracy
            for i in 1:total_size
                @test fit_output[i]≈coeffs[i] atol=1e-10
            end
        end

        # Clean up
        SparseIR.spir_sampling_release(sampling)
        SparseIR.spir_basis_release(basis)
        SparseIR.spir_sve_result_release(sve)
        SparseIR.spir_kernel_release(kernel)
    end

    # Test 4D evaluation with complex data and row-major layout (corresponds to C++ test_tau_sampling_evaluation_4d_row_major_complex)
    function test_tau_sampling_evaluation_4d_row_major_complex(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-10

        # Create basis
        kernel_status = Ref{Int32}(0)
        kernel = SparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = SparseIR.spir_sve_result_new(kernel, epsilon, typemax(Int32), -1, SparseIR.SPIR_TWORK_AUTO, sve_status)
        @test sve_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = SparseIR.spir_basis_new(statistics, beta, wmax, epsilon, kernel, sve, -1, basis_status)
        @test basis_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Create sampling
        sampling = create_tau_sampling(basis)

        # Get basis and sampling sizes
        basis_size_ref = Ref{Int32}(0)
        size_status = SparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test size_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        basis_size = basis_size_ref[]

        n_points_ref = Ref{Int32}(0)
        points_status = SparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]

        # Set up 4D tensor dimensions
        d1, d2, d3 = 2, 3, 4
        ndim = 4

        # Test evaluation and fitting along each dimension
        for dim in 0:3
            # Create dimension arrays for different target dimensions
            if dim == 0
                dims = Int32[basis_size, d1, d2, d3]
                output_dims = Int32[n_points, d1, d2, d3]
            elseif dim == 1
                dims = Int32[d1, basis_size, d2, d3]
                output_dims = Int32[d1, n_points, d2, d3]
            elseif dim == 2
                dims = Int32[d1, d2, basis_size, d3]
                output_dims = Int32[d1, d2, n_points, d3]
            else # dim == 3
                dims = Int32[d1, d2, d3, basis_size]
                output_dims = Int32[d1, d2, d3, n_points]
            end

            target_dim = dim
            total_size = prod(dims)
            output_total_size = prod(output_dims)

            # Create random complex test data (row-major layout)
            coeffs_real = rand(Float64, total_size) .- 0.5
            coeffs_imag = rand(Float64, total_size) .- 0.5
            coeffs = [ComplexF64(coeffs_real[i], coeffs_imag[i]) for i in 1:total_size]

            # Test evaluation
            evaluate_output = Vector{ComplexF64}(undef, output_total_size)
            evaluate_status = SparseIR.spir_sampling_eval_zz(
                sampling, SparseIR.SPIR_ORDER_ROW_MAJOR, ndim,
                dims, target_dim, coeffs, evaluate_output)
            @test evaluate_status == SparseIR.SPIR_COMPUTATION_SUCCESS

            # Test fitting
            fit_output = Vector{ComplexF64}(undef, total_size)
            fit_status = SparseIR.spir_sampling_fit_zz(
                sampling, SparseIR.SPIR_ORDER_ROW_MAJOR, ndim, output_dims,
                target_dim, evaluate_output, fit_output)
            @test fit_status == SparseIR.SPIR_COMPUTATION_SUCCESS

            # Check round-trip accuracy
            for i in 1:total_size
                @test real(fit_output[i])≈real(coeffs[i]) atol=1e-10
                @test imag(fit_output[i])≈imag(coeffs[i]) atol=1e-10
            end
        end

        # Clean up
        SparseIR.spir_sampling_release(sampling)
        SparseIR.spir_basis_release(basis)
        SparseIR.spir_sve_result_release(sve)
        SparseIR.spir_kernel_release(kernel)
    end

    # Test 4D evaluation with complex data and column-major layout (corresponds to C++ test_tau_sampling_evaluation_4d_column_major_complex)
    function test_tau_sampling_evaluation_4d_column_major_complex(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-10

        # Create basis
        kernel_status = Ref{Int32}(0)
        kernel = SparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = SparseIR.spir_sve_result_new(kernel, epsilon, typemax(Int32), -1, SparseIR.SPIR_TWORK_AUTO, sve_status)
        @test sve_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = SparseIR.spir_basis_new(statistics, beta, wmax, epsilon, kernel, sve, -1, basis_status)
        @test basis_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Create sampling
        sampling = create_tau_sampling(basis)

        # Get basis and sampling sizes
        basis_size_ref = Ref{Int32}(0)
        size_status = SparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test size_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        basis_size = basis_size_ref[]

        n_points_ref = Ref{Int32}(0)
        points_status = SparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]

        # Set up 4D tensor dimensions
        d1, d2, d3 = 2, 3, 4
        ndim = 4

        # Test evaluation and fitting along each dimension
        for dim in 0:3
            # Create dimension arrays for different target dimensions
            if dim == 0
                dims = Int32[basis_size, d1, d2, d3]
                output_dims = Int32[n_points, d1, d2, d3]
            elseif dim == 1
                dims = Int32[d1, basis_size, d2, d3]
                output_dims = Int32[d1, n_points, d2, d3]
            elseif dim == 2
                dims = Int32[d1, d2, basis_size, d3]
                output_dims = Int32[d1, d2, n_points, d3]
            else # dim == 3
                dims = Int32[d1, d2, d3, basis_size]
                output_dims = Int32[d1, d2, d3, n_points]
            end

            target_dim = dim
            total_size = prod(dims)
            output_total_size = prod(output_dims)

            # Create random complex test data (column-major layout)
            coeffs_real = rand(Float64, total_size) .- 0.5
            coeffs_imag = rand(Float64, total_size) .- 0.5
            coeffs = [ComplexF64(coeffs_real[i], coeffs_imag[i]) for i in 1:total_size]

            # Test evaluation
            evaluate_output = Vector{ComplexF64}(undef, output_total_size)
            evaluate_status = SparseIR.spir_sampling_eval_zz(
                sampling, SparseIR.SPIR_ORDER_COLUMN_MAJOR, ndim,
                dims, target_dim, coeffs, evaluate_output)
            @test evaluate_status == SparseIR.SPIR_COMPUTATION_SUCCESS

            # Test fitting
            fit_output = Vector{ComplexF64}(undef, total_size)
            fit_status = SparseIR.spir_sampling_fit_zz(
                sampling, SparseIR.SPIR_ORDER_COLUMN_MAJOR, ndim, output_dims,
                target_dim, evaluate_output, fit_output)
            @test fit_status == SparseIR.SPIR_COMPUTATION_SUCCESS

            # Check round-trip accuracy
            for i in 1:total_size
                @test real(fit_output[i])≈real(coeffs[i]) atol=1e-10
                @test imag(fit_output[i])≈imag(coeffs[i]) atol=1e-10
            end
        end

        # Clean up
        SparseIR.spir_sampling_release(sampling)
        SparseIR.spir_basis_release(basis)
        SparseIR.spir_sve_result_release(sve)
        SparseIR.spir_kernel_release(kernel)
    end

    # Test error status handling (corresponds to C++ TauSampling Error Status section)
    function test_tau_sampling_error_status(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-10

        # Create basis
        kernel_status = Ref{Int32}(0)
        kernel = SparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = SparseIR.spir_sve_result_new(kernel, epsilon, typemax(Int32), -1, SparseIR.SPIR_TWORK_AUTO, sve_status)
        @test sve_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = SparseIR.spir_basis_new(statistics, beta, wmax, epsilon, kernel, sve, -1, basis_status)
        @test basis_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Create sampling
        sampling = create_tau_sampling(basis)

        # Get basis and sampling sizes
        basis_size_ref = Ref{Int32}(0)
        size_status = SparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test size_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        basis_size = basis_size_ref[]

        n_points_ref = Ref{Int32}(0)
        points_status = SparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]

        # Set up 4D tensor dimensions
        d1, d2, d3 = 2, 3, 4
        ndim = 4
        total_size = basis_size * d1 * d2 * d3

        # Create test data
        coeffs = rand(Float64, total_size) .- 0.5
        output_double = Vector{Float64}(undef, total_size)
        output_complex = Vector{ComplexF64}(undef, total_size)
        fit_output_double = Vector{Float64}(undef, total_size)
        fit_output_complex = Vector{ComplexF64}(undef, total_size)

        # Test dimension mismatch errors for different target dimensions
        dims1 = Int32[basis_size, d1, d2, d3]

        for dim in 1:3  # Skip dim=0 as it should work
            target_dim = dim

            # Test dimension mismatch for evaluation
            status_dimension_mismatch = SparseIR.spir_sampling_eval_dd(
                sampling, SparseIR.SPIR_ORDER_COLUMN_MAJOR, ndim,
                dims1, target_dim, coeffs, output_double)
            @test status_dimension_mismatch == SparseIR.SPIR_INPUT_DIMENSION_MISMATCH

            # Test dimension mismatch for fitting
            fit_status_dimension_mismatch = SparseIR.spir_sampling_fit_zz(
                sampling, SparseIR.SPIR_ORDER_COLUMN_MAJOR, ndim, dims1,
                target_dim, output_complex, fit_output_complex)
            @test fit_status_dimension_mismatch == SparseIR.SPIR_INPUT_DIMENSION_MISMATCH
        end

        # Clean up
        SparseIR.spir_sampling_release(sampling)
        SparseIR.spir_basis_release(basis)
        SparseIR.spir_sve_result_release(sve)
        SparseIR.spir_kernel_release(kernel)
    end

    @testset "TauSampling Constructor (fermionic)" begin
        test_tau_sampling(SparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "TauSampling Constructor (bosonic)" begin
        test_tau_sampling(SparseIR.SPIR_STATISTICS_BOSONIC)
    end

    @testset "TauSampling Evaluation 1-dimensional input COLUMN-MAJOR" begin
        test_tau_sampling_evaluation_1d(SparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "TauSampling Evaluation 4-dimensional input ROW-MAJOR" begin
        test_tau_sampling_evaluation_4d_row_major(SparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "TauSampling Evaluation 4-dimensional input COLUMN-MAJOR" begin
        test_tau_sampling_evaluation_4d_column_major(SparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "TauSampling Evaluation 4-dimensional complex input/output ROW-MAJOR" begin
        test_tau_sampling_evaluation_4d_row_major_complex(SparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "TauSampling Evaluation 4-dimensional complex input/output COLUMN-MAJOR" begin
        test_tau_sampling_evaluation_4d_column_major_complex(SparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "TauSampling Error Status" begin
        test_tau_sampling_error_status(SparseIR.SPIR_STATISTICS_FERMIONIC)
    end
end

@testitem "MatsubaraSampling" begin
    using SparseIR

    # Helper function to create matsubara sampling (corresponds to C++ create_matsubara_sampling)
    function create_matsubara_sampling(basis::Ptr{SparseIR.spir_basis}, positive_only::Bool)
        status = Ref{Int32}(0)

        n_matsubara_points_ref = Ref{Int32}(0)
        points_status = SparseIR.spir_basis_get_n_default_matsus(
            basis, positive_only, n_matsubara_points_ref)
        @test points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_matsubara_points = n_matsubara_points_ref[]

        smpl_points = Vector{Int64}(undef, n_matsubara_points)
        get_points_status = SparseIR.spir_basis_get_default_matsus(
            basis, positive_only, smpl_points)
        @test get_points_status == SparseIR.SPIR_COMPUTATION_SUCCESS

        sampling = SparseIR.spir_matsu_sampling_new(
            basis, positive_only, n_matsubara_points, smpl_points, status)
        @test status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test sampling != C_NULL

        return sampling
    end

    # Test matsubara sampling constructor (corresponds to C++ test_matsubara_sampling_constructor)
    function test_matsubara_sampling_constructor(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-10

        # Create basis
        kernel_status = Ref{Int32}(0)
        kernel = SparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = SparseIR.spir_sve_result_new(kernel, epsilon, typemax(Int32), -1, SparseIR.SPIR_TWORK_AUTO, sve_status)
        @test sve_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = SparseIR.spir_basis_new(statistics, beta, wmax, epsilon, kernel, sve, -1, basis_status)
        @test basis_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Test with positive_only = false
        n_points_org_ref = Ref{Int32}(0)
        status = SparseIR.spir_basis_get_n_default_matsus(basis, false, n_points_org_ref)
        @test status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points_org = n_points_org_ref[]
        @test n_points_org > 0

        smpl_points_org = Vector{Int64}(undef, n_points_org)
        points_status = SparseIR.spir_basis_get_default_matsus(
            basis, false, smpl_points_org)
        @test points_status == SparseIR.SPIR_COMPUTATION_SUCCESS

        sampling_status = Ref{Int32}(0)
        sampling = SparseIR.spir_matsu_sampling_new(
            basis, false, n_points_org, smpl_points_org, sampling_status)
        @test sampling_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test sampling != C_NULL

        # Test with positive_only = true
        n_points_positive_only_org_ref = Ref{Int32}(0)
        positive_status = SparseIR.spir_basis_get_n_default_matsus(
            basis, true, n_points_positive_only_org_ref)
        @test positive_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points_positive_only_org = n_points_positive_only_org_ref[]
        @test n_points_positive_only_org > 0

        smpl_points_positive_only_org = Vector{Int64}(undef, n_points_positive_only_org)
        positive_points_status = SparseIR.spir_basis_get_default_matsus(
            basis, true, smpl_points_positive_only_org)
        @test positive_points_status == SparseIR.SPIR_COMPUTATION_SUCCESS

        sampling_positive_status = Ref{Int32}(0)
        sampling_positive_only = SparseIR.spir_matsu_sampling_new(
            basis, true, n_points_positive_only_org,
            smpl_points_positive_only_org, sampling_positive_status)
        @test sampling_positive_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test sampling_positive_only != C_NULL

        # Test getting number of points
        n_points_ref = Ref{Int32}(0)
        get_points_status = SparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test get_points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]
        @test n_points > 0

        n_points_positive_only_ref = Ref{Int32}(0)
        get_positive_points_status = SparseIR.spir_sampling_get_npoints(
            sampling_positive_only, n_points_positive_only_ref)
        @test get_positive_points_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        n_points_positive_only = n_points_positive_only_ref[]
        @test n_points_positive_only > 0

        # Clean up
        SparseIR.spir_sampling_release(sampling_positive_only)
        SparseIR.spir_sampling_release(sampling)
        SparseIR.spir_basis_release(basis)
        SparseIR.spir_sve_result_release(sve)
        SparseIR.spir_kernel_release(kernel)
    end

    @testset "MatsubaraSampling Constructor (fermionic)" begin
        test_matsubara_sampling_constructor(SparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "MatsubaraSampling Constructor (bosonic)" begin
        test_matsubara_sampling_constructor(SparseIR.SPIR_STATISTICS_BOSONIC)
    end
end
