# Tests corresponding to test/cpp/cinterface_dlr.cxx
# Comprehensive DLR (Discrete Lehmann Representation) functionality tests

@testitem "DiscreteLehmannRepresentation" tags=[:cinterface] begin
    using SparseIR

    # Helper function equivalent to C++ _spir_basis_new
    function _spir_basis_new(
            statistics::Integer, beta::Float64, omega_max::Float64, epsilon::Float64)
        status = Ref{Int32}(0)

        # Create logistic kernel
        kernel_status = Ref{Int32}(0)
        kernel = SparseIR.spir_logistic_kernel_new(beta * omega_max, kernel_status)
        if kernel_status[] != SparseIR.SPIR_COMPUTATION_SUCCESS || kernel == C_NULL
            return C_NULL, kernel_status[]
        end

        # Create SVE result
        sve_status = Ref{Int32}(0)
        sve = SparseIR.spir_sve_result_new(kernel, epsilon, sve_status)
        if sve_status[] != SparseIR.SPIR_COMPUTATION_SUCCESS || sve == C_NULL
            SparseIR.spir_kernel_release(kernel)
            return C_NULL, sve_status[]
        end

        # Create basis
        basis_status = Ref{Int32}(0)
        basis = SparseIR.spir_basis_new(
            statistics, beta, omega_max, kernel, sve, basis_status)
        if basis_status[] != SparseIR.SPIR_COMPUTATION_SUCCESS || basis == C_NULL
            SparseIR.spir_sve_result_release(sve)
            SparseIR.spir_kernel_release(kernel)
            return C_NULL, basis_status[]
        end

        # Clean up intermediate objects (like C++ version)
        SparseIR.spir_sve_result_release(sve)
        SparseIR.spir_kernel_release(kernel)

        return basis, SparseIR.SPIR_COMPUTATION_SUCCESS
    end

    # Template function equivalent for different statistics (corresponds to C++ template functions)
    function test_finite_temp_basis_dlr(statistics::Integer)
        beta = 10000.0  # Same as C++ version
        wmax = 1.0
        epsilon = 1e-12

        # Create IR basis using helper function (equivalent to C++ _spir_basis_new)
        basis, basis_status = _spir_basis_new(statistics, beta, wmax, epsilon)
        @test basis_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis != C_NULL

        # Get basis size
        basis_size_ref = Ref{Int32}(0)
        size_status = SparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test size_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        basis_size = basis_size_ref[]
        @test basis_size >= 0

        # Get default poles (corresponds to C++ spir_basis_get_default_ws)
        num_default_poles_ref = Ref{Int32}(0)
        poles_status = SparseIR.spir_basis_get_n_default_ws(basis, num_default_poles_ref)
        @test poles_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        num_default_poles = num_default_poles_ref[]
        @test num_default_poles >= 0

        default_poles = Vector{Float64}(undef, num_default_poles)
        get_poles_status = SparseIR.spir_basis_get_default_ws(basis, default_poles)
        @test get_poles_status == SparseIR.SPIR_COMPUTATION_SUCCESS

        # DLR constructor using default poles (corresponds to C++ spir_dlr_new)
        dlr_status = Ref{Int32}(0)
        dlr = SparseIR.spir_dlr_new(basis, dlr_status)
        @test dlr_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test dlr != C_NULL

        # DLR constructor using custom poles (corresponds to C++ spir_dlr_new_with_poles)
        dlr_with_poles_status = Ref{Int32}(0)
        dlr_with_poles = SparseIR.spir_dlr_new_with_poles(
            basis, num_default_poles, default_poles, dlr_with_poles_status)
        @test dlr_with_poles_status[] == SparseIR.SPIR_COMPUTATION_SUCCESS
        @test dlr_with_poles != C_NULL

        # Test number of poles consistency
        num_poles_ref = Ref{Int32}(0)
        npoles_status = SparseIR.spir_dlr_get_npoles(dlr, num_poles_ref)
        @test npoles_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        num_poles = num_poles_ref[]
        @test num_poles == num_default_poles

        num_poles_with_poles_ref = Ref{Int32}(0)
        npoles_with_poles_status = SparseIR.spir_dlr_get_npoles(
            dlr_with_poles, num_poles_with_poles_ref)
        @test npoles_with_poles_status == SparseIR.SPIR_COMPUTATION_SUCCESS
        num_poles_with_poles = num_poles_with_poles_ref[]
        @test num_poles_with_poles == num_default_poles

        # Test poles reconstruction (corresponds to C++ strict comparison)
        poles_reconst = Vector{Float64}(undef, num_poles)
        reconst_status = SparseIR.spir_dlr_get_poles(dlr, poles_reconst)
        @test reconst_status == SparseIR.SPIR_COMPUTATION_SUCCESS

        # Strict numerical comparison (corresponds to C++ Approx comparison)
        for i in 1:num_poles
            @test poles_reconst[i]≈default_poles[i] atol=1e-14
        end

        # Clean up (corresponds to C++ cleanup)
        SparseIR.spir_basis_release(dlr_with_poles)
        SparseIR.spir_basis_release(dlr)
        SparseIR.spir_basis_release(basis)
    end

    @testset "DiscreteLehmannRepresentation Constructor Fermionic" begin
        test_finite_temp_basis_dlr(SparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "DiscreteLehmannRepresentation Constructor Bosonic" begin
        test_finite_temp_basis_dlr(SparseIR.SPIR_STATISTICS_BOSONIC)
    end
end
