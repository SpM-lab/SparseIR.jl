@testitem "spir_funcs_deriv" tags=[:cinterface] begin
    using SparseIR.C_API: C_API
    using Test

@testset "C_API: spir_funcs_deriv" begin
    # Create a basis for testing
    lambda = 10.0
    eps = 1e-6
    beta = 10.0
    wmax = 1.0
    
    status = Ref{C_API.StatusCode}(C_API.SPIR_COMPUTATION_SUCCESS)
    
    # Create kernel and SVE
    kernel = C_API.spir_logistic_kernel_new(lambda, status)
    @test status[] == C_API.SPIR_COMPUTATION_SUCCESS
    @test kernel != C_NULL
    
    sve = C_API.spir_sve_result_new(kernel, eps, -1, -1, C_API.SPIR_TWORK_FLOAT64, status)
    @test status[] == C_API.SPIR_COMPUTATION_SUCCESS
    @test sve != C_NULL
    
    # Create basis
    basis = C_API.spir_basis_new(
        C_API.SPIR_STATISTICS_FERMIONIC,
        beta,
        wmax,
        eps,
        kernel,
        sve,
        -1,
        status
    )
    @test status[] == C_API.SPIR_COMPUTATION_SUCCESS
    @test basis != C_NULL
    
    # Get u functions
    u_funcs = C_API.spir_basis_get_u(basis, status)
    @test status[] == C_API.SPIR_COMPUTATION_SUCCESS
    @test u_funcs != C_NULL
    
    @testset "Basic derivative computation" begin
        # Test n=0 (should return clone)
        deriv0 = C_API.spir_funcs_deriv(u_funcs, 0, status)
        @test status[] == C_API.SPIR_COMPUTATION_SUCCESS
        @test deriv0 != C_NULL
        
        # Test n=1 (first derivative)
        deriv1 = C_API.spir_funcs_deriv(u_funcs, 1, status)
        @test status[] == C_API.SPIR_COMPUTATION_SUCCESS
        @test deriv1 != C_NULL
        
        # Test n=2 (second derivative)
        deriv2 = C_API.spir_funcs_deriv(u_funcs, 2, status)
        @test status[] == C_API.SPIR_COMPUTATION_SUCCESS
        @test deriv2 != C_NULL
        
        # Verify different objects
        @test deriv0 != deriv1
        @test deriv1 != deriv2
        
        # Cleanup
        C_API.spir_funcs_release(deriv2)
        C_API.spir_funcs_release(deriv1)
        C_API.spir_funcs_release(deriv0)
    end
    
    @testset "Numerical consistency" begin
        # Get derivative
        deriv1 = C_API.spir_funcs_deriv(u_funcs, 1, status)
        @test status[] == C_API.SPIR_COMPUTATION_SUCCESS
        
        # Get basis size
        basis_size = Ref{Cint}(0)
        C_API.spir_basis_get_size(basis, basis_size)
        n = basis_size[]
        
        # Test at several points
        h = 1e-8
        test_points = [0.2, 0.5, 0.8]
        
        for x in test_points
            f_plus = zeros(n)
            f_minus = zeros(n)
            deriv_analytical = zeros(n)
            
            C_API.spir_funcs_eval(u_funcs, x + h, f_plus)
            C_API.spir_funcs_eval(u_funcs, x - h, f_minus)
            C_API.spir_funcs_eval(deriv1, x, deriv_analytical)
            
            # Check first few basis functions
            for i in 1:min(3, n)
                numerical = (f_plus[i] - f_minus[i]) / (2.0 * h)
                analytical = deriv_analytical[i]
                
                rel_error = if abs(analytical) > 1e-10
                    abs((numerical - analytical) / analytical)
                else
                    abs(numerical - analytical)
                end
                
                @test rel_error < 1e-4
            end
        end
        
        C_API.spir_funcs_release(deriv1)
    end
    
    @testset "Error handling" begin
        # Test NULL pointer
        result = C_API.spir_funcs_deriv(C_NULL, 1, status)
        @test status[] == C_API.SPIR_INVALID_ARGUMENT
        @test result == C_NULL
        
        # Test negative derivative order
        status[] = C_API.SPIR_COMPUTATION_SUCCESS
        result = C_API.spir_funcs_deriv(u_funcs, -1, status)
        @test status[] == C_API.SPIR_INVALID_ARGUMENT
        @test result == C_NULL
    end
    
    # Cleanup
    C_API.spir_basis_release(basis)
    C_API.spir_sve_result_release(sve)
    C_API.spir_kernel_release(kernel)
end

end  # @testitem
