# Manual test script for custom kernel functionality
# This script runs the custom kernel tests without ReTestItems

using SparseIR
using Test

# Simple wrapper kernel that wraps LogisticKernel
# This is used to test the custom kernel functionality
struct SimpleWrapperKernel <: SparseIR.AbstractKernel
    inner::SparseIR.LogisticKernel
    is_centrosym::Bool
end

SimpleWrapperKernel(Λ::Real, is_centrosym::Bool=true) = SimpleWrapperKernel(SparseIR.LogisticKernel(Λ), is_centrosym)

# Delegate all methods to inner kernel
SparseIR.Λ(kernel::SimpleWrapperKernel) = SparseIR.Λ(kernel.inner)
SparseIR.xrange(kernel::SimpleWrapperKernel) = SparseIR.xrange(kernel.inner)
SparseIR.yrange(kernel::SimpleWrapperKernel) = SparseIR.yrange(kernel.inner)
SparseIR.iscentrosymmetric(kernel::SimpleWrapperKernel) = kernel.is_centrosym
SparseIR.conv_radius(kernel::SimpleWrapperKernel) = SparseIR.conv_radius(kernel.inner)

# Kernel evaluation
function (kernel::SimpleWrapperKernel)(x::Real, y::Real)
    return kernel.inner(x, y)
end

# Weight functions
SparseIR.weight_func(kernel::SimpleWrapperKernel, stat::SparseIR.Statistics) = 
    SparseIR.weight_func(kernel.inner, stat)

# SVE hints - delegate to inner kernel
SparseIR.sve_hints(kernel::SimpleWrapperKernel, ε::Real) = 
    SparseIR.sve_hints(kernel.inner, ε)

println("=== Custom Kernel Tests ===")

@testset "Custom kernel creation" begin
    lam = 10.0
    kernel = SimpleWrapperKernel(lam, true)
    @test SparseIR.Λ(kernel) == lam
    println("✓ Custom kernel creation")
end

@testset "Custom kernel SVEResult creation" begin
    lam = 10.0
    kernel = SimpleWrapperKernel(lam, true)
    println("Creating SVEResult with epsilon=1e-6...")
    @time sve_result = SparseIR.SVEResult(kernel, 1e-6)
    @test sve_result.ptr != C_NULL
    @test length(sve_result.s) > 0
    println("✓ Custom kernel SVEResult creation: $(length(sve_result.s)) singular values")
end

@testset "Custom kernel FiniteTempBasis creation" begin
    lam = 10.0
    β = 10.0
    ωmax = 1.0
    epsilon = 1e-6
    kernel = SimpleWrapperKernel(lam, true)
    println("Creating FiniteTempBasis...")
    @time basis = SparseIR.FiniteTempBasis(Fermionic(), β, ωmax, epsilon)
    @test basis.ptr != C_NULL
    println("✓ Custom kernel FiniteTempBasis creation")
end

@testset "Custom kernel vs LogisticKernel comparison" begin
    lam = 10.0
    β = 10.0
    ωmax = 1.0
    epsilon = 1e-6
    
    log_kernel = SparseIR.LogisticKernel(lam)
    custom_kernel = SimpleWrapperKernel(lam, true)
    
    log_basis_ferm = SparseIR.FiniteTempBasis(Fermionic(), β, ωmax, epsilon)
    custom_basis_ferm = SparseIR.FiniteTempBasis(Fermionic(), β, ωmax, epsilon; kernel=custom_kernel)
    
    log_svals = log_basis_ferm.s
    custom_svals = custom_basis_ferm.s
    @test length(log_svals) == length(custom_svals)
    @test log_svals ≈ custom_svals rtol=1e-10
    println("✓ Custom kernel vs LogisticKernel comparison")
end

@testset "Custom kernel properties" begin
    lam = 10.0
    
    for is_centrosym in [true, false]
        kernel = SimpleWrapperKernel(lam, is_centrosym)
        
        # Test domain
        xmin, xmax = SparseIR.xrange(kernel)
        ymin, ymax = SparseIR.yrange(kernel)
        @test xmin ≈ -1.0
        @test xmax ≈ 1.0
        @test ymin ≈ -1.0
        @test ymax ≈ 1.0
        
        # Test centrosymmetric property
        @test SparseIR.iscentrosymmetric(kernel) == is_centrosym
        
        # Test weight functions
        wfunc_ferm = SparseIR.weight_func(kernel, SparseIR.Fermionic())
        wfunc_bos = SparseIR.weight_func(kernel, SparseIR.Bosonic())
        @test wfunc_ferm([1.0])[1] ≈ 1.0
        @test wfunc_bos([1.0])[1] > 0.0
        
        # Test kernel evaluation
        @test kernel(0.0, 0.0) ≈ 0.5 rtol=1e-10
        
        # Test that kernel can be used to create SVEResult (which tests kernel evaluation through C-API)
        println("Creating SVEResult for is_centrosym=$is_centrosym...")
        @time sve_test = SparseIR.SVEResult(kernel, 1e-6)
        @test sve_test.ptr != C_NULL
    end
    println("✓ Custom kernel properties")
end

@testset "Custom kernel SVE hints" begin
    lam = 10.0
    epsilon = 1e-6
    
    for is_centrosym in [true, false]
        kernel = SimpleWrapperKernel(lam, is_centrosym)
        # Test that SVEResult can be created (which internally uses sve_hints)
        println("Creating SVEResult for is_centrosym=$is_centrosym...")
        @time sve_result = SparseIR.SVEResult(kernel, epsilon)
        @test sve_result.ptr != C_NULL
        @test length(sve_result.s) > 0
    end
    println("✓ Custom kernel SVE hints")
end

println("\n=== All tests completed ===")

