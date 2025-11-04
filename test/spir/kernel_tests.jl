@testitem "kernel.jl" tags=[:julia, :spir] begin
    import SparseIR as SparseIR

    @testset "Logistic kernel" begin
        lam = 42
        kernel = LogisticKernel(lam)
        @test SparseIR.Λ(kernel) == lam
        @test iscentrosymmetric(kernel)
    end

    @testset "Regularized Bose kernel" begin
        lam = 42
        kernel = RegularizedBoseKernel(lam)
        @test SparseIR.Λ(kernel) == lam
        @test iscentrosymmetric(kernel)
    end
end
