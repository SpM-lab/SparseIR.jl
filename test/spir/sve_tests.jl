@testitem "sve.jl" tags=[:julia, :lib] begin
    using SparseIR

    @testset "sve_result/LogisticKernel" begin
        kernel = LogisticKernel(10)
        sve_result = SparseIR.SVEResult(kernel, 1e-10)
        @test true
    end

    @testset "sve_result/RegularizedBoseKernel" begin
        kernel = RegularizedBoseKernel(10)
        sve_result = SparseIR.SVEResult(kernel, 1e-10)
        @test true
    end
end
