using Test
using SparseIR

include("util.jl")

@testset "bessels.jl" begin
    @test typestable(SparseIR.sphericalbesselj, [Int, Float64])

    @testset "small_x" begin
        n = 11
        x = 1e0
        # Mathematica
        ref = 3.099551854790080034457495747083911933213405593516888829346e-12
        @test SparseIR.sphericalbesselj(n, x) ≈ ref
    end

    @testset "large_x" begin
        n = 11
        x = 2e9
        # Mathematica
        ref = 2.020549136012222873136295240464724116017069579717738807316e-10
        @test SparseIR.sphericalbesselj(n, x) ≈ ref
    end
end
