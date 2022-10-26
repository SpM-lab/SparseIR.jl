using Test
using SparseIR

@testset "_roots.jl" begin
    @testset "discrete_extrema" begin
        nonnegative = collect(0:8)
        symmetric = collect(-8:8)
        @test SparseIR.discrete_extrema(x -> x, nonnegative) == [8]
        @test SparseIR.discrete_extrema(x -> x - eps(), nonnegative) == [0, 8]
        @test SparseIR.discrete_extrema(x -> x^2, symmetric) == [-8, 0, 8]
        @test SparseIR.discrete_extrema(x -> 1, symmetric) == []
    end

    @testset "midpoint" begin
        @test SparseIR.midpoint(typemax(Int), typemax(Int)) === typemax(Int)
        @test SparseIR.midpoint(typemin(Int), typemax(Int)) === -1
        @test SparseIR.midpoint(typemin(Int), typemin(Int)) === typemin(Int)
        @test SparseIR.midpoint(Int16(1000), Int32(2000)) === Int32(1500)
        @test SparseIR.midpoint(floatmax(Float64), floatmax(Float64)) === floatmax(Float64)
        @test SparseIR.midpoint(Float16(0), floatmax(Float32)) === floatmax(Float32) / 2
        @test SparseIR.midpoint(Float16(0), floatmax(BigFloat)) == floatmax(BigFloat) / 2
        @test SparseIR.midpoint(Int16(0), big"99999999999999999999") == big"99999999999999999999" ÷ 2
        @test SparseIR.midpoint(-10., 1) === -4.5
    end
end