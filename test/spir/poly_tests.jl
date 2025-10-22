@testitem "poly.jl" tags=[:julia, :lib] begin
    using Test
    using SparseIR
    import SparseIR as SparseIR

    β = 10.0
    ωmax = 4.0
    ε = 1e-6
    basis = FiniteTempBasis{Fermionic}(β, ωmax, ε)

    ρ₀(ω) = 2 / π * √(1 - clamp(ω, -1, +1)^2)

    @testset "u" begin
        @test SparseIR.xmin(basis.u) == -β
        @test SparseIR.xmax(basis.u) == β
        @test basis.u.period == β

        @test basis.u[1] isa SparseIR.PiecewiseLegendrePoly
        @test basis.u[1:2] isa SparseIR.PiecewiseLegendrePolyVector
        @test basis.u[2:2] isa SparseIR.PiecewiseLegendrePoly
    end

    @testset "v" begin
        @test SparseIR.xmin(basis.v) == -ωmax
        @test SparseIR.xmax(basis.v) == ωmax
        @test basis.v.period == 0.0
    end

    @testset "overlap" begin
        using LinearAlgebra: I
        @test isapprox(SparseIR.overlap(basis.u[1], basis.u[1], 0.0, β), 1.0, atol=1e-12)
        @test isapprox(SparseIR.overlap(basis.u[1], basis.u[2], 0.0, β), 0.0, atol=1e-12)
        @test isapprox(SparseIR.overlap(basis.v[1], basis.v[1], -ωmax, ωmax), 1.0, atol=1e-12)

        N = 4
        @test isapprox(SparseIR.overlap(basis.u[1:N], basis.u[1:N], 0.0, β), Matrix(I, N, N), atol=1e-12)
    end
end
