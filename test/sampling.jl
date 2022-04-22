using Test
using SparseIR

using Random, LinearAlgebra

#include("conftest.jl")

@testset "sampling.jl" begin
    @testset "decomp" begin
        Random.seed!(420)
        A = randn(49, 39)

        Ad = svd(A)
        norm_A = first(Ad.S) / last(Ad.S)
        @test all(isapprox.(A, Matrix(Ad), atol=1e-15 * norm_A, rtol=0))

        x = randn(39)
        @test A * x ≈ Ad.U * Diagonal(Ad.S) * Ad.Vt * x atol = 1e-14 * norm_A rtol = 0

        x = randn(39, 3)
        @test A * x ≈ Ad.U * Diagonal(Ad.S) * Ad.Vt * x atol = 1e-14 * norm_A rtol = 0

        y = randn(49)
        @test A \ y ≈ Ad \ y atol = 1e-14 * norm_A rtol = 0

        y = randn(49, 2)
        @test A \ y ≈ Ad \ y atol = 1e-14 * norm_A rtol = 0
    end

    @testset "τ noise" begin
        for stat in (boson, fermion)
            Λ = 42
            basis = IRBasis(stat, Λ; sve_result=sve_logistic[Λ])
            smpl = TauSampling(basis)
            Random.seed!(5318008)

            ρℓ = basis.v([-0.999, -0.01, 0.5]) * [0.8, -0.2, 0.5]
            Gℓ = basis.s .* ρℓ
            Gℓ_magn = norm(Gℓ)
            Gτ = evaluate(smpl, Gℓ)
            @test evaluate_opt(smpl, Gℓ) ≈ Gτ

            noise = 1e-5
            Gτ_n = Gτ + noise * norm(Gτ) * randn(size(Gτ)...)
            Gℓ_n = fit(smpl, Gτ_n)

            @test Gℓ ≈ Gℓ_n atol = 12 * noise * Gℓ_magn rtol = 0
        end
    end

    @testset "wn noise" begin
        for stat in (boson, fermion)
            Λ = 42
            basis = IRBasis(stat, Λ; sve_result=sve_logistic[Λ])
            smpl = MatsubaraSampling(basis)
            Random.seed!(1312)

            ρℓ = basis.v([-0.999, -0.01, 0.5]) * [0.8, -0.2, 0.5]
            Gℓ = basis.s .* ρℓ
            Gℓ_magn = norm(Gℓ)
            Giw = evaluate(smpl, Gℓ)

            noise = 1e-5
            Gwn_n = Giw + noise * norm(Giw) * randn(size(Giw)...)
            Gℓ_n = fit(smpl, Gwn_n)

            @test Gℓ ≈ Gℓ_n atol = 12 * noise * Gℓ_magn rtol = 0
        end
    end
end
