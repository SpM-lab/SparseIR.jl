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

    @testset "τ noise with stat = $stat" for stat in (boson, fermion)
        Λ = 42
        basis = DimensionlessBasis(stat, Λ; sve_result=sve_logistic[Λ])
        smpl = TauSampling(basis)
        Random.seed!(5318008)

        ρℓ = basis.v([-0.999, -0.01, 0.5]) * [0.8, -0.2, 0.5]
        Gℓ = basis.s .* ρℓ
        Gℓ_magn = norm(Gℓ)
        @inferred evaluate(smpl, Gℓ)
        @inferred evaluate(smpl, Gℓ, dim=1)
        Gτ = evaluate(smpl, Gℓ)

        noise = 1e-5
        Gτ_n = Gτ + noise * norm(Gτ) * randn(size(Gτ)...)
        @inferred fit(smpl, Gτ_n)
        @inferred fit(smpl, Gτ_n, dim=1)
        Gℓ_n = fit(smpl, Gτ_n)

        @test Gℓ ≈ Gℓ_n atol = 12 * noise * Gℓ_magn rtol = 0
    end

    @testset "wn noise with stat = $stat" for stat in (boson, fermion)
        Λ = 42
        basis = DimensionlessBasis(stat, Λ; sve_result=sve_logistic[Λ])
        smpl = MatsubaraSampling(basis)
        Random.seed!(1312)

        ρl = basis.v([-0.999, -0.01, 0.5]) * [0.8, -0.2, 0.5]
        Gl = basis.s .* ρl
        Gl_magn = norm(Gl)
        @inferred evaluate(smpl, Gl)
        @inferred evaluate(smpl, Gl, dim=1)
        Giw = evaluate(smpl, Gl)

        noise = 1e-5
        Giwn_n = Giw + noise * norm(Giw) * randn(size(Giw)...)
        @inferred fit(smpl, Giwn_n)
        @inferred fit(smpl, Giwn_n, dim=1)
        Gl_n = fit(smpl, Giwn_n)

        @test isapprox(Gl, Gl_n, atol=12 * noise * Gl_magn, rtol=0) broken = true
    end
end
