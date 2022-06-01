using Test
using SparseIR

using Random, LinearAlgebra

include("__conftest.jl")

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
        @test A * x ≈ Ad.U * Diagonal(Ad.S) * Ad.Vt * x atol = 2e-14 * norm_A rtol = 0

        y = randn(49)
        @test A \ y ≈ Ad \ y atol = 1e-14 * norm_A rtol = 0

        y = randn(49, 2)
        @test A \ y ≈ Ad \ y atol = 1e-14 * norm_A rtol = 0
    end

    @testset "fit from tau with stat = $stat, Λ = $Λ" for stat in (boson, fermion), Λ in (10, 42)
        basis = DimensionlessBasis(stat, Λ; sve_result=sve_logistic[Λ])
        smpl = TauSampling(basis)
        @test issorted(smpl.sampling_points)
        Random.seed!(5318008)

        shape = (2, 3, 4)
        rhol = randn(ComplexF64, (length(basis), shape...))
        originalgl = -basis.s .* rhol
        for dim in 1:ndims(rhol)
            gl = SparseIR.movedim(originalgl, 1 => dim)
            gtau = evaluate(smpl, gl; dim)
            @test size(gtau) == (
                size(gl)[1:(dim - 1)]...,
                length(smpl.sampling_points),
                size(gl)[(dim + 1):end]...,
            )

            gl_from_tau = fit(smpl, gtau; dim)
            @test gl_from_tau ≈ gl

            gl_from_tau2 = similar(gl_from_tau)
            fit!(gl_from_tau2, smpl, gtau; dim)
            @test gl_from_tau2 ≈ gl
        end
    end

    @testset "τ noise with stat = $stat, Λ = $Λ" for stat in (boson, fermion), Λ in (10, 42)
        basis = DimensionlessBasis(stat, Λ; sve_result=sve_logistic[Λ])
        smpl = TauSampling(basis)
        @test issorted(smpl.sampling_points)
        Random.seed!(5318008)

        ρℓ = basis.v([-0.999, -0.01, 0.5]) * [0.8, -0.2, 0.5]
        Gℓ = basis.s .* ρℓ
        Gℓ_magn = norm(Gℓ)
        @inferred evaluate(smpl, Gℓ)
        @inferred evaluate(smpl, Gℓ, dim=1)
        Gτ = evaluate(smpl, Gℓ)

        Gτ_inplace = similar(Gτ)
        evaluate!(Gτ_inplace, smpl, Gℓ)
        @test Gτ == Gτ_inplace

        noise = 1e-5
        Gτ_n = Gτ + noise * norm(Gτ) * randn(size(Gτ)...)
        @inferred fit(smpl, Gτ_n)
        @inferred fit(smpl, Gτ_n, dim=1)
        Gℓ_n = fit(smpl, Gτ_n)

        Gℓ_n_inplace = similar(Gℓ_n)
        fit!(Gℓ_n_inplace, smpl, Gτ_n)
        @test Gℓ_n == Gℓ_n_inplace

        @test isapprox(Gℓ, Gℓ_n, atol=12 * noise * Gℓ_magn, rtol=0)
    end

    @testset "iω noise with stat = $stat, Λ = $Λ" for stat in (boson, fermion),
        Λ in (10, 42)

        basis = DimensionlessBasis(stat, Λ; sve_result=sve_logistic[Λ])
        smpl = MatsubaraSampling(basis)
        @test issorted(smpl.sampling_points)
        Random.seed!(1312 + 161)

        ρℓ = basis.v([-0.999, -0.01, 0.5]) * [0.8, -0.2, 0.5]
        Gℓ = basis.s .* ρℓ
        Gℓ_magn = norm(Gℓ)
        @inferred evaluate(smpl, Gℓ)
        @inferred evaluate(smpl, Gℓ, dim=1)
        Giw = evaluate(smpl, Gℓ)

        Giw_inplace = similar(Giw)
        evaluate!(Giw_inplace, smpl, Gℓ)
        @test Giw == Giw_inplace

        noise = 1e-5
        Giwn_n = Giw + noise * norm(Giw) * randn(size(Giw)...)
        @inferred fit(smpl, Giwn_n)
        @inferred fit(smpl, Giwn_n, dim=1)
        Gℓ_n = fit(smpl, Giwn_n)

        Gℓ_n_inplace = similar(Gℓ_n)
        fit!(Gℓ_n_inplace, smpl, Giwn_n)
        @test Gℓ_n == Gℓ_n_inplace

        @test isapprox(Gℓ, Gℓ_n, atol=40 * noise * Gℓ_magn, rtol=0)
    end
end
