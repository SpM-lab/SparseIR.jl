
@testitem "augment.jl" tags=[:julia, :sparseir, :debug] begin
    using Test
    using SparseIR
    using LinearAlgebra
    using StableRNGs

    @testset "Augmented bosonic basis" begin
        ωmax = 2
        β = 1000
        basis = FiniteTempBasis{Bosonic}(β, ωmax, 1e-6)
        basis_aug = AugmentedBasis(basis, TauConst, TauLinear)

        @test all(isone, SparseIR.significance(basis_aug)[1:3])
        rng = StableRNG(42)

        gτ = rand(rng, length(basis_aug))
        τ_smpl = TauSampling(basis_aug)
        gl_fit = fit(τ_smpl, gτ)
        gτ_reconst = evaluate(τ_smpl, gl_fit)

        @test size(gτ_reconst) == size(gτ)

        @test isapprox(gτ_reconst, gτ, atol=1e-8)
    end

    @testset "Vertex basis with stat = $stat" for stat in (Fermionic(), Bosonic())
        ωmax = 2
        β = 1000
        basis = FiniteTempBasis(stat, β, ωmax, 1e-6)
        basis_aug = AugmentedBasis(basis, MatsubaraConst)
        @test !isnothing(basis_aug.uhat)

        # G(iν) = c + 1 / (iν - pole)
        pole = 1.0
        c = 1.0
        matsu_smpl = MatsubaraSampling(basis_aug)
        giν = @. c + 1 / (SparseIR.valueim(matsu_smpl.ωn, β) - pole)
        gl = fit(matsu_smpl, giν)

        giν_reconst = evaluate(matsu_smpl, gl)

        @test isapprox(giν_reconst, giν, atol=maximum(abs, giν) * 1e-7)
    end

    @testset "unit tests" begin
        β = 1000
        ωmax = 2
        basis = FiniteTempBasis{Bosonic}(β, ωmax, 1e-6)
        basis_aug = AugmentedBasis(basis, TauConst, TauLinear)

        @testset "getindex" begin
            @test length(basis_aug.u[1:5]) == 5
            @test_throws ErrorException basis_aug.u[1:2]
            @test_throws ErrorException basis_aug.u[3:7]
            @test basis_aug.u[1] isa TauConst
            @test basis_aug.u[2] isa TauLinear
        end

        len_basis = length(basis)
        len_aug = len_basis + 2

        @test size(basis_aug) == (len_aug,)
        @test SparseIR.accuracy(basis_aug) == SparseIR.accuracy(basis)
        @test SparseIR.Λ(basis_aug) == β * ωmax
        @test SparseIR.ωmax(basis_aug) == ωmax

        @test size(basis_aug.u) == (len_aug,)
        @test length(basis_aug.u(0.8)) == len_aug

        @testset "create" begin
            @test SparseIR.create(MatsubaraConst(42), basis) == MatsubaraConst(42)
            @test SparseIR.create(MatsubaraConst, basis) == MatsubaraConst(β)
        end

        @testset "TauConst" begin
            @test_throws DomainError TauConst(-34)
            tc = TauConst(123)
            @test SparseIR.β(tc) == 123.0
            @test_throws DomainError tc(-123)
            @test_throws DomainError tc(321)
            @test tc(50) == 1 / sqrt(123)
            @test tc(MatsubaraFreq(0)) == sqrt(123)
            @test tc(MatsubaraFreq(92)) == 0.0
            @test_throws ErrorException tc(MatsubaraFreq(93))
            @test SparseIR.deriv(tc)(4.2) == 0.0
            @test SparseIR.deriv(tc, Val(0)) == tc
        end

        @testset "TauLinear" begin
            @test_throws DomainError TauLinear(-34)
            tl = TauLinear(123)
            @test SparseIR.β(tl) == 123.0
            @test_throws DomainError tl(-1003)
            @test_throws DomainError tl(1003)
            @test tl(50) ≈ sqrt(3 / 123) * (2 / 123 * 50 - 1)
            @test tl(MatsubaraFreq(0)) == 0.0
            @test tl(MatsubaraFreq(92)) ≈ sqrt(3 / 123) * 2 / im * 123 / (92 * π)
            @test_throws ErrorException tl(MatsubaraFreq(93))
            @test SparseIR.deriv(tl, Val(0)) == tl
            @test SparseIR.deriv(tl)(4.2) ≈ sqrt(3 / 123) * 2 / 123
            @test SparseIR.deriv(tl, Val(2))(4.2) == 0.0
        end

        @testset "MatsubaraConst" begin
            @test_throws DomainError MatsubaraConst(-34)
            mc = MatsubaraConst(123)
            @test SparseIR.β(mc) == 123.0
            @test_throws DomainError mc(-123)
            @test_throws DomainError mc(321)
            @test isnan(mc(30))
            @test mc(MatsubaraFreq(0)) == 1.0
            @test mc(MatsubaraFreq(92)) == 1.0
            @test mc(MatsubaraFreq(93)) == 1.0
            @test SparseIR.deriv(mc) == mc
            @test SparseIR.deriv(mc, Val(0)) == mc
        end
    end
end
