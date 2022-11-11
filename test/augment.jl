using Test
using SparseIR
using SparseIR.LinearAlgebra

@testset "augment.jl" begin
    @testset "Augmented bosonic basis" begin
        ωmax = 2
        β = 1000
        basis = FiniteTempBasis{Bosonic}(β, ωmax, 1e-6)
        basis_aug = AugmentedBasis(basis, TauConst, TauLinear)

        @test all(isone, SparseIR.significance(basis_aug)[1:3])

        # G(τ) = c - e^{-τ * pole} / (1 - e^{-β * pole})
        pole = 1.0
        c = 1e-2
        τ_smpl = TauSampling(basis_aug)
        @test length(τ_smpl.τ) == length(basis_aug)
        gτ = c .+ transpose(basis.u(τ_smpl.τ)) * (-basis.s .* basis.v(pole))
        magn = maximum(abs, gτ)

        # This illustrates that "naive" fitting is a problem if the fitting matrix
        # is not well-conditioned.
        gl_fit_bad = pinv(τ_smpl.matrix) * gτ
        gτ_reconst_bad = evaluate(τ_smpl, gl_fit_bad)
        @test !isapprox(gτ_reconst_bad, gτ, atol=1e-13 * magn)
        @test isapprox(gτ_reconst_bad, gτ, atol=5e-16 * cond(τ_smpl) * magn)
        @test cond(τ_smpl) > 1e7
        @test size(τ_smpl.matrix) == (length(basis_aug), length(τ_smpl.τ))

        # Now do the fit properly
        gl_fit = fit(τ_smpl, gτ)
        gτ_reconst = evaluate(τ_smpl, gl_fit)

        @test isapprox(gτ_reconst, gτ, atol=1e-14 * magn)
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

            @test length(basis_aug[1:5]) == 5
            @test_throws ErrorException basis_aug[1:2]
            @test_throws ErrorException basis_aug[3:7]
        end

        len_basis = length(basis)
        len_aug = len_basis + 2

        @test size(basis_aug) == (len_aug,)
        @test SparseIR.accuracy(basis_aug) == SparseIR.accuracy(basis)
        @test SparseIR.Λ(basis_aug) == β * ωmax
        @test SparseIR.ωmax(basis_aug) == ωmax

        @test size(basis_aug.u) == (len_aug,)
        @test length(basis_aug.u(0.8)) == len_aug
        @test length(basis_aug.uhat(MatsubaraFreq(4))) == len_aug
        @test SparseIR.xmin(basis_aug.u) == 0.0
        @test SparseIR.xmax(basis_aug.u) == β

        @test SparseIR.deriv(basis_aug.u)(0.8)[3:end] == SparseIR.deriv.(SparseIR.fbasis(basis_aug.u))(0.8)

        @test SparseIR.zeta(basis_aug.uhat) == 0

        @testset "create" begin
            @test SparseIR.create(MatsubaraConst(42), basis) == MatsubaraConst(42)
            @test SparseIR.create(MatsubaraConst, basis) == MatsubaraConst(β)
        end

        @testset "TauConst" begin
            @test_throws DomainError TauConst(-34)
            tc = TauConst(123)
            @test SparseIR.β(tc) == 123.0
            @test_throws DomainError tc(-3)
            @test_throws DomainError tc(321)
            @test tc(100) == 1/sqrt(123)
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
            @test_throws DomainError tl(-3)
            @test_throws DomainError tl(321)
            @test tl(100) ≈ sqrt(3 / 123) * (2 / 123 * 100 - 1)
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
            @test_throws DomainError mc(-3)
            @test_throws DomainError mc(321)
            @test isnan(mc(100))
            @test mc(MatsubaraFreq(0)) == 1.0
            @test mc(MatsubaraFreq(92)) == 1.0
            @test mc(MatsubaraFreq(93)) == 1.0
            @test SparseIR.deriv(mc) == mc
            @test SparseIR.deriv(mc, Val(0)) == mc
        end
    end
end
