using Test
using SparseIR
using AssociatedLegendrePolynomials: Plm
using LinearAlgebra

@testset "augment.jl" begin
    @testset "Augmented bosonic basis" begin
        ωmax = 2
        β = 1000
        basis = FiniteTempBasis(boson, β, ωmax, 1e-6)
        basis_comp = AugmentedBasis(basis, TauConst, TauLinear)

        # G(τ) = c - e^{-τ * pole} / (1 - e^{-β * pole})
        pole = 1.0
        c = 1e-2
        τ_smpl = TauSampling(basis_comp)
        @test length(τ_smpl.τ) == length(basis_comp)
        gτ = c .+ transpose(basis.u(τ_smpl.τ)) * (-basis.s .* basis.v(pole))
        magn = maximum(abs, gτ)
        # @show magn

        # This illustrates that "naive" fitting is a problem if the fitting matrix
        # is not well-conditioned.
        gl_fit_bad = pinv(τ_smpl.matrix) * gτ
        gτ_reconst_bad = evaluate(τ_smpl, gl_fit_bad)
        @test !isapprox(gτ_reconst_bad, gτ, atol=1e-13 * magn)
        # @show norm(gτ_reconst_bad - gτ) norm(gτ) 1e-13 * magn 5e-16 * cond(τ_smpl) * magn cond(τ_smpl)
        @test isapprox(gτ_reconst_bad, gτ, atol=5e-16 * cond(τ_smpl) * magn)

        # Now do the fit properly
        gl_fit = fit(τ_smpl, gτ)
        gτ_reconst = evaluate(τ_smpl, gl_fit)
        @test isapprox(gτ_reconst, gτ, atol=1e-14 * magn)
    end

    @testset "Vertex basis with stat = $stat" for stat in (fermion, boson)
        ωmax = 2
        β = 1000
        basis = FiniteTempBasis(stat, β, ωmax, 1e-6)
        basis_comp = AugmentedBasis(basis, MatsubaraConst)
        @test !isnothing(basis_comp.uhat)

        # G(iν) = c + 1 / (iν - pole)
        pole = 1.0
        c = 1.0
        matsu_smpl = MatsubaraSampling(basis_comp)
        giν = @. c + 1 / (SparseIR.valueim(matsu_smpl.ωn, β) - pole)
        gl = fit(matsu_smpl, giν)

        giν_reconst = evaluate(matsu_smpl, gl)

        @test isapprox(giν_reconst, giν, atol=maximum(abs, giν) * 1e-7)
    end
end
