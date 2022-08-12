using Test
using SparseIR
using LinearAlgebra

include("_conftest.jl")

function check_composite_poly(u_comp, u_list, test_points)
    @test length(u_comp) == sum(length, u_list)
    @test size(u_comp) == (length(u_comp), )
    @test u_comp(test_points) ≈ mapreduce(u -> u(test_points), vcat, u_list)
    idx = 1
    for isub in eachindex(u_list)
        for ip in 1:length(u_list[isub])
            @test u_comp[idx](test_points) ≈ u_list[isub][ip](test_points)
            idx += 1
        end
    end
end

@testset "composite.jl" begin
    @testset "Composite Poly" begin
        u, s, v = sve_logistic[42]
        l = length(s)

        u_comp = CompositeBasisFunction([u, u])
        check_composite_poly(u_comp, [u, u], range(-1, 1; length=10))

        uhat = SparseIR.PiecewiseLegendreFTVector(u, fermion)
        uhat_comp = CompositeBasisFunction([uhat, uhat])
        check_composite_poly(uhat_comp, [uhat, uhat], [-3, 1, 5])
    end

    @testset "Composite Basis" begin
        β = 7
        ωmax = 6
        basis = FiniteTempBasis(fermion, β, ωmax, 1e-6; sve_result=sve_logistic[β * ωmax])
        basis2 = FiniteTempBasis(fermion, β, ωmax, 1e-3; sve_result=sve_logistic[β * ωmax])
        basis_comp = CompositeBasis([basis, basis2])
        check_composite_poly(basis_comp.u, [basis.u, basis2.u], range(0, β; length=10))
        check_composite_poly(basis_comp.uhat, [basis.uhat, basis2.uhat], [1, 3])
        @test SparseIR.β(basis_comp) == β
        @test SparseIR.statistics(basis_comp) == SparseIR.statistics(basis)
    end

    @testset "Augmented basis with stat = $stat" for stat in (fermion, boson)
        ωmax = 2
        β = 1000

        basis = FiniteTempBasis(stat, β, ωmax, 1e-6)
        basis_legg = LegendreBasis(stat, β, 2)
        basis_comp = CompositeBasis([basis_legg, basis])

        @test basis_comp.u(1.0) ≈ vcat(basis_legg.u(1.0), basis.u(1.0))

        # v = 0.1:0.1:1.0
        # @test basis_comp.u(v) ≈ vcat(basis_legg.u(v), basis.u(v))

        # n = MatsubaraFreq(6 + SparseIR.zeta(stat))
        # res = vcat(basis_legg.uhat(n), basis.uhat(n))
        # @test basis_comp.uhat(n) ≈ res
        # @test basis_comp.uhat(Integer(n)) ≈ res

        # nn = n : n+4*pioverbeta
        # res = vcat(basis_legg.uhat(nn), basis.uhat(nn))
        # @test basis_comp.uhat(nn) ≈ res
        # @test basis_comp.uhat(Integer.(nn)) ≈ res

        # G(τ) = c - e^{-τ*pole}/(1 - e^{-β*pole})
        pole = 1.0
        c = 1e-2
        τ_smpl = TauSampling(basis_comp)
        gτ = c .+ transpose(basis.u(SparseIR.sampling_points(τ_smpl))) * (-basis.s .* basis.v(pole))
        magn = maximum(abs, gτ)
        # gl_from_τ = fit(τ_smpl, gτ)

        # This illustrates that "naive" fitting is a problem if the fitting matrix
        # is not well-conditioned.
        gl_fit_bad = pinv(τ_smpl.matrix) * gτ
        gτ_reconst_bad = evaluate(τ_smpl, gl_fit_bad)
        @test !isapprox(gτ_reconst_bad, gτ; atol=1e-13 * magn, rtol=0)
        @test isapprox(gτ_reconst_bad, gτ; atol=5e-16 * cond(τ_smpl) * magn, rtol=0)

        # Now do the fit properly
        gl_fit = fit(τ_smpl, gτ)
        gτ_reconst = evaluate(τ_smpl, gl_fit)
        @test isapprox(gτ_reconst, gτ; atol=1e-14 * magn, rtol=0)

        # gτ_reconst = evaluate(τ_smpl, gl_from_τ)
        # @test isapprox(gτ, gτ_reconst; atol=1e-14 * maximum(abs, gτ), rtol=0)

        # sgn = SparseIR.significance(basis_comp)
        # @test issorted(sgn; rev=true)
        # @test all(<=(1), sgn)
        # @test all(>=(ϵ), sgn)
    end

    @testset "Vertex Basis with stat = $stat" for stat in (fermion, boson)
        ωmax = 2
        β = 1000
        basis = FiniteTempBasis(stat, β, ωmax, 1e-6)
        basis_const = MatsubaraConstBasis(stat, β)
        basis_comp = CompositeBasis([basis_const, basis])
        @test !isnothing(basis_comp.uhat)

        # G(iv) = c + 1/(iv-pole)
        pole = 1.0
        c = 1.0
        matsu_smpl = MatsubaraSampling(basis_comp)
        giv = c .+ 1.0 ./ (SparseIR.valueim.(SparseIR.sampling_points(matsu_smpl), β) .- pole)
        gl = fit(matsu_smpl, giv)

        giv_reconst = evaluate(matsu_smpl, gl)

        @test isapprox(giv, giv_reconst; atol=maximum(abs, giv) * 1e-7, rtol=0)
    end
end
