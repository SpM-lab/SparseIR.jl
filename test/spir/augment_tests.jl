
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

        @testset "normalize_tau" begin
            β = 10.0
            
            # Test Bosonic statistics
            @testset "Bosonic" begin
                # Normal range [0, β]
                @test SparseIR.normalize_tau(Bosonic, 5.0, β) == (5.0, 1.0)
                @test SparseIR.normalize_tau(Bosonic, 0.0, β) == (0.0, 1.0)
                @test SparseIR.normalize_tau(Bosonic, β, β) == (β, 1.0)
                
                # Negative range [-β, 0)
                @test SparseIR.normalize_tau(Bosonic, -3.0, β) == (7.0, 1.0)
                @test SparseIR.normalize_tau(Bosonic, -β, β) == (0.0, 1.0)
                
                # Negative zero
                @test SparseIR.normalize_tau(Bosonic, -0.0, β) == (β, 1.0)
                
                # Out of range
                @test_throws DomainError SparseIR.normalize_tau(Bosonic, -β - 0.1, β)
                @test_throws DomainError SparseIR.normalize_tau(Bosonic, β + 0.1, β)
            end
            
            # Test Fermionic statistics
            @testset "Fermionic" begin
                # Normal range [0, β]
                @test SparseIR.normalize_tau(Fermionic, 5.0, β) == (5.0, 1.0)
                @test SparseIR.normalize_tau(Fermionic, 0.0, β) == (0.0, 1.0)
                @test SparseIR.normalize_tau(Fermionic, β, β) == (β, 1.0)
                
                # Negative range [-β, 0) - anti-periodic
                @test SparseIR.normalize_tau(Fermionic, -3.0, β) == (7.0, -1.0)
                @test SparseIR.normalize_tau(Fermionic, -β, β) == (0.0, -1.0)
                
                # Negative zero - anti-periodic
                @test SparseIR.normalize_tau(Fermionic, -0.0, β) == (β, -1.0)
                
                # Out of range
                @test_throws DomainError SparseIR.normalize_tau(Fermionic, -β - 0.1, β)
                @test_throws DomainError SparseIR.normalize_tau(Fermionic, β + 0.1, β)
            end
        end

        @testset "TauConst" begin
            β = 123.0
            @test_throws DomainError TauConst(-34)
            
            # Backward compatibility (defaults to Bosonic)
            tc = TauConst(β)
            @test SparseIR.β(tc) == β
            @test tc(50) == 1 / sqrt(β)
            @test tc(BosonicFreq(0)) == sqrt(β)
            @test tc(BosonicFreq(92)) == 0.0
            @test SparseIR.deriv(tc)(4.2) == 0.0
            @test SparseIR.deriv(tc, Val(0)) == tc
            
            # Test periodicity for Bosonic
            @testset "Bosonic periodicity" begin
                tc_b = TauConst{Bosonic}(β)
                @test tc_b(β/2) == 1 / sqrt(β)
                @test tc_b(-β/2) == 1 / sqrt(β)  # Periodic
                @test tc_b(0.0) == 1 / sqrt(β)
                @test tc_b(-0.0) == 1 / sqrt(β)  # Negative zero, periodic
            end
            
            # Test anti-periodicity for Fermionic
            @testset "Fermionic anti-periodicity" begin
                tc_f = TauConst{Fermionic}(β)
                @test tc_f(β/2) == 1 / sqrt(β)
                @test tc_f(-β/2) == -1 / sqrt(β)  # Anti-periodic
                @test tc_f(0.0) == 1 / sqrt(β)
                @test tc_f(-0.0) == -1 / sqrt(β)  # Negative zero, anti-periodic
            end
        end

        @testset "TauLinear" begin
            β = 123.0
            @test_throws DomainError TauLinear(-34)
            
            # Backward compatibility (defaults to Bosonic)
            tl = TauLinear(β)
            @test SparseIR.β(tl) == β
            @test tl(50) ≈ sqrt(3 / β) * (2 / β * 50 - 1)
            @test tl(BosonicFreq(0)) == 0.0
            @test tl(BosonicFreq(92)) ≈ sqrt(3 / β) * 2 / im * β / (92 * π)
            @test SparseIR.deriv(tl, Val(0)) == tl
            @test SparseIR.deriv(tl)(4.2) ≈ sqrt(3 / β) * 2 / β
            @test SparseIR.deriv(tl, Val(2))(4.2) == 0.0
            
            # Test periodicity for Bosonic
            @testset "Bosonic periodicity" begin
                tl_b = TauLinear{Bosonic}(β)
                val_pos = tl_b(β/4)
                val_neg = tl_b(-β/4)
                # Periodic: tl(τ + β) = tl(τ), so tl(-β/4) wraps to tl(3β/4)
                val_wrapped = tl_b(3*β/4)
                @test val_neg ≈ val_wrapped
            end
            
            # Test anti-periodicity for Fermionic
            @testset "Fermionic anti-periodicity" begin
                tl_f = TauLinear{Fermionic}(β)
                val_pos = tl_f(β/4)
                val_neg = tl_f(-β/4)
                # Anti-periodic: tl(τ + β) = -tl(τ), so tl(-β/4) wraps to -tl(3β/4)
                val_wrapped = tl_f(3*β/4)
                @test val_neg ≈ -val_wrapped
            end
        end

        @testset "MatsubaraConst" begin
            β = 123.0
            @test_throws DomainError MatsubaraConst(-34)
            mc = MatsubaraConst(β)
            @test SparseIR.β(mc) == β
            @test_throws DomainError mc(-β - 1)
            @test_throws DomainError mc(β + 1)
            @test isnan(mc(30))
            @test isnan(mc(-30))  # Now supports negative tau
            @test mc(FermionicFreq(1)) == 1.0
            @test mc(BosonicFreq(0)) == 1.0
            @test mc(BosonicFreq(92)) == 1.0
            @test SparseIR.deriv(mc) == mc
            @test SparseIR.deriv(mc, Val(0)) == mc
        end
    end
end
