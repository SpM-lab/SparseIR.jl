
@testitem "augment.jl" tags=[:julia, :sparseir, :debug] begin
    using Test
    using SparseIR
    import SparseIR as SparseIR
    using LinearAlgebra
    using StableRNGs

    @testset "Augmented bosonic basis" begin
        ωmax = 2
        β = 1000
        basis = FiniteTempBasis{Bosonic}(β, ωmax, 1e-6)
        @show size(basis.u)
        size_sve = Ref{Int32}(0)
        SparseIR.spir_sve_result_get_size(basis.sve_result.ptr, size_sve)
        @show size_sve[]
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
end
