using Test
using SparseIR
using Logging: with_logger, NullLogger

isdefined(Main, :sve_logistic) || include("_conftest.jl")

@testset "basis.jl" begin
    @testset "consistency" begin
        β = 2.0
        ωmax = 5.0
        ε = 1e-5
        basis_f, basis_b = SparseIR.finite_temp_bases(β, ωmax, ε)
        bs = FiniteTempBasisSet(β, ωmax, ε)

        @test length(bs.basis_f) == length(basis_f)
        @test length(bs.basis_b) == length(basis_b)
    end

    @testset "consistency2" begin
        β = 2.0
        ωmax = 5.0
        ε = 1e-5

        sve_result = sve_logistic[β * ωmax]
        basis_f, basis_b = SparseIR.finite_temp_bases(β, ωmax, ε, sve_result)
        smpl_τ_f = TauSampling(basis_f)
        smpl_τ_b = TauSampling(basis_b)
        smpl_wn_f = MatsubaraSampling(basis_f)
        smpl_wn_b = MatsubaraSampling(basis_b)

        bs = FiniteTempBasisSet(β, ωmax, ε; sve_result)
        @test smpl_τ_f.sampling_points == smpl_τ_b.sampling_points
        @test bs.smpl_tau_f.matrix == smpl_τ_f.matrix
        @test bs.smpl_tau_b.matrix == smpl_τ_b.matrix

        @test bs.smpl_wn_f.matrix == smpl_wn_f.matrix
        @test bs.smpl_wn_b.matrix == smpl_wn_b.matrix
    end

    @testset "FiniteTempBasis" begin
        with_logger(NullLogger()) do
            basis = FiniteTempBasis{Fermionic}(1e-3, 1e-3, 1e-100)
            @test SparseIR.sve_result(basis).s * sqrt(1e-3 / 2 * 1e-3) ≈ basis.s
            @test SparseIR.accuracy(basis) ≈ last(basis.s) / first(basis.s)
        end
        basis = FiniteTempBasis{Fermionic}(3, 4, 1e-6)
        io = IOBuffer()
        show(io, basis)
        s = String(take!(io))
        @test occursin(r"FiniteTempBasis{Fermionic} with β = 3\.0, ωmax = 4\.0", s)

        @test SparseIR.ωmax(SparseIR.rescale(basis, 2)) ≈ 6

        sve = SparseIR.sve_result(basis)

        @test_logs (:warn, r"""
        Expecting to get 100 sampling points for corresponding basis function,
        instead got \d+\. This may happen if not enough precision is
        left in the polynomial\.
        """) SparseIR.default_sampling_points(sve.u, 100)


        basis = FiniteTempBasis{Bosonic}(3, 4, 1e-6)

        @test_logs (:warn, r"""
        Requesting 13 Bosonic\(\) sampling frequencies for basis size
        L = \d+, but \d+ were returned\. This may indicate a problem with precision\.
        """) SparseIR.default_matsubara_sampling_points(basis.uhat, 12; fence=true)
    end

    @testset "unit tests" begin
        β = 23
        ωmax = 3e-2
        ε = 1e-5

        bset = FiniteTempBasisSet(β, ωmax, ε)

        basis_f = FiniteTempBasis{Fermionic}(β, ωmax, ε)
        basis_b = FiniteTempBasis{Bosonic}(β, ωmax, ε)
        @test SparseIR.β(bset) == β
        @test SparseIR.ωmax(bset) == ωmax
        @test bset.tau == SparseIR.sampling_points(TauSampling(basis_f))
        @test bset.wn_f == SparseIR.sampling_points(MatsubaraSampling(basis_f))
        @test bset.wn_b == SparseIR.sampling_points(MatsubaraSampling(basis_b))
        @test bset.sve_result.s ≈ SparseIR.SVEResult(LogisticKernel(β * ωmax); ε).s
        @test :tau ∈ propertynames(bset)
        io = IOBuffer()
        show(io, bset)
        @test String(take!(io)) == "FiniteTempBasisSet with β = $β.0, ωmax = $ωmax"
    end
end
