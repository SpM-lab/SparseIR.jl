# Smoke test of the public surface (spm-agent-rules testing.md). `SMOKE` uses every
# exported name once with minimal valid arguments and checks a meaningful property
# of the result; it is compared with `names(SparseIR)` in both directions, so a new
# export without an entry and a stale entry both fail. The second table does the
# same for the function sets returned by the bases. The Python suite has the same
# two tables (tests/test_public_surface.py in SpM-lab/sparse-ir).

@testitem "public surface: every export is callable" tags=[:julia, :surface] setup=[SIRTestSetup] begin
    using Test
    using SparseIR
    using LinearAlgebra: norm

    β, ωmax, ε = 10.0, 1.0, 1e-6
    bf = get_basis(Fermionic(), β, ωmax, ε)
    bb = get_basis(Bosonic(), β, ωmax, ε)
    gl = collect(range(1.0, 0.1; length=length(bf)))

    roundtrip(smpl) = maximum(abs, fit(smpl, evaluate(smpl, gl)) .- gl) <= 1e-10
    SMOKE = Dict{Symbol,Function}(
        :Fermionic => () -> SparseIR.zeta(Fermionic()) == 1,
        :Bosonic => () -> SparseIR.zeta(Bosonic()) == 0,
        :MatsubaraFreq => () -> MatsubaraFreq(3) === FermionicFreq(3),
        :FermionicFreq => () -> SparseIR.value(FermionicFreq(3), β) ≈ 3π / β,
        :BosonicFreq => () -> SparseIR.valueim(BosonicFreq(2), β) ≈ 2im * π / β,
        :pioverbeta => () -> Int(3 * pioverbeta) == 3,
        :FiniteTempBasis => () -> FiniteTempBasis(Fermionic(), β, ωmax, ε).s ≈ bf.s,
        :FiniteTempBasisSet => () -> FiniteTempBasisSet(β, ωmax, ε).tau ==
                                     sampling_points(TauSampling(bf)),
        :DiscreteLehmannRepresentation => () -> begin
            dlr = DiscreteLehmannRepresentation(bf)
            g = -bf.s .* bf.v(0.3)
            norm(to_IR(dlr, from_IR(dlr, g)) - g) <= 300ε * norm(g)
        end,
        :overlap => () -> isapprox(overlap(bf.u[1], bf.u[1]), 1; atol=1e-12),
        :LogisticKernel => () -> SparseIR.Λ(LogisticKernel(42.0)) == 42.0,
        :RegularizedBoseKernel => () -> SparseIR.Λ(RegularizedBoseKernel(42.0)) == 42.0,
        :iscentrosymmetric => () -> iscentrosymmetric(LogisticKernel(1.0)),
        :AugmentedBasis => () -> length(AugmentedBasis(bb, TauConst, TauLinear)) ==
                                 length(bb) + 2,
        :TauConst => () -> TauConst(β)(3.0) ≈ 1 / sqrt(β) && TauConst(β)(BosonicFreq(0)) ≈ sqrt(β),
        :TauLinear => () -> TauLinear(β)(0.0) ≈ -sqrt(3 / β),
        :MatsubaraConst => () -> isnan(MatsubaraConst(β)(1.0)) &&
                                 MatsubaraConst(β)(BosonicFreq(4)) == 1,
        :TauSampling => () -> roundtrip(TauSampling(bf)),
        :MatsubaraSampling => () -> roundtrip(MatsubaraSampling(bf)),
        :evaluate => () -> evaluate(TauSampling(bf), gl) ≈ transpose(bf.u(sampling_points(TauSampling(bf)))) * gl,
        :fit => () -> roundtrip(TauSampling(bf)),
        :evaluate! => () -> begin
            smpl = TauSampling(bf)
            out = zeros(npoints(smpl))
            evaluate!(out, smpl, gl) === out && out ≈ evaluate(smpl, gl)
        end,
        :fit! => () -> begin
            smpl = TauSampling(bf)
            out = zeros(length(bf))
            fit!(out, smpl, evaluate(smpl, gl)) === out && maximum(abs, out - gl) <= 1e-10
        end,
        :sampling_points => () -> issorted(sampling_points(TauSampling(bf))),
        :npoints => () -> npoints(TauSampling(bf)) == length(sampling_points(TauSampling(bf))),
        :from_IR => () -> length(from_IR(DiscreteLehmannRepresentation(bf), gl)) ==
                          npoles(DiscreteLehmannRepresentation(bf)),
        :to_IR => () -> length(to_IR(DiscreteLehmannRepresentation(bf),
            ones(npoles(DiscreteLehmannRepresentation(bf))))) == length(bf),
        :npoles => () -> npoles(DiscreteLehmannRepresentation(bf)) ==
                         length(default_omega_sampling_points(bf)),
        :get_poles => () -> get_poles(DiscreteLehmannRepresentation(bf, [-0.5, 0.5])) == [-0.5, 0.5],
        :default_omega_sampling_points => () -> all(abs.(default_omega_sampling_points(bf)) .<= ωmax))

    exported = Set(n for n in names(SparseIR) if n !== :SparseIR)
    @test Set(keys(SMOKE)) == exported
    @testset "$name" for name in sort(collect(exported))
        @test isdefined(SparseIR, name)
        @test SMOKE[name]()
    end
end

@testitem "public surface: function sets of the bases" tags=[:julia, :surface] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    β, ωmax, ε = 10.0, 1.0, 1e-6
    bf = get_basis(Fermionic(), β, ωmax, ε)
    bb = get_basis(Bosonic(), β, ωmax, ε)
    aug = AugmentedBasis(bb, TauConst, TauLinear)
    vertex = AugmentedBasis(bf, MatsubaraConst)
    dlr = DiscreteLehmannRepresentation(bf)

    # (label, function set, valid point, provides deriv, kind)
    sets = [("FiniteTempBasis.u", bf.u, 0.5, true, :tau),
        ("FiniteTempBasis.v", bf.v, 0.5, true, :omega),
        ("FiniteTempBasis.uhat", bf.uhat, FermionicFreq(3), false, :matsubara),
        ("AugmentedBasis.u", aug.u, 0.5, true, :tau),
        ("AugmentedBasis.uhat", aug.uhat, BosonicFreq(2), false, :matsubara),
        ("vertex AugmentedBasis.uhat", vertex.uhat, FermionicFreq(3), false, :matsubara),
        ("DiscreteLehmannRepresentation.u", dlr.u, 0.5, false, :tau),
        ("DiscreteLehmannRepresentation.uhat", dlr.uhat, FermionicFreq(3), false, :matsubara)]

    @testset "$label" for (label, fs, x, has_deriv, kind) in sets
        values = fs(x)
        n = length(fs)
        @test length(values) == n
        @test sum(abs2, values) > 0
        @test fs[1](x) ≈ values[1]
        @test fs[end](x) ≈ values[end]
        @test fs[n](x) ≈ values[n]
        @test fs[1:n](x) ≈ values
        if kind === :matsubara
            @test SparseIR.zeta(fs) == SparseIR.zeta(x)
            @test fs(Int(x)) == values           # integer reduced frequencies
        else
            bound = kind === :tau ? β : ωmax
            @test (SparseIR.xmin(fs), SparseIR.xmax(fs)) == (-bound, bound)
        end
        if has_deriv
            h = 1e-6
            fd = (fs(x + h) - fs(x - h)) / (2h)
            @test maximum(abs, SparseIR.deriv(fs)(x) - fd) <= 1e-5 * maximum(abs, fd)
        end
    end
end
