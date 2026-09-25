# Boundary contracts that are checked before any call into libsparseir: evaluation
# domains, Matsubara parity, sampling points, parameters, positive_only, element
# types, array wrappers and dimensions. The same contracts are tested in the
# Python suite (tests/test_boundary.py and tests/test_ffi_boundary.py in
# SpM-lab/sparse-ir).

@testitem "boundary: parameters" tags=[:julia, :boundary] begin
    using Test
    using SparseIR

    for (β, ωmax, ε) in ((0.0, 1.0, 1e-6), (-1.0, 1.0, 1e-6), (Inf, 1.0, 1e-6),
        (NaN, 1.0, 1e-6), (10.0, 0.0, 1e-6), (10.0, -1.0, 1e-6), (10.0, Inf, 1e-6),
        (10.0, 1.0, 0.0), (10.0, 1.0, -1e-6), (10.0, 1.0, NaN))
        @test_throws DomainError FiniteTempBasis(Fermionic(), β, ωmax, ε)
    end
    @test_throws DomainError FiniteTempBasis(Fermionic(), 10.0, 1.0, 1e-6; max_size=0)
    @test_throws DomainError FiniteTempBasisSet(-1.0, 1.0, 1e-6)
    for K in (LogisticKernel, RegularizedBoseKernel), Λ in (0.0, -1.0, Inf, NaN)
        @test_throws DomainError K(Λ)
    end
    @test_throws DomainError SparseIR.SVEResult(LogisticKernel(10.0), -1e-6)
    @test_throws DomainError SparseIR.SVEResult(LogisticKernel(10.0), NaN)

    # The kernel and a precomputed SVE must belong to the basis.
    @test_throws ArgumentError FiniteTempBasis(Fermionic(), 10.0, 1.0, 1e-6;
        kernel=RegularizedBoseKernel(10.0))
    @test_throws ArgumentError FiniteTempBasis(Fermionic(), 10.0, 1.0, 1e-6;
        kernel=LogisticKernel(42.0))
    sve_42 = SparseIR.SVEResult(LogisticKernel(42.0), 1e-6)
    @test_throws ArgumentError FiniteTempBasis(Fermionic(), 10.0, 1.0, 1e-6;
        sve_result=sve_42)

    # DLR poles must lie in the frequency window.
    basis = FiniteTempBasis(Fermionic(), 10.0, 1.0, 1e-6)
    @test_throws DomainError DiscreteLehmannRepresentation(basis, [-5.0, 0.1, 5.0])
end

@testitem "boundary: evaluation domains of u, v and the DLR functions" tags=[
    :julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    β, ωmax, ε = 10.0, 1.0, 1e-6
    @testset "$(nameof(typeof(stat)))" for stat in (Fermionic(), Bosonic())
        basis = get_basis(stat, β, ωmax, ε)
        dlr = DiscreteLehmannRepresentation(basis)
        for f in (basis.u, basis.u[1], dlr.u), τ in (1.5β, -1.5β, NaN, Inf)
            @test_throws DomainError f(τ)
            @test_throws DomainError f([0.5, τ])
        end
        for f in (basis.v, basis.v[2]), ω in (1.5ωmax, -1.5ωmax, NaN)
            @test_throws DomainError f(ω)
        end
        # The ends of the domains are accepted.
        U = basis.u([-β, 0.0, β])
        @test U[:, 3] == basis.u(β)
        @test sum(abs2, U) > 0
        @test sum(abs2, basis.v([-ωmax, ωmax])) > 0
    end
end

@testitem "boundary: Matsubara index parity and statistics in uhat" tags=[:julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    β, ωmax, ε = 10.0, 1.0, 1e-6
    bf = get_basis(Fermionic(), β, ωmax, ε)
    bb = get_basis(Bosonic(), β, ωmax, ε)
    for f in (bf.uhat, bf.uhat[1], DiscreteLehmannRepresentation(bf).uhat)
        @test_throws ArgumentError f(BosonicFreq(2))
        @test_throws DomainError f(2)
        @test_throws DomainError f([1, 2])
    end
    for f in (bb.uhat, bb.uhat[1], DiscreteLehmannRepresentation(bb).uhat)
        @test_throws ArgumentError f(FermionicFreq(1))
        @test_throws DomainError f(1)
    end
    # Integers of the right parity are the same as MatsubaraFreq.
    @test bf.uhat([1, -3]) == bf.uhat([FermionicFreq(1), FermionicFreq(-3)])
    @test bb.uhat(4) == bb.uhat(BosonicFreq(4))
end

@testitem "boundary: empty evaluation arrays" tags=[:julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    basis = get_basis(Fermionic(), 10.0, 1.0, 1e-6)
    L = length(basis)
    @test basis.u(Float64[]) isa Matrix{Float64}
    @test size(basis.u(Float64[])) == (L, 0)
    @test size(basis.v(Float64[])) == (L, 0)
    @test basis.uhat(FermionicFreq[]) isa Matrix{ComplexF64}
    @test size(basis.uhat(Int[])) == (L, 0)
end

@testitem "boundary: a C-level failure surfaces as SparseIRError" tags=[:julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    # The DLR basis functions are not piecewise polynomials; the C library
    # reports SPIR_NOT_SUPPORTED for their derivative.
    dlr = DiscreteLehmannRepresentation(get_basis(Fermionic(), 10.0, 1.0, 1e-6))
    err = try
        SparseIR.deriv(dlr.u)
        nothing
    catch e
        e
    end
    @test err isa SparseIR.SparseIRError
    @test err.status == SparseIR.C_API.SPIR_NOT_SUPPORTED
end

@testitem "boundary: sampling points" tags=[:julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    β, ωmax, ε = 10.0, 1.0, 1e-6
    basis = get_basis(Fermionic(), β, ωmax, ε)
    L = length(basis)

    # Order is kept, and evaluate follows it.
    points = reverse(collect(range(0.1, 9.9; length=L + 3)))
    smpl = TauSampling(basis; sampling_points=points)
    @test sampling_points(smpl) == points
    gl = [sin(0.7l) for l in 1:L]
    @test maximum(abs, evaluate(smpl, gl) - transpose(basis.u(points)) * gl) <=
          1e-13 * maximum(abs, evaluate(smpl, gl))

    # Views and wrappers of the points give the same sampling.
    for (label, v) in strided_views(points)
        @test sampling_points(TauSampling(basis; sampling_points=v)) == points
    end

    # Fewer points than basis functions are accepted for evaluation.
    few = [0.1, 0.4]
    @test evaluate(TauSampling(basis; sampling_points=few), gl) ≈
          transpose(basis.u(few)) * gl

    # Matsubara points: integers of the right parity only.
    @test_throws ArgumentError MatsubaraSampling(basis; sampling_points=[1.9, 3.0])
    @test_throws DomainError MatsubaraSampling(basis; sampling_points=[1, 2])
    @test_throws ArgumentError MatsubaraSampling(basis; sampling_points=[BosonicFreq(2)])
    @test sampling_points(MatsubaraSampling(basis; sampling_points=[1.0, -3.0])) ==
          [FermionicFreq(1), FermionicFreq(-3)]
    @test_throws ArgumentError MatsubaraSampling(basis; positive_only=true,
        sampling_points=[-1, 1, 3])
end

@testitem "boundary: Matsubara sampling keeps the given order" tags=[
    :julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR
    using LinearAlgebra: cond
    using Random: shuffle
    using StableRNGs: StableRNG

    # The C library orders Matsubara points ascending; the results must still
    # follow the order of the points the caller gave, as for TauSampling.
    function check_order(smpl, basis, pts)
        @test Int.(sampling_points(smpl)) == Int.(pts)
        gl = randn(StableRNG(7), length(basis))
        ref = transpose(basis.uhat(pts)) * gl
        @test maximum(abs, evaluate(smpl, gl) - ref) <= 1e-13 * maximum(abs, ref)
        atol = 100 * cond(smpl) * eps() * maximum(abs, gl)
        @test maximum(abs, fit(smpl, ref) - gl) <= atol
        gl2 = permutedims(hcat(gl, -2gl))            # 2 × L, sampled along dim 2
        ref2 = permutedims(hcat(ref, -2ref))
        @test maximum(abs, evaluate(smpl, gl2; dim=2) - ref2) <= 2e-13 * maximum(abs, ref)
        @test maximum(abs, fit(smpl, ref2; dim=2) - gl2) <= 2atol
        out = zeros(ComplexF64, length(pts))
        @test maximum(abs, evaluate!(out, smpl, gl) - ref) <= 1e-13 * maximum(abs, ref)
        out = zeros(ComplexF64, length(basis))
        @test maximum(abs, fit!(out, smpl, ref) - gl) <= atol
    end

    @testset "$(nameof(typeof(stat))), positive_only=$positive_only" for stat in (
            Fermionic(), Bosonic()), positive_only in (false, true)
        basis = get_basis(stat, 10.0, 1.0, 1e-6)
        pts = sampling_points(MatsubaraSampling(basis; positive_only))
        pts = shuffle(StableRNG(11), pts)
        check_order(
            MatsubaraSampling(basis; positive_only, sampling_points=pts), basis, pts)
    end

    @testset "augmented basis and DLR" begin
        bb = get_basis(Bosonic(), 10.0, 1.0, 1e-6)
        bf = get_basis(Fermionic(), 10.0, 1.0, 1e-6)
        for basis in (AugmentedBasis(bb, MatsubaraConst), DiscreteLehmannRepresentation(bf))
            pts = shuffle(StableRNG(11), sampling_points(MatsubaraSampling(basis)))
            check_order(MatsubaraSampling(basis; sampling_points=pts), basis, pts)
        end
    end
end

@testitem "boundary: positive_only rejects complex coefficients" tags=[:julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    basis = get_basis(Fermionic(), 10.0, 1.0, 1e-6)
    L = length(basis)
    smpl = MatsubaraSampling(basis; positive_only=true)
    gl = [cos(0.3l) for l in 1:L]
    @test_throws ArgumentError evaluate(smpl, gl .+ 1im .* gl)
    # A real quantity is accepted, also in a complex array such as a fit result.
    giv = evaluate(smpl, gl)
    fitted = fit(smpl, giv)
    @test fitted isa Vector{ComplexF64}
    @test maximum(abs, evaluate(smpl, fitted) - giv) <= 1e-12 * maximum(abs, giv)
end

@testitem "boundary: element types, wrappers and non-finite input" tags=[:julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    basis = get_basis(Fermionic(), 10.0, 1.0, 1e-6)
    L = length(basis)
    dlr = DiscreteLehmannRepresentation(basis)
    τs = TauSampling(basis)
    iω = MatsubaraSampling(basis)
    gl = [(-1)^l * 0.5^l for l in 1:L]
    gl32 = Float32.(gl)          # exactly representable: conversion is exact
    reference(f, x) = f(x isa AbstractArray{<:Complex} ? ComplexF64.(x) : Float64.(x))

    transforms = ["TauSampling evaluate" => x -> evaluate(τs, x),
        "TauSampling fit" => x -> fit(τs, x),
        "MatsubaraSampling evaluate" => x -> evaluate(iω, x),
        "from_IR" => x -> from_IR(dlr, x)]
    @testset "$name" for (name, f) in transforms
        for x in (gl32, ComplexF32.(gl32 .+ 1im .* gl32), round.(Int, 8 .* gl32),
            Float16.(gl32), Rational{Int}.(round.(Int, 8 .* gl32)))
            @test f(x) == reference(f, x)
        end
        for (label, v) in strided_views(gl)
            @test f(v) == f(gl)
        end
        @test_throws ArgumentError f([gl[1:(end - 1)]; NaN])
        @test_throws ArgumentError f(fill("a", L))
    end
    @testset "to_IR" begin
        g_dlr = from_IR(dlr, gl)
        @test to_IR(dlr, Float32.(g_dlr)) == to_IR(dlr, Float64.(Float32.(g_dlr)))
        for (label, v) in strided_views(g_dlr)
            @test to_IR(dlr, v) == to_IR(dlr, g_dlr)
        end
        @test_throws ArgumentError to_IR(dlr, [g_dlr[1:(end - 1)]; Inf])
    end
    @testset "MatsubaraSampling fit" begin
        giv = evaluate(iω, gl)
        @test fit(iω, ComplexF32.(giv)) == fit(iω, ComplexF64.(ComplexF32.(giv)))
        @test fit(iω, real.(giv)) == fit(iω, ComplexF64.(real.(giv)))
        for (label, v) in strided_views(giv)
            @test fit(iω, v) == fit(iω, giv)
        end
    end

    # Output buffers must be dense Float64/ComplexF64 arrays.
    @test_throws ArgumentError evaluate!(zeros(Float32, npoints(τs)), τs, gl)
    @test_throws ArgumentError evaluate!(
        view(zeros(2npoints(τs)), 1:2:(2npoints(τs))), τs, gl)
    @test_throws ArgumentError fit!(zeros(Float32, L), τs, evaluate(τs, gl))
    @test_throws DimensionMismatch evaluate!(zeros(npoints(τs), 1), τs, gl)

    # Wrong lengths are reported before the call.
    @test_throws DimensionMismatch evaluate(τs, rand(L + 1))
    @test_throws DimensionMismatch fit(τs, rand(npoints(τs) + 1))
    @test_throws DimensionMismatch evaluate!(zeros(npoints(τs) + 1), τs, gl)
    @test_throws DimensionMismatch from_IR(dlr, rand(L + 1))
    @test_throws ArgumentError evaluate(τs, gl; dim=2)
end

@testitem "boundary: dim on three-dimensional input" tags=[:julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    @testset "$(nameof(typeof(stat)))" for stat in (Fermionic(), Bosonic())
        basis = get_basis(stat, 10.0, 1.0, 1e-6)
        L = length(basis)
        dlr = DiscreteLehmannRepresentation(basis)
        pairs = ["TauSampling" => (x, d) -> evaluate(TauSampling(basis), x; dim=d),
            "TauSampling fit" => (x, d) -> fit(TauSampling(basis),
                evaluate(TauSampling(basis), x; dim=d); dim=d),
            "MatsubaraSampling" => (x, d) -> evaluate(MatsubaraSampling(basis), x; dim=d),
            "MatsubaraSampling(positive_only)" => (x, d) -> fit(
                MatsubaraSampling(basis; positive_only=true),
                evaluate(MatsubaraSampling(basis; positive_only=true), x; dim=d); dim=d),
            "from_IR" => (x, d) -> from_IR(dlr, x, d),
            "to_IR(from_IR)" => (x, d) -> to_IR(dlr, from_IR(dlr, x, d), d)]
        data1 = [sin(1.3i + 0.7j - 0.2k) for i in 1:L, j in 1:3, k in 1:5]   # basis axis first
        for (name, f) in pairs, d in 1:3
            perm = d == 1 ? (1, 2, 3) : d == 2 ? (2, 1, 3) : (2, 3, 1)
            data = permutedims(data1, perm)          # basis axis at d
            out = f(data, d)
            ref = f(data1, 1)
            @test maximum(abs, permutedims(out, invperm(perm)) - ref) <=
                  1e-12 * maximum(abs, ref)
        end
    end
end

@testitem "boundary: function selections, sizes and basis truncation" tags=[
    :julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR

    basis = get_basis(Fermionic(), 10.0, 1.0, 1e-6)
    L = length(basis)
    for fs in (basis.u, basis.v, basis.uhat)
        @test size(fs) == (L,)
        # The C library panics on an empty selection (SpM-lab/sparse-ir-rs#269).
        @test_throws ArgumentError fs[1:0]
        @test_throws ArgumentError fs[Int[]]
    end
    # Untyped vectors of real points are accepted.
    @test basis.u(Any[0.1, 0.2]) == basis.u([0.1, 0.2])

    # basis[1:n] keeps the n most significant singular values and functions.
    part = basis[1:3]
    @test part isa FiniteTempBasis
    @test length(part) == 3
    @test part.s == basis.s[1:3]
    @test isapprox(part.u(0.5), basis.u(0.5)[1:3]; rtol=1e-14)
    @test isapprox(part.uhat(3), basis.uhat(3)[1:3]; rtol=1e-14)
    @test length(basis[1:L]) == L
    @test_throws ArgumentError basis[2:3]
    @test_throws BoundsError basis[1:(L + 1)]
    @test_throws BoundsError basis[1:0]
    aug = AugmentedBasis(get_basis(Bosonic(), 10.0, 1.0, 1e-6), TauConst, TauLinear)
    @test length(aug[1:5]) == 5
end

@testitem "boundary: augmented and DLR arguments, aliasing, conditioning" tags=[
    :julia, :boundary] setup=[SIRTestSetup] begin
    using Test
    using SparseIR
    using LinearAlgebra: cond
    using StableRNGs

    bf = get_basis(Fermionic(), 10.0, 1.0, 1e-6)
    bb = get_basis(Bosonic(), 10.0, 1.0, 1e-6)
    aug = AugmentedBasis(bb, TauConst, TauLinear)
    vertex = AugmentedBasis(bf, MatsubaraConst)

    # The sampling-point checks of a plain basis hold for an augmented one.
    @test_throws ArgumentError MatsubaraSampling(aug; sampling_points=[1.9])
    @test_throws DomainError MatsubaraSampling(aug; sampling_points=[1])
    @test_throws ArgumentError MatsubaraSampling(aug; sampling_points=[FermionicFreq(1)])
    @test_throws ArgumentError MatsubaraSampling(aug; positive_only=true,
        sampling_points=[-2, 0, 2])
    @test_throws ArgumentError MatsubaraSampling(bf; sampling_points=[1e300])

    # Parity and statistics of the augmented functions
    @test_throws DomainError aug.uhat(1)
    # MatsubaraConst works identically for both statistics (its docstring);
    # TauConst and TauLinear are defined for one statistics only.
    @test vertex.uhat[1](FermionicFreq(3)) == 1
    @test_throws ArgumentError TauConst(10.0)(FermionicFreq(1))
    @test_throws ArgumentError TauLinear(10.0)(FermionicFreq(1))

    # A DLR samples at the default points of its IR basis, as in the Python
    # wrapper (the C API gives a DLR no default Matsubara points of its own),
    # and is built on an IR basis only.
    dlr = DiscreteLehmannRepresentation(bf)
    for positive_only in (false, true)
        @test sampling_points(MatsubaraSampling(dlr; positive_only)) ==
              sampling_points(MatsubaraSampling(bf; positive_only))
    end
    @test_throws ArgumentError DiscreteLehmannRepresentation(aug)

    # An SVE of another kernel type is rejected like one of another cutoff.
    sve_rb = SparseIR.SVEResult(RegularizedBoseKernel(10.0), 1e-6)
    @test_throws ArgumentError FiniteTempBasis(
        Bosonic(), 10.0, 1.0, 1e-6; sve_result=sve_rb)

    # Output and input must not share memory.
    smpl = TauSampling(bf)                   # as many points as functions
    g = randn(StableRNG(5), length(bf))
    @test_throws ArgumentError evaluate!(g, smpl, g)
    @test_throws ArgumentError fit!(g, smpl, g)

    # positive_only: cond is that of the real least-squares problem the fit
    # solves; the C library reports that of the complex matrix
    # (SpM-lab/sparse-ir-rs#270).
    for basis in (bf, bb, aug)
        smpl = MatsubaraSampling(basis; positive_only=true)
        A = transpose(basis.uhat(sampling_points(smpl)))
        @test isapprox(cond(smpl), cond(vcat(real(A), imag(A))); rtol=1e-10)
    end
end
