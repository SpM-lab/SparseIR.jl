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

@testitem "boundary: evaluation domains of u, v and the DLR functions" tags=[:julia, :boundary] setup=[SIRTestSetup] begin
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
