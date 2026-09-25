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
