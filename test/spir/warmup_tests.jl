@testitem "warmup: touches the Matsubara paths and changes no result" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR

    for (S, ns) in ((Fermionic(), (1, -1)), (Bosonic(), (0, 2)))
        basis = FiniteTempBasis(S, 10.0, 4.0, 1e-8)
        n_basis = length(basis)

        # The values the caller gets, before and after the warmup.
        before = basis.uhat([ns[1], ns[2]])

        @test warmup(basis) === nothing
        @test warmup(basis) === nothing          # idempotent

        after = basis.uhat([ns[1], ns[2]])
        @test after == before
        @test after isa Matrix{ComplexF64}
        @test size(after) == (n_basis, 2)
    end
end

@testitem "warmup: works for a DLR, whose uhat comes from its IR basis" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR

    basis = FiniteTempBasis(Fermionic(), 10.0, 4.0, 1e-8)
    dlr = DiscreteLehmannRepresentation(basis, [0.1, -0.3])

    before = dlr.uhat([1, -1])
    @test warmup(dlr) === nothing
    @test dlr.uhat([1, -1]) == before
end

@testitem "warmup: the warm path is what a threaded caller hits first" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR
    using Base.Threads: @threads

    basis = FiniteTempBasis(Fermionic(), 10.0, 4.0, 1e-8)
    freqs = FermionicFreq[-FermionicFreq(1), FermionicFreq(1)]
    reference = ComplexF64.(basis.uhat(freqs))

    warmup(basis)

    # Two worker threads evaluate for the first time. Without MPI this is a
    # plain smoke test; with an MPI library that intercepts SIGSEGV it is the
    # shape of SpM-lab/SparseIR.jl#148.
    results = Matrix{ComplexF64}[]
    lk = ReentrantLock()
    @threads for _ in 1:2
        values = ComplexF64.(basis.uhat(freqs))
        lock(lk) do
            push!(results, values)
        end
    end

    @test length(results) == 2
    @test all(r -> r == reference, results)
end
