"""
    warmup(basis::AbstractBasis) -> nothing

Evaluate the Matsubara-frequency functions of `basis` once on the calling
thread, so that the first call from a worker thread finds everything compiled.

Julia's garbage collector synchronises threads with `SIGSEGV`, and some MPI
libraries install a handler for that signal (UCX; the transports Intel MPI 2021
selects do this). The process then aborts with a backtrace inside the compiler
the first time a worker thread has to compile an evaluation after
`MPI.Init()`, instead of compiling it (SpM-lab/SparseIR.jl#148,
JuliaParallel/MPI.jl#337). Evaluating once on the main thread avoids it;
`warmup` is that workaround spelled out.

The paths covered are those of [`uhat`](@ref): the vector call for both an
integer vector and a `MatsubaraFreq` vector, the single-function call, and the
`ComplexF64.(...)` conversion callers usually write around the result. The time
and frequency domains have the same exposure: evaluating `basis.u` and
`basis.v` once on the calling thread warms those.

Returns `nothing`.
"""
function warmup(basis::AbstractBasis)
    polys = uhat(basis)
    S = statistics(basis)
    # One of each parity the object accepts: odd for fermions, even for bosons.
    n_first, n_second = S isa Fermionic ? (1, -1) : (0, 2)

    # The integer-vector path.
    values = polys([n_first, n_second])
    # What callers write around it.
    ComplexF64.(values)
    # The MatsubaraFreq path, and the single function, when there is one.
    freqs = MatsubaraFreq{typeof(S)}[MatsubaraFreq{typeof(S)}(n_first)]
    polys(freqs)
    if length(polys) > 0
        ComplexF64.(polys(freqs))
        polys[1](n_first)
    end
    return nothing
end
