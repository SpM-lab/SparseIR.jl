if !@isdefined sve_logistic
    const sve_logistic = Dict(
        10 => SparseIR.compute_sve(LogisticKernel(10.0)),
        42 => SparseIR.compute_sve(LogisticKernel(42.0)),
        10_000 => SparseIR.compute_sve(LogisticKernel(10_000.0)),
        (10_000, 1e-12) => SparseIR.compute_sve(LogisticKernel(10_000.0); ε=1e-12),
    )
end
