using SparseIR

const sve_logistic = Dict(
    10 => SparseIR.SVEResult(LogisticKernel(10.0)),
    42 => SparseIR.SVEResult(LogisticKernel(42.0)),
    10_000 => SparseIR.SVEResult(LogisticKernel(10_000.0)),
    (10_000, 1e-12) => SparseIR.SVEResult(LogisticKernel(10_000.0); ε=1e-12),
)
