@testmodule CommonTestData begin
    const sve_logistic = Dict(
        10 => SparseIR.SVEResult(SparseIR.LogisticKernel(10.0)),
        42 => SparseIR.SVEResult(SparseIR.LogisticKernel(42.0)),
        10_000 => SparseIR.SVEResult(SparseIR.LogisticKernel(10_000.0)),
        (10_000, 1e-12) => SparseIR.SVEResult(SparseIR.LogisticKernel(10_000.0); ε=1e-12)
    )
end
