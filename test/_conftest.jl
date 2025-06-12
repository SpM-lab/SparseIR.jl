@testmodule CommonTestData begin
    using SparseIR: SVEResult, LogisticKernel

    const sve_logistic = Dict(
        10 => SVEResult(LogisticKernel(10.0)),
        42 => SVEResult(LogisticKernel(42.0)),
        10_000 => SVEResult(LogisticKernel(10_000.0)),
        (10_000, 1e-12) => SVEResult(LogisticKernel(10_000.0); ε=1e-12)
    )
end
