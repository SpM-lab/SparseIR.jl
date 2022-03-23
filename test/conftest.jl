using SparseIR

const compute_sve = compute

const sve_logistic = Dict(10 => compute_sve(LogisticKernel(10)),
                          42 => compute_sve(LogisticKernel(42)),
                          10_000 => compute_sve(LogisticKernel(10_000)))

const sve_reg_bose = Dict(10 => compute_sve(RegularizedBoseKernel(10)),
                          10_000 => compute_sve(RegularizedBoseKernel(10_000)))