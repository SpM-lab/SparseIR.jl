const sve_logistic = Dict(10 => compute(LogisticKernel(10.0)),
                          42 => compute(LogisticKernel(42.0)),
                          10_000 => compute(LogisticKernel(10_000.0)))

const sve_reg_bose = Dict(10 => compute(RegularizedBoseKernel(10.0)),
                          10_000 => compute(RegularizedBoseKernel(10_000.0)))