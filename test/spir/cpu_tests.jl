@testitem "FMA requirement of libsparseir (sparse-ir-rs#332)" tags=[:julia, :sparseir] begin
    using Test
    using SparseIR

    # The package loaded, so the library runs on this CPU
    @test SparseIR._check_cpu_supports_build() === nothing

    @test SparseIR._fma_requirement_error(false, false) === nothing
    @test SparseIR._fma_requirement_error(false, true) === nothing
    @test SparseIR._fma_requirement_error(true, true) === nothing
    err = SparseIR._fma_requirement_error(true, false)
    @test err isa SparseIR.SparseIRError
    @test occursin("fused multiply-add", err.msg)
    @test occursin("target-feature=-fma", err.msg)
end
