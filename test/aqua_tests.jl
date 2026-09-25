@testitem "Aqua" tags=[:julia, :aqua] begin
    using Test
    import Aqua
    using SparseIR

    # deps_compat stays off as before: ReTestItems in [extras] has no [compat] entry.
    Aqua.test_all(SparseIR; deps_compat=false)
end
