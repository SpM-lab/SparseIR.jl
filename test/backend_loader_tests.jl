@testitem "backend loader" tags=[:julia] begin
    using Libdl: dlext
    using Test

    include(joinpath(dirname(@__DIR__), "src", "backend_loader.jl"))
    using .BackendLoader

    mktempdir() do root
        err = @test_throws ErrorException BackendLoader.require_backend_library(root)
        @test occursin("Pkg.build(\"SparseIR\")", sprint(showerror, err.value))
    end

    mktempdir() do root
        deps_dir = joinpath(root, "deps")
        mkpath(deps_dir)
        libpath = joinpath(deps_dir, "libsparse_ir_capi.$(dlext)")
        write(libpath, "")
        @test BackendLoader.backend_stamp_path(root) == joinpath(root, "deps", "backend.stamp")
        @test BackendLoader.backend_library_path(root) == libpath
        @test BackendLoader.require_backend_library(root) == libpath
    end
end
