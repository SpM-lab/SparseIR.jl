module BackendLoader

using Libdl: dlext

deps_dir(root::AbstractString=dirname(@__DIR__)) = joinpath(root, "deps")
backend_stamp_path(root::AbstractString=dirname(@__DIR__)) = joinpath(deps_dir(root), "backend.stamp")
backend_library_path(root::AbstractString=dirname(@__DIR__)) = joinpath(deps_dir(root), "libsparse_ir_capi.$(dlext)")
generated_c_api_path(root::AbstractString=dirname(@__DIR__)) = joinpath(deps_dir(root), "C_API.jl")

function require_backend_library(root::AbstractString=dirname(@__DIR__))
    libpath = backend_library_path(root)
    isfile(libpath) || error("SparseIR backend not found at $libpath. Run `Pkg.build(\"SparseIR\")`.")
    return libpath
end

end
