using Libdl: Libdl
using libsparseir_jll: libsparseir_jll

function get_libsparseir()
    deps_dir = joinpath(dirname(@__DIR__), "deps")
    local_libsparseir_path = joinpath(deps_dir, "libsparse_ir_capi.$(Libdl.dlext)")
    if isfile(local_libsparseir_path)
        @info "Using local libsparseir: $local_libsparseir_path"
        return local_libsparseir_path
    else
        return libsparseir_jll.libsparseir
    end
end

const libsparseir = get_libsparseir()
