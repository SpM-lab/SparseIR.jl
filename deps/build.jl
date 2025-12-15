using Pkg

if VERSION < v"1.11"
    Pkg.add(url="https://github.com/AtelierArith/RustToolChains.jl.git")
end

using RustToolChains: cargo
using Libdl: dlext

const DEV_DIR::String = joinpath(dirname(dirname(@__DIR__)), "sparse-ir-rs")
# Check if the sparse-ir-rs directory exists locally; if not, do nothing.
# If it exists, build the Rust project and copy libsparse_ir_capi.<ext> to deps/.
if isdir(DEV_DIR)
    cd(DEV_DIR) do
        run(`$(cargo()) build --release --features system-blas`)
    end
    libsparseir_path = joinpath(DEV_DIR, "target", "release", "libsparse_ir_capi.$(dlext)")
    cp(libsparseir_path, joinpath(@__DIR__, "libsparse_ir_capi.$(dlext)"); force=true)

    cd(joinpath(dirname(@__DIR__), "utils")) do
        run(`$(Base.julia_cmd()) --project generate_C_API.jl`)
    end
end
