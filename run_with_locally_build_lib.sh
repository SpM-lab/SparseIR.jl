rm -rf ~/.julia/compiled/v*/SparseIR

export SPARSEIR_LIB_PATH="/Users/hiroshi/projects/sparse-ir/sparseir-rust/target/release/libsparse_ir_capi.dylib"
export RUST_BACKTRACE=1
export SPARSEIR_DEBUG=1

julia --project=@. -e "using Pkg; Pkg.test()"
