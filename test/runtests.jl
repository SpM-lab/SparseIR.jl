#ENV["SPARSEIR_LIB_PATH"] = "/Users/hiroshi/projects/sparse-ir/sparseir-rust/target/release/libsparse_ir_capi.dylib"
#ENV["SPARSEIR_DEBUG"] = "1"
if "SPARSEIR_LIB_PATH" in keys(ENV)
    println("Running tests with SPARSEIR_LIB_PATH: ", ENV["SPARSEIR_LIB_PATH"])
end
if "SPARSEIR_DEBUG" in keys(ENV)
    println("Running tests with SPARSEIR_DEBUG: ", ENV["SPARSEIR_DEBUG"])
end
using SparseIR
using ReTestItems

# Run every test item: the high-level suite (tag :julia), the low-level C API
# items (tag :cinterface) and Aqua. Select a subset with the `tags` keyword of
# `ReTestItems.runtests` when iterating locally.
runtests(SparseIR)
