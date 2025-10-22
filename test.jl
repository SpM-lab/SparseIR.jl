ENV["SPARSEIR_LIB_PATH"] = "/Users/hiroshi/opt/libsparseir/lib/libsparseir.0.5.2.dylib"
using Revise
using ReTestItems
runtests("augment_tests.jl")
