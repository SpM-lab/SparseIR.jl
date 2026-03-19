using RustToolChain: cargo

include("build_support.jl")
using .BuildSupport

BuildSupport.main(; cargo_cmd=cargo())
