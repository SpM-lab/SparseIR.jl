# deps

The `build.jl` script provides developer-focused support for building the Rust backend required by `SparseIR.jl`. It assumes that the Rust crate `sparse-ir-rs` is located in the same parent directory as the `SparseIR.jl` package.

After making changes to the Rust code, it is assumed that you will rebuild the Julia package as follows:

```
$ cd path/to/SparseIR.jl
$ ls
src/ utils/ test/ ...
$ julia -e 'using Pkg; Pkg.build()'
```

This process will update `src/C_API.jl` and copy the `libsparse_ir_capi.dylib` (or the appropriate shared library) to the `deps/` directory. During `Pkg.test()`, the dynamic library `deps/libsparse_ir_capi.[dylib|so]` will be linked.
