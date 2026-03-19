# deps

`Pkg.build("SparseIR")` builds the Rust backend used by `SparseIR.jl` and writes
the runtime artifacts into `deps/`.

Build source priority:

1. use `SPARSEIR_RUST_BACKEND_DIR` when it is set and points to an existing checkout
2. otherwise use a sibling checkout at `../sparse-ir-rs` when it exists
3. otherwise download pinned `sparse-ir-capi` `0.8.1` from crates.io and build it in a temporary workspace

Relative paths in `SPARSEIR_RUST_BACKEND_DIR` are resolved against the
`SparseIR.jl` package root.

After a successful build, the script:

- copies `libsparse_ir_capi.(dylib|so|dll)` into `deps/`
- regenerates `deps/C_API.jl` for installed package trees
- keeps `src/C_API.jl` as the source-tree fallback binding file
- updates `deps/backend.stamp`
- records build status in `deps/build-state.toml`
- writes detailed logs to `deps/build.log`

Build-time environment variables:

- `SPARSEIR_RUST_BACKEND_DIR=/path/to/sparse-ir-rs`
  Selects a local Rust backend checkout before the sibling default is checked.
- `SPARSEIR_BUILD_DEBUG=1`
  Keeps the temporary crates.io workspace after a successful build. Failed builds
  always keep the workspace path recorded in `deps/build-state.toml`.
- `SPARSEIR_BUILD_DEBUGINFO=none|line|limited|full`
  Controls the Rust debuginfo level passed to cargo. The default is `line`,
  which maps to Rust's `line-tables-only`.

Examples:

```bash
julia -e 'using Pkg; Pkg.build("SparseIR")'
SPARSEIR_BUILD_DEBUG=1 julia -e 'using Pkg; Pkg.build("SparseIR")'
SPARSEIR_BUILD_DEBUGINFO=full julia -e 'using Pkg; Pkg.build("SparseIR")'
```

`Pkg.add("SparseIR")` runs the build step automatically on first install.
`Pkg.develop(...)` does not, so development checkouts must run
`Pkg.build("SparseIR")` explicitly.
