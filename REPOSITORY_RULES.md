# Repository Rules

These are `SparseIR.jl`-specific facts and conventions. Apply them on top of
the shared rules in [`SpM-lab/spm-agent-rules`](https://github.com/SpM-lab/spm-agent-rules)
(see `AGENTS.md`). Every rule below is verified against the current source
tree, `Project.toml`, and `.github/workflows/`; if the code changes in a way
that contradicts this file, fix this file in the same PR.

## Source Layout

- `src/C_API.jl` — low-level, near-mechanical `ccall` bindings to
  `libsparseir`. It is largely generated (see `utils/generate_C_API.jl`,
  `utils/generator.toml`, `utils/prologue.jl`, which use `Clang.jl` to parse
  the C headers). Opaque C types (`spir_basis`, `spir_kernel`,
  `spir_sve_result`, ...) are wrapped as `struct ... _private::Ptr{Cvoid} end`
  and are deliberately not exposed to users directly.
- `src/SparseIR.jl` — the module entry point. It `include`s `C_API.jl`,
  initializes the BLAS backend handed to the Rust/C side
  (`_init_sparseir_blas_backend`, called from `__init__()`), and then
  `include`s the high-level wrapper files in this order: `freq.jl`,
  `abstract.jl`, `kernel.jl`, `sve.jl`, `poly.jl`, `basis.jl`, `sampling.jl`,
  `dlr.jl`, `basis_set.jl`, `augment.jl`.
- High-level wrapper files under `src/` provide the user-facing, idiomatic
  Julia API on top of `C_API.jl`: e.g. `basis.jl` (`FiniteTempBasis`),
  `sampling.jl` (`TauSampling`, `MatsubaraSampling`, `evaluate`/`fit`),
  `dlr.jl` (`DiscreteLehmannRepresentation`), `freq.jl` (`MatsubaraFreq`,
  `BosonicFreq`, `FermionicFreq`), `kernel.jl` (`LogisticKernel`,
  `RegularizedBoseKernel`), `augment.jl` (`AugmentedBasis`, `TauConst`,
  `TauLinear`, `MatsubaraConst`), `basis_set.jl` (`FiniteTempBasisSet`).
  New user-facing functionality belongs here, not in `C_API.jl`.
- The public API surface is the `export` list in `src/SparseIR.jl`. Check it
  before adding, renaming, or removing a public symbol.

## The `libsparseir` Dependency And Local Override

- `SparseIR.jl` depends on the prebuilt `libsparseir_jll` package
  (`Project.toml`, `[deps]`/`[compat]`: `libsparseir_jll = "0.8, 0.9"`) for the
  compiled `libsparseir` C library. This is what most users and CI (`CI.yml`)
  use.
- `src/C_API.jl` resolves the library path at load time via
  `get_libsparseir()`: if `deps/libsparse_ir_capi.<dlext>` exists locally, it
  is used (with an `@info "Using local libsparseir: ..."` log line);
  otherwise it falls back to `libsparseir_jll.libsparseir`. There is no
  `SPARSEIR_LIB_PATH` environment-variable override in the current source —
  `test/runtests.jl` only prints `ENV["SPARSEIR_LIB_PATH"]`/`ENV["SPARSEIR_DEBUG"]`
  if they happen to be set, for operator visibility; setting
  `SPARSEIR_LIB_PATH` alone has no effect on which library gets loaded.
- The local override is populated by `deps/build.jl` (run via `Pkg.build`):
  if a sibling directory `../sparse-ir-rs` exists (i.e. checked out next to
  `SparseIR.jl`, not inside it), it runs
  `cargo build --release --features system-blas` there, copies
  `libsparse_ir_capi.<dlext>` into `deps/`, and regenerates
  `src/C_API.jl` by running `utils/generate_C_API.jl`. If `../sparse-ir-rs`
  does not exist, `deps/build.jl` does nothing and the JLL package is used.
  See `development.md` ("Using Local libsparseir for Development") for the
  step-by-step workflow.
- `SPARSEIR_DEBUG=1` (an actual environment variable read at runtime, see
  `README.md`) enables debug output from the underlying library, independent
  of which `libsparseir` binary is loaded.

## Running Tests

- The test suite uses `ReTestItems.jl` (`@testitem` blocks), not plain
  `Test.jl` scripts. `test/runtests.jl` calls
  `runtests(SparseIR; tags=[:julia])` — **only test items tagged `:julia` run
  by default** under `Pkg.test()`.
- Standard invocation: `julia --project=. -e 'using Pkg; Pkg.test()'` (or
  `Pkg.test("SparseIR")` from the parent environment).
- Test files live under two directories:
  - `test/spir/*.jl` — high-level API tests (tags include `:julia`, plus one
    of `:lib`, `:spir`, or `:debug`; e.g. `basis_tests.jl` uses
    `tags=[:julia, :lib]`, `dlr_tests.jl` uses `tags=[:julia, :spir]`).
  - `test/C_API/*.jl` — low-level C-API/ccall tests, tagged `:cinterface`
    (e.g. `cinterface_core_tests.jl`, `cinterface_sampling_tests.jl`,
    `cinterface_dlr_tests.jl`). These are **not** run by the default
    `tags=[:julia]` filter in `test/runtests.jl`; run them explicitly with
    `ReTestItems.runtests(SparseIR; tags=[:cinterface])` (or without a tag
    filter to run everything) when changing `src/C_API.jl` or anything that
    crosses the `ccall` boundary.
  - `test/aqua_tests.jl` runs `Aqua.test_all` for package-quality checks
    (ambiguities, stale deps, etc.) as part of the `[extras]`/`[targets]`
    `test` target declared in `Project.toml`.
- To exercise the code against an unreleased `libsparseir`, follow the local
  override workflow above (sibling `../sparse-ir-rs` checkout + `Pkg.build`)
  before running tests; this is also what `CI_with_latest_rust_backend.yml`
  automates in CI (see below).

## CI Entry Points

- `.github/workflows/CI.yml` — the main CI workflow. Runs on push/PR to
  `main`/`develop_v2xx`, weekly on a schedule, and on manual dispatch. Matrix:
  Julia `lts` and `1` on `ubuntu-latest`/x64 and `macOS-latest`/arm64
  (Windows is commented out — "libsparseir is not yet tested on Windows").
  Steps: `julia-actions/julia-buildpkg`, `julia-actions/julia-runtest`,
  coverage via `julia-actions/julia-processcoverage` +
  `codecov/codecov-action`. This job builds against the released
  `libsparseir_jll`, not a local build.
  A separate `docs` job builds and deploys documentation with
  `julia-actions/julia-docdeploy`.
- `.github/workflows/CI_with_latest_rust_backend.yml` — same Julia/OS matrix,
  but first checks out `SpM-lab/sparse-ir-rs` into `../sparse-ir-rs` (sibling
  of the checked-out `SparseIR.jl`) before building, so
  `deps/build.jl` picks it up and compiles against the latest Rust/C backend
  instead of the pinned JLL release. It also installs OpenBLAS
  (`libopenblas-dev` on Ubuntu, `openblas` via Homebrew on macOS) since the
  local build uses `--features system-blas`.
- `.github/workflows/PkgAdd.yml`, `CompatHelper.yml`, `Spelling.yml`,
  `TagBot.yml` — package registration checks, dependency-compat automation,
  spell-checking, and release tag-bot automation, respectively.
- When changing anything that affects the `ccall` boundary or the C API
  version pin (`libsparseir_jll` compat bounds in `Project.toml`), consider
  both `CI.yml` (released JLL) and `CI_with_latest_rust_backend.yml` (latest
  Rust backend) — a change can pass one and fail the other.

## Formatting

- `.JuliaFormatter.toml` configures `JuliaFormatter.jl` for this repository;
  format changed files with it before committing.
