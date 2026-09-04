# AGENTS.md

SparseIR.jl is a Julia wrapper over the [libsparseir](https://github.com/SpM-lab/libsparseir)
C API, accessed via `ccall`. It exposes the sparse intermediate representation
(IR) of many-body Green's functions: IR basis construction, sparse sampling in
imaginary time and Matsubara frequency, and the discrete Lehmann
representation (DLR).

## Shared Rules

Read the SpM-lab shared agent rules before making changes:
<https://github.com/SpM-lab/spm-agent-rules> — start at
[`rules/index.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/index.md)
and load only the rule files the current task needs.

If network access is unavailable, look for a sibling checkout at
`../spm-agent-rules`.

For this repository, the routing table in `rules/index.md` typically resolves to:

- [`common.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/common.md) — cross-language repository policy
- [`ffi-boundary.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/ffi-boundary.md) — the C boundary crossed by every `ccall`
- [`numerical-conventions.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/numerical-conventions.md) — the physics contracts (statistics, `tau`, dtype, `eps`)
- [`testing.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/testing.md) — what the test suite must cover
- [`julia.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/julia.md) — `SparseIR.jl`- and `ccall`-specific rules

Do not bulk-load the whole shared-rules repository; load only what the task needs.

## Repository-Specific Rules

See [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md) for facts specific to this
repository: source layout, how to run the test suite, the local-`libsparseir`
override used for development against an unreleased C API, and CI entry
points.

## Precedence

Repository-local rules in `REPOSITORY_RULES.md` override the shared rules
above when they are more specific. Note any such override in the pull request
description when it affects review.
