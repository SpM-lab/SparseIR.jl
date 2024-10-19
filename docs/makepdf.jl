using SparseIR
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(SparseIR, :DocTestSetup, :(using SparseIR))

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)

makedocs(; modules  = [SparseIR],
    authors  = "Samuel Badr",
    sitename = "SparseIR.jl Guide",
    format   = Documenter.LaTeX(; platform="tectonic"),
    pages    = ["Guide" => "guide.md"],
    plugins  = [bib]
)
