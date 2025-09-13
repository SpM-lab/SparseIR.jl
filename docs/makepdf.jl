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
    sitename = "SparseIRjl example",
    format   = Documenter.LaTeX(; platform="tectonic"),
    pages    = ["Guide" => "guide.md"],
    plugins  = [bib]
)

pdffile = joinpath(@__DIR__, "build", "SparseIRjlexample.pdf")
titlepage = joinpath(@__DIR__, "src", "assets", "titlepage_sparseirjl.pdf")
run(`cpdf -merge $titlepage $pdffile 4-end -o $pdffile`)
