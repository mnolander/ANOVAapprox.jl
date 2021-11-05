using Documenter, ANOVAapprox

makedocs(
    sitename = "ANOVAapprox.jl",
    format = Documenter.HTML(; prettyurls = false),
    modules = [ANOVAapprox],
    pages = [
        "Home" => "index.md",
        "Apprximation" => "approx.md",
        "Errors" => "errors.md",
        "Analysis" => "analysis.md",
        "About" => "about.md",
    ],
)

deploydocs(
    repo = "github.com/NFFT/ANOVAapprox.jl.git",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
)
