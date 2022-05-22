using Documenter, HighFrequencyCovariance

makedocs(
    format = Documenter.HTML(),
    sitename = "HighFrequencyCovariance.jl",
    modules = [HighFrequencyCovariance],
    pages = Any[
        "Introduction"=>"index.md",
        "First Steps with HighFrequencyCovariance.jl"=>Any[
            "Algorithms"=>"1_algorithms.md",
            "Data Structures"=>"2_data_structures.md",
            "Writing Code"=>"3_WritingCode.md",
            "References"=>"9_references.md",
        ],
        "API"=>Any[
            "Types"=>"types.md",
            "Estimation Functions"=>"functions.md",
            "Helper Functions"=>"helper_functions.md",
            "Internal Functions"=>"internals.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/s-baumann/HighFrequencyCovariance.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
    devbranch = "main",
)
