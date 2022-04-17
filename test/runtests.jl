using HighFrequencyCovariance
using Test

# Run tests

println("Test combining of covariance matrices")
include("main_tests_combine.jl")

println("Test finding nearest correlation matrix")
include("test_31_Higham.jl")

println("Test on syncronous observations")
include("main_tests_syncronous.jl")

println("Test on asyncronous observations")
include("main_tests_asyncronous.jl")

println("Test on Convenience Functions")
include("main_tests_convenienceFunctions.jl")

println("Test on Generating Random Draws")
include("test_getting_draws.jl")
