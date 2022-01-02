using HighFrequencyCovariance
using Test

# Run tests

println("Test combining of covariance matrices")
@time @test include("main_tests_combine.jl")

println("Test finding nearest correlation matrix")
@time @test include("test_31_Higham.jl")

println("Test on syncronous observations")
@time @test include("main_tests_syncronous.jl")

println("Test on asyncronous observations")
@time @test include("main_tests_asyncronous.jl")

println("Test on Convenience Functions")
@time @test include("main_tests_convenienceFunctions.jl")

println("Test on Generating Random Draws")
@time @test include("test_getting_draws.jl")
