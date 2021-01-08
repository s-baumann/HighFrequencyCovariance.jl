using HighFrequencyCovariance
using Test

# Run tests

println("Test various lowlevel functions")
@time @test include("test_lowlevel_functions.jl")

println("Test combining of covariance matrices")
@time @test include("main_tests_combine.jl")

println("Test finding nearest correlation matrix")
@time @test include("test_31_Higham.jl")

println("Test on syncronous observations")
@time @test include("main_tests_syncronous.jl")

println("Test on asyncronous observations")
@time @test include("main_tests_asyncronous.jl")
