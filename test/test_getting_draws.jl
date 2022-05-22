using Test

@testset "Test getting draws" begin
    using DataFrames
    using Dates
    using LinearAlgebra
    using Statistics: std, var, mean, cov, cor
    using StochasticIntegrals
    using HighFrequencyCovariance
    using Random
    using Test

    brownian_corr_matrix = Hermitian([
        1.0 0.75 0.5 0.0
        0.0 1.0 0.5 0.25
        0.0 0.0 1.0 0.25
        0.0 0.0 0.0 1.0
    ])
    assets = [:BARC, :HSBC, :VODL, :RYAL]
    vols = [0.5, 0.6, 0.7, 0.8]
    time_period_per_unit = Dates.Hour(1)

    covar = CovarianceMatrix(brownian_corr_matrix, vols, assets, time_period_per_unit)

    draws = StochasticIntegrals.to_dataframe(get_draws(covar, 100000))

    distancefrom(x, tol) = (abs(x) < tol)

    # Testing everything is mean zero.
    @test distancefrom(mean(draws.BARC), 0.005)
    @test distancefrom(mean(draws.HSBC), 0.005)
    @test distancefrom(mean(draws.VODL), 0.005)
    @test distancefrom(mean(draws.RYAL), 0.005)

    # Testing vols
    @test distancefrom(std(draws.HSBC) - 0.6, 0.005)
    @test distancefrom(std(draws.BARC) - 0.5, 0.005)
    @test distancefrom(std(draws.VODL) - 0.7, 0.005)
    @test distancefrom(std(draws.RYAL) - 0.8, 0.005)

    # Testing Correlations.
    @test distancefrom(cor(draws.BARC, draws.HSBC) - 0.75, 0.005)
    @test distancefrom(cor(draws.BARC, draws.RYAL) - 0.0, 0.005)
    @test distancefrom(cor(draws.BARC, draws.VODL) - 0.5, 0.005)
    @test distancefrom(cor(draws.HSBC, draws.VODL) - 0.5, 0.005)
    @test distancefrom(cor(draws.HSBC, draws.RYAL) - 0.25, 0.005)
    @test distancefrom(cor(draws.VODL, draws.RYAL) - 0.25, 0.005)
end
