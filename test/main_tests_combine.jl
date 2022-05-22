using Test

@testset "Combining CovarianceMatrix structs" begin
    using DataFrames
    using LinearAlgebra
    using Statistics: std, var, mean, cov
    using HighFrequencyCovariance
    using Random
    using StableRNGs

    brownian_corr_matrix = Hermitian([
        1.0 0.75 0.5 0.0
        0.0 1.0 0.5 0.25
        0.0 0.0 1.0 0.25
        0.0 0.0 0.0 1.0
    ])
    assets = [:BARC, :HSBC, :VODL, :RYAL]
    assets1 = [:BARC]
    assets2 = [:BARC, :VODL, :RYAL]
    assets3 = [:HSBC, :VODL, :RYAL]
    rng = StableRNG(1)

    ts1, true_covar, micro_noise, update_rates = generate_random_path(
        4,
        2000;
        brownian_corr_matrix = brownian_corr_matrix,
        assets = assets,
        vols = [0.02, 0.03, 0.04, 0.05],
        rng = deepcopy(rng),
        syncronous = true,
    )
    ts2, true_covar, micro_noise, update_rates = generate_random_path(
        4,
        2000;
        brownian_corr_matrix = brownian_corr_matrix,
        assets = assets,
        vols = [0.02, 0.03, 0.04, 0.05],
        rng = deepcopy(rng),
        syncronous = true,
    )
    ts3, true_covar, micro_noise, update_rates = generate_random_path(
        4,
        2000;
        assets = assets,
        vols = [0.02, 0.03, 0.04, 0.05],
        rng = deepcopy(rng),
        syncronous = true,
    )


    iscloser(a, b) = (
        a.Correlation_error + a.Volatility_error < b.Correlation_error + b.Volatility_error
    )

    # Simple Volatility
    simple_vol = simple_volatility(ts1)
    @test all(values(simple_vol) .< 0.1)

    # Preav Convergence
    preav_estimate1 = preaveraged_covariance(ts1, assets1)
    preav_estimate2 = preaveraged_covariance(ts2, assets2)
    preav_estimate3 = preaveraged_covariance(ts3, assets3)

    # Testing the combination of multiple covariance matrices.
    vector_of_covars = [preav_estimate1, preav_estimate2, preav_estimate3]
    combined = combine_covariance_matrices(vector_of_covars, [1, 2, 3], [3, 2, 1])
    @test isnan(get_correlation(combined, :BARC, :HSBC)) # As they never appear together.

    # Testing order does not matter
    preav_estimate4 = preaveraged_covariance(ts3, assets)
    vector_of_covars = [preav_estimate1, preav_estimate2, preav_estimate4]
    combined = combine_covariance_matrices(vector_of_covars, [1, 2, 3], [3, 2, 1])
    vector_of_covars = [preav_estimate4, preav_estimate2, preav_estimate1]
    combined2 = combine_covariance_matrices(vector_of_covars, [3, 2, 1], [1, 2, 3])
    @test calculate_mean_abs_distance(combined, combined2).Correlation_error .< 10 * eps()
    @test calculate_mean_abs_distance(combined, combined2).Volatility_error .< 10 * eps()
end
