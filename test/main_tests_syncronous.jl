using Test

@testset "Synchronous timing" begin
    using DataFrames
    using Dates
    using LinearAlgebra
    using Statistics: std, var, mean, cov
    using HighFrequencyCovariance
    using Random
    using StableRNGs
    using Test

    brownian_corr_matrix = Hermitian([1.0 0.75 0.5 0.0;
                                      0.0 1.0 0.5 0.25;
                                      0.0 0.0 1.0 0.25;
                                      0.0 0.0 0.0 1.0])
    assets = [:BARC, :HSBC, :VODL, :RYAL]
    time_period_per_unit = Dates.Hour(1)
    rng  = StableRNG(1)

    ts0, true_covar, micro_noise, update_rates = generate_random_path(4, 4; brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05], rng = deepcopy(rng), syncronous = true, time_period_per_unit = time_period_per_unit)
    ts1, true_covar, micro_noise, update_rates = generate_random_path(4, 2000; brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05], rng = deepcopy(rng), syncronous = true, time_period_per_unit = time_period_per_unit)
    ts2, true_covar, micro_noise, update_rates = generate_random_path(4, 50000; brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05], rng = deepcopy(rng), syncronous = true, time_period_per_unit = time_period_per_unit)

    iscloser(a,b) = (a.Correlation_error + a.Volatility_error < b.Correlation_error + b.Volatility_error)

    # Simple Volatility
    simple_vol = simple_volatility(ts1)
    @test all(values(simple_vol) .< 0.1)
    simple_vol = simple_volatility(ts1; use_all_obs = true)
    @test all(values(simple_vol) .< 1)
    simple_vol = simple_volatility(ts1; fixed_spacing = 20.0)
    @test all(values(simple_vol) .< 0.1)
    simple_vol = simple_volatility(ts1; fixed_spacing = Dict(assets .=> repeat([20],4)))
    @test all(values(simple_vol) .< 0.1)

    # Preav Convergence
    @test_logs (:warn,"Cannot estimate the correlation matrix with the preaveraging technique with 4 ticks. There are insufficient ticks for [:BARC, :HSBC, :VODL, :RYAL]") preaveraged_covariance(ts0, assets) # This will not work due to insufficient data.
    preav_estimate1 = preaveraged_covariance(ts1, assets)
    preav_estimate2 = preaveraged_covariance(ts2, assets; regularisation = missing)
    @test iscloser(calculate_mean_abs_distance(preav_estimate2, true_covar), calculate_mean_abs_distance(preav_estimate1, true_covar))
    @test valid_correlation_matrix(preav_estimate1)
    @test valid_correlation_matrix(preav_estimate2)

    # simple Convergence
    simple_estimate1 = simple_covariance(ts1, assets; refresh_times = true) # Testing refresh sampling
    simple_estimate1 = simple_covariance(ts1, assets; fixed_spacing = 10) # Testing fixed spacing
    simple_estimate1 = simple_covariance(ts1, assets)
    simple_estimate2 = simple_covariance(ts2, assets)
    @test valid_correlation_matrix(simple_estimate1)
    @test valid_correlation_matrix(simple_estimate2)

    # bnhls Convergence
    @test_logs (:warn,"Cannot estimate the correlation matrix with the bnhls method with only 4 ticks.") bnhls_covariance(ts0, assets) # This will not work due to insufficient data.
    bnhls_estimate1 = bnhls_covariance(ts1, assets; kernel = quadratic_spectral)
    bnhls_estimate1 = bnhls_covariance(ts1, assets; kernel = fejer)
    bnhls_estimate1 = bnhls_covariance(ts1, assets; kernel = tukey_hanning)
    bnhls_estimate1 = bnhls_covariance(ts1, assets; kernel = bnhls_2008)
    bnhls_estimate1 = bnhls_covariance(ts1, assets)
    bnhls_estimate2 = bnhls_covariance(ts2, assets)
    @test iscloser(calculate_mean_abs_distance(bnhls_estimate2, true_covar), calculate_mean_abs_distance(bnhls_estimate1, true_covar))
    @test valid_correlation_matrix(bnhls_estimate1)
    @test valid_correlation_matrix(bnhls_estimate2)

    # Preav Convergence
    spectral_estimate1 = spectral_covariance(ts1, assets; num_blocks = 1) # not many observations so need to reduce the number of blocks here
    spectral_estimate2 = spectral_covariance(ts2, assets)
    iscloser(calculate_mean_abs_distance(spectral_estimate2, true_covar), calculate_mean_abs_distance(spectral_estimate1, true_covar))
    @test valid_correlation_matrix(spectral_estimate1)
    @test valid_correlation_matrix(spectral_estimate2)

    # two scales Convergence
    two_scales_estimate1 = two_scales_covariance(ts1, assets)
    two_scales_estimate2 = two_scales_covariance(ts2, assets)
    @test valid_correlation_matrix(two_scales_estimate1)
    @test valid_correlation_matrix(two_scales_estimate2)


    #############################
    # Conversion to a covariance function
    dura = Dates.Minute(137)
    covarr = covariance(two_scales_estimate2, dura )
    corr, vols =  cov_to_cor_and_vol(covarr, Nanosecond(two_scales_estimate2.time_period_per_unit).value, Nanosecond(dura).value     )
    corr, vols =  cov_to_cor_and_vol(covarr,   Nanosecond(dura).value / Nanosecond(two_scales_estimate2.time_period_per_unit).value     )
    @test all(corr .< two_scales_estimate2.correlation) .< 10*eps()
    @test all(vols .< two_scales_estimate2.volatility) .< 10*eps()

    #############################
     # Serialisation and deserialisation

    true_df = DataFrame(true_covar, Dict([:estimation] .=> ["True Covariance Matrix"]))
    reconstituted_df = CovarianceMatrix(true_df)
    @test calculate_mean_abs_distance(true_covar, reconstituted_df).Correlation_error .< 10*eps()
    @test calculate_mean_abs_distance(true_covar, reconstituted_df).Volatility_error .< 10*eps()
    # Reshuffling rows to make sure it still works.
    true_df2 = true_df[randperm(MersenneTwister(1234), nrow(true_df)),:]
    reconstituted_df = CovarianceMatrix(true_df2)
    @test calculate_mean_abs_distance(true_covar, reconstituted_df).Correlation_error .< 10*eps()
    @test calculate_mean_abs_distance(true_covar, reconstituted_df).Volatility_error .< 10*eps()


    # Other regularisation algos:
    two_scales_estimate_iden = two_scales_covariance(ts2, assets; regularisation = :identity_regularisation)
    two_scales_estimate_nearest_corr = two_scales_covariance(ts2, assets; regularisation = :nearest_correlation_matrix)
    two_scales_estimate_nearest_psd = two_scales_covariance(ts2, assets; regularisation = :nearest_psd_matrix)
    two_scales_estimate_eigen = two_scales_covariance(ts2, assets; regularisation = :eigenvalue_clean)
    # Testing that these are different (due to different regularisation)
    dist = calculate_mean_abs_distance(two_scales_estimate_iden, two_scales_estimate_nearest_corr)
    @test dist.Correlation_error .> 1000*eps()
    dist = calculate_mean_abs_distance(two_scales_estimate_iden, two_scales_estimate_eigen)
    @test dist.Correlation_error .> 1000*eps()


    # Running regularistation on a CovarianceMatrix's correlation matrix.
    psd_mat = two_scales_covariance(ts2, assets; regularisation = :nearest_correlation_matrix)
    @test valid_correlation_matrix(psd_mat)
    reg1 = identity_regularisation(psd_mat, 0.5; apply_to_covariance = false) # Inputting a weight explicitly.
    @test calculate_mean_abs_distance(psd_mat, reg1).Correlation_error .> 10*eps()
    @test calculate_mean_abs_distance(psd_mat, reg1).Volatility_error .< 10*eps()

    reg1_corr = identity_regularisation(psd_mat, ts2; apply_to_covariance = false)
    @test calculate_mean_abs_distance(psd_mat, reg1_corr).Correlation_error .> 10*eps()
    @test calculate_mean_abs_distance(psd_mat, reg1_corr).Volatility_error .< 10*eps()

    reg2 = nearest_correlation_matrix(psd_mat, ts2)
    @test calculate_mean_abs_distance(psd_mat, reg2).Correlation_error .< 10*eps() # No change as we are already psd
    @test calculate_mean_abs_distance(psd_mat, reg2).Volatility_error .< 10*eps()

    reg3 = nearest_psd_matrix(psd_mat, ts2)
    @test calculate_mean_abs_distance(psd_mat, reg3).Correlation_error .< 10*eps() # No change as we are already psd
    @test calculate_mean_abs_distance(psd_mat, reg3).Volatility_error .< 1000*eps()

    reg3_corr = nearest_psd_matrix(psd_mat, ts2; apply_to_covariance = false)
    @test calculate_mean_abs_distance(psd_mat, reg3_corr).Volatility_error .< 10*eps()
    @test calculate_mean_abs_distance(psd_mat, reg3_corr).Correlation_error .< 10*eps() # No change as we are already psd

    reg4 = eigenvalue_clean(psd_mat, ts2)
    @test calculate_mean_abs_distance(psd_mat, reg4).Correlation_error .> 2*eps()
    @test calculate_mean_abs_distance(psd_mat, reg4).Volatility_error .> 10*eps()

    reg4_cov = eigenvalue_clean(psd_mat, ts2; apply_to_covariance = false)
    @test calculate_mean_abs_distance(psd_mat, reg4_cov).Correlation_error .< 2*eps()
    @test calculate_mean_abs_distance(psd_mat, reg4_cov).Volatility_error .< 10*eps()

    # Testing the covariance Function
    @test all((covariance(true_covar, Hour(1)) * 24 .-   covariance(true_covar, Day(1))   ) .< 10*eps())
end
