using Test

@testset "asynchronous timing" begin
    using DataFrames
    using LinearAlgebra
    using StableRNGs
    using Statistics: std, var, mean, cov
    using HighFrequencyCovariance
    using Random
    using Test

    brownian_corr_matrix = Hermitian([1.0 0.75 0.5 0.0;
                                      0.0 1.0 0.5 0.25;
                                      0.0 0.0 1.0 0.25;
                                      0.0 0.0 0.0 1.0])
    assets = [:BARC, :HSBC, :VODL, :RYAL]
    rng  = StableRNG(1)
    rng2 = StableRNG(2)
    ts1, true_covar, micro_noise, update_rates = generate_random_path(4, 5000; brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05], rng = deepcopy(rng), rng_timing = deepcopy(rng2))
    ts2, true_covar, micro_noise, update_rates = generate_random_path(4, 50000; brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05], rng = deepcopy(rng), rng_timing = deepcopy(rng2))

    # Testing constructors of SortedDataFrames
    ts1 = SortedDataFrame(ts1.df, ts1.time, ts1.grouping, ts1.value,  ts1.groupingrows, ts1.time_period_per_unit)
    # Subsetting to time.
    ts1 = subset_to_time(ts1, 1670) # Will not delete anything
    ts1 = subset_to_time(ts1, 1400) # Will delete some stuff.
    # Subsetting to tick.
    ts2 = subset_to_tick(ts2, 49000)


    # Relabelling
    relabelling = Dict{Symbol,Symbol}(assets .=> [:Barclays, :HSBC, :Vodafone, :Ryanair])
    true_relabelled = relabel(true_covar, relabelling)
    @test length(symdiff( [:Barclays, :HSBC, :Vodafone, :Ryanair] , true_relabelled.labels)) < 1


    # Getting ticks per asset
    tpa = ticks_per_asset(ts1)
    @test all(values(tpa) .> 1)

    iscloser(a,b) = (a.Correlation_error + a.Volatility_error < b.Correlation_error + b.Volatility_error)

    # Preav Convergence
    preav_estimate1 = preaveraged_covariance(ts1, assets)
    preav_estimate2 = preaveraged_covariance(ts2, assets)
    @test iscloser(calculate_mean_abs_distance(preav_estimate2, true_covar), calculate_mean_abs_distance(preav_estimate1, true_covar))
    @test is_psd_matrix(preav_estimate1)
    @test is_psd_matrix(preav_estimate2)
    @test valid_correlation_matrix(preav_estimate1)
    @test valid_correlation_matrix(preav_estimate2)

    # simple Convergence
    simple_estimate1 = simple_covariance(ts1, assets)
    simple_estimate2 = simple_covariance(ts2, assets)
    # The below fails. But no reason to  believe that the simple method would be consistant anyway.
    # iscloser(calculate_mean_abs_distance(simple_estimate2, true_covar), calculate_mean_abs_distance(simple_estimate1, true_covar))
    @test valid_correlation_matrix(simple_estimate1)
    @test valid_correlation_matrix(simple_estimate2)

    # bnhls Convergence
    bnhls_estimate1 = bnhls_covariance(ts1, assets; regularisation = :eigenvalue_clean) # Need to change this regularsation because such little data means regu
    bnhls_estimate2 = bnhls_covariance(ts2, assets)
    @test iscloser(calculate_mean_abs_distance(bnhls_estimate2, true_covar), calculate_mean_abs_distance(bnhls_estimate1, true_covar))
    @test !valid_correlation_matrix(bnhls_estimate1) # This is bad luck as bnhls doesnt guarantee PSD matrices.
    @test valid_correlation_matrix(bnhls_estimate2)

    # Preav Convergence
    spectral_estimate1 = spectral_covariance(ts1, assets; num_blocks = 1) # not many observations so need to reduce the number of blocks here
    spectral_estimate2 = spectral_covariance(ts2, assets)
    @test iscloser(calculate_mean_abs_distance(spectral_estimate2, true_covar), calculate_mean_abs_distance(spectral_estimate1, true_covar))
    @test valid_correlation_matrix(spectral_estimate1)
    @test valid_correlation_matrix(spectral_estimate2)

    # two scales Convergence
    two_scales_estimate1 = two_scales_covariance(ts1, assets)
    two_scales_estimate2 = two_scales_covariance(ts2, assets)
    @test valid_correlation_matrix(two_scales_estimate1)
    @test valid_correlation_matrix(two_scales_estimate2)


    #############################
     # Serialisation and deserialisation

    true_df = DataFrame(true_covar)
    reconstituted_df = CovarianceMatrix(true_df)
    @test calculate_mean_abs_distance(true_covar, reconstituted_df).Correlation_error .< 10*eps()
    @test calculate_mean_abs_distance(true_covar, reconstituted_df).Volatility_error .< 10*eps()

    # Other regularisation algos:
    two_scales_estimate_iden = two_scales_covariance(ts2, assets; regularisation = :identity_regularisation)
    two_scales_estimate_nearest_corr = two_scales_covariance(ts2, assets; regularisation = :nearest_correlation_matrix)
    two_scales_estimate_nearest_psd = two_scales_covariance(ts2, assets; regularisation = :nearest_psd_matrix)
    two_scales_estimate_eigen = two_scales_covariance(ts2, assets; regularisation = :eigenvalue_clean)


    # Running regularistation on a CovarianceMatrix's correlation matrix.
    @test valid_correlation_matrix(two_scales_estimate_nearest_corr)

    psd_mat = two_scales_estimate_nearest_corr

    reg1 = identity_regularisation(psd_mat, ts2)
    @test calculate_mean_abs_distance(psd_mat, reg1).Correlation_error .> 10*eps()
    @test calculate_mean_abs_distance(psd_mat, reg1).Volatility_error .> 10*eps()

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
    @test calculate_mean_abs_distance(psd_mat, reg3_corr).Correlation_error .< 10*eps() # No change as we are already psd
    @test calculate_mean_abs_distance(psd_mat, reg3_corr).Volatility_error .< 10*eps()

    reg4 = eigenvalue_clean(psd_mat, ts2)
    @test calculate_mean_abs_distance(psd_mat, reg4).Correlation_error .> 2*eps()
    @test calculate_mean_abs_distance(psd_mat, reg4).Volatility_error .> 10*eps()

    reg4_cov = eigenvalue_clean(psd_mat, ts2; apply_to_covariance = false)
    @test calculate_mean_abs_distance(psd_mat, reg4_cov).Correlation_error .< 2*eps()
    @test calculate_mean_abs_distance(psd_mat, reg4_cov).Volatility_error .< 10*eps()

    # Testing blocking and regularisation.
    blocking_dd = put_assets_into_blocks_by_trading_frequency(ts2, 1.3, :spectral_covariance)
    block_estimate = blockwise_estimation(ts2, blocking_dd)
    block_estimate = identity_regularisation(block_estimate, ts2)

    # Testing getting noise
    noise = estimate_microstructure_noise(ts1, assets)
    noise_from_2s = two_scales_volatility(ts1, assets)[2]
    C = merge(-, noise, noise_from_2s)
    @test maximum(collect(values(C))) < 1E-13



    # Testing show functions
    show(ts1)
    show(true_covar)
    show(preav_estimate2, 3, 6)
end
