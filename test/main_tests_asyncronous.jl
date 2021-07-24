using DataFrames
using LinearAlgebra
using Statistics: std, var, mean, cov
using HighFrequencyCovariance
using Random

brownian_corr_matrix = Hermitian([1.0 0.75 0.5 0.0;
                                  0.0 1.0 0.5 0.25;
                                  0.0 0.0 1.0 0.25;
                                  0.0 0.0 0.0 1.0])
assets = [:BARC, :HSBC, :VODL, :RYAL]
twister = MersenneTwister(1)

ts1, true_covar, micro_noise, update_rates = generate_random_path(4, 2000; brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05], twister = deepcopy(twister))
ts2, true_covar, micro_noise, update_rates = generate_random_path(4, 50000; brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05], twister = deepcopy(twister))
ts3 = deepcopy(ts2)
ts3.df.Value = exp.(ts3.df.Value)

iscloser(a,b) = (a.Correlation_error + a.Volatility_error < b.Correlation_error + b.Volatility_error)

# Preav Convergence
preav_estimate1 = preaveraged_covariance(ts1, assets)
preav_estimate2 = preaveraged_covariance(ts2, assets)
preav_estimate3 = preaveraged_covariance(ts3, assets; return_calc = HighFrequencyCovariance.log_returns)
iscloser(calculate_mean_abs_distance(preav_estimate2, true_covar), calculate_mean_abs_distance(preav_estimate1, true_covar))
calculate_mean_abs_distance(preav_estimate2, preav_estimate3).Correlation_error .< 10*eps()
calculate_mean_abs_distance(preav_estimate2, preav_estimate3).Volatility_error .< 10*eps()
valid_correlation_matrix(preav_estimate1)
valid_correlation_matrix(preav_estimate2)

# simple Convergence
simple_estimate1 = simple_covariance(ts1, assets)
simple_estimate2 = simple_covariance(ts2, assets)
simple_estimate3 = simple_covariance(ts3, assets; return_calc = HighFrequencyCovariance.log_returns)
iscloser(calculate_mean_abs_distance(simple_estimate2, true_covar), calculate_mean_abs_distance(simple_estimate1, true_covar))
calculate_mean_abs_distance(simple_estimate2, simple_estimate3).Correlation_error .< 10*eps()
calculate_mean_abs_distance(simple_estimate2, simple_estimate3).Volatility_error .< 10*eps()
valid_correlation_matrix(simple_estimate1)
valid_correlation_matrix(simple_estimate2)

# bnhls Convergence
bnhls_estimate1 = bnhls_covariance(ts1, assets)
bnhls_estimate2 = bnhls_covariance(ts2, assets)
bnhls_estimate3 = bnhls_covariance(ts3, assets; return_calc = HighFrequencyCovariance.log_returns)
iscloser(calculate_mean_abs_distance(bnhls_estimate2, true_covar), calculate_mean_abs_distance(bnhls_estimate1, true_covar))
calculate_mean_abs_distance(bnhls_estimate2, bnhls_estimate3).Correlation_error .< 100*eps()
calculate_mean_abs_distance(bnhls_estimate2, bnhls_estimate3).Volatility_error .< 10*eps()
valid_correlation_matrix(bnhls_estimate1)
valid_correlation_matrix(bnhls_estimate2)

# Preav Convergence
spectral_estimate1 = spectral_covariance(ts1, assets; num_blocks = 1) # not many observations so need to reduce the number of blocks here
spectral_estimate2 = spectral_covariance(ts2, assets)
spectral_estimate3 = spectral_covariance(ts3, assets; return_calc = HighFrequencyCovariance.log_returns)
iscloser(calculate_mean_abs_distance(spectral_estimate2, true_covar), calculate_mean_abs_distance(spectral_estimate1, true_covar))
calculate_mean_abs_distance(spectral_estimate2, spectral_estimate3).Correlation_error .< 10*eps()
calculate_mean_abs_distance(spectral_estimate2, spectral_estimate3).Volatility_error .< 10*eps()
valid_correlation_matrix(spectral_estimate1)
valid_correlation_matrix(spectral_estimate2)

# two scales Convergence
two_scales_estimate1 = two_scales_covariance(ts1, assets)
two_scales_estimate2 = two_scales_covariance(ts2, assets)
two_scales_estimate3 = two_scales_covariance(ts3, assets; return_calc = HighFrequencyCovariance.log_returns)
iscloser(calculate_mean_abs_distance(two_scales_estimate2, true_covar), calculate_mean_abs_distance(two_scales_estimate1, true_covar))
calculate_mean_abs_distance(two_scales_estimate2, two_scales_estimate3).Correlation_error .< 10*eps()
calculate_mean_abs_distance(two_scales_estimate2, two_scales_estimate3).Volatility_error .< 10*eps()
valid_correlation_matrix(two_scales_estimate1)
valid_correlation_matrix(two_scales_estimate2)



two_scales_volatility(ts3, assets; return_calc = HighFrequencyCovariance.log_returns)




two_scales_volatility(ts3, assets)

#############################
 # Serialisation and deserialisation

true_df = to_dataframe(true_covar)
reconstituted_df = dataframe_to_covariancematrix(true_df)
calculate_mean_abs_distance(true_covar, reconstituted_df).Correlation_error .< 10*eps()
calculate_mean_abs_distance(true_covar, reconstituted_df).Volatility_error .< 10*eps()

# Other regularisation algos:
two_scales_estimate_iden = two_scales_covariance(ts2, assets; regularisation = :Identity)
two_scales_estimate_nearest_corr = two_scales_covariance(ts2, assets; regularisation = :NearestCorrelation)
two_scales_estimate_nearest_psd = two_scales_covariance(ts2, assets; regularisation = :NearestPSD)
two_scales_estimate_eigen = two_scales_covariance(ts2, assets; regularisation = :EigenClean)

# Running regularistation on a CovarianceMatrix's correlation matrix.
valid_correlation_matrix(two_scales_estimate_nearest_corr)


reg1 = identity_regularisation(psd_mat, ts2)
calculate_mean_abs_distance(psd_mat, reg1).Correlation_error .> 10*eps()
calculate_mean_abs_distance(psd_mat, reg1).Volatility_error .> 10*eps()

reg1_corr = identity_regularisation(psd_mat, ts2; apply_to_covariance = false)
calculate_mean_abs_distance(psd_mat, reg1_corr).Correlation_error .> 10*eps()
calculate_mean_abs_distance(psd_mat, reg1_corr).Volatility_error .< 10*eps()

reg2 = nearest_correlation_matrix(psd_mat, ts2)
calculate_mean_abs_distance(psd_mat, reg2).Correlation_error .< 10*eps() # No change as we are already psd
calculate_mean_abs_distance(psd_mat, reg2).Volatility_error .< 10*eps()

reg3 = nearest_psd_matrix(psd_mat, ts2)
calculate_mean_abs_distance(psd_mat, reg3).Correlation_error .< 10*eps() # No change as we are already psd
calculate_mean_abs_distance(psd_mat, reg3).Volatility_error .< 1000*eps()

reg3_corr = nearest_psd_matrix(psd_mat, ts2; apply_to_covariance = false)
calculate_mean_abs_distance(psd_mat, reg3_corr).Correlation_error .< 10*eps() # No change as we are already psd
calculate_mean_abs_distance(psd_mat, reg3_corr).Volatility_error .< 10*eps()

reg4 = eigenvalue_clean(psd_mat, ts2)
calculate_mean_abs_distance(psd_mat, reg4).Correlation_error .> 2*eps()
calculate_mean_abs_distance(psd_mat, reg4).Volatility_error .> 10*eps()

reg4_cov = eigenvalue_clean(psd_mat, ts2; apply_to_covariance = false)
calculate_mean_abs_distance(psd_mat, reg4_cov).Correlation_error .> 2*eps()
calculate_mean_abs_distance(psd_mat, reg4_cov).Volatility_error .< 10*eps()

# Testing blocking and regularisation.
blocking_dd = put_assets_into_blocks_by_trading_frequency(ts2, 1.3, spectral_covariance)
block_estimate = blockwise_estimation(ts2, blocking_dd)
block_estimate = identity_regularisation(block_estimate, ts2)
iscloser(calculate_mean_abs_distance(block_estimate, true_covar), calculate_mean_abs_distance(spectral_estimate2, true_covar))
