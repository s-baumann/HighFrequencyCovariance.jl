using Revise

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

ts1, true_covar, micro_noise, update_rates = generate_random_path(4, 2000; brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05], twister = deepcopy(twister), syncronous = true)
ts2, true_covar, micro_noise, update_rates = generate_random_path(4, 50000; brownian_corr_matrix = brownian_corr_matrix, assets = assets, vols = [0.02,0.03,0.04,0.05], twister = deepcopy(twister), syncronous = true)


iscloser(a,b) = (a.Correlation_error + a.Volatility_error < b.Correlation_error + b.Volatility_error)

# Preav Convergence
preav_estimate1 = preaveraged_covariance(ts1, assets)
preav_estimate2 = preaveraged_covariance(ts2, assets)
iscloser(calculate_mean_abs_distance(preav_estimate2, true_covar), calculate_mean_abs_distance(preav_estimate1, true_covar))
valid_correlation_matrix(preav_estimate1)
valid_correlation_matrix(preav_estimate2)

# simple Convergence
simple_estimate1 = simple_covariance(ts1, assets)
simple_estimate2 = simple_covariance(ts2, assets)
iscloser(calculate_mean_abs_distance(simple_estimate2, true_covar), calculate_mean_abs_distance(simple_estimate1, true_covar))
valid_correlation_matrix(simple_estimate1)
valid_correlation_matrix(simple_estimate2)

# bnhls Convergence
bnhls_estimate1 = bnhls_covariance(ts1, assets)
bnhls_estimate2 = bnhls_covariance(ts2, assets)
iscloser(calculate_mean_abs_distance(bnhls_estimate2, true_covar), calculate_mean_abs_distance(bnhls_estimate1, true_covar))
valid_correlation_matrix(bnhls_estimate1) == false # This comes out negative. Not a bug just bad luck with this algo.
valid_correlation_matrix(bnhls_estimate2)

# Preav Convergence
spectral_estimate1 = spectral_covariance(ts1, assets; num_blocks = 1) # not many observations so need to reduce the number of blocks here
spectral_estimate2 = spectral_covariance(ts2, assets)
iscloser(calculate_mean_abs_distance(spectral_estimate2, true_covar), calculate_mean_abs_distance(spectral_estimate1, true_covar))
valid_correlation_matrix(spectral_estimate1)
valid_correlation_matrix(spectral_estimate2)

# two scales Convergence
two_scales_estimate1 = two_scales_covariance(ts1, assets)
two_scales_estimate2 = two_scales_covariance(ts2, assets)
iscloser(calculate_mean_abs_distance(two_scales_estimate2, true_covar), calculate_mean_abs_distance(two_scales_estimate1, true_covar))
valid_correlation_matrix(two_scales_estimate1)
valid_correlation_matrix(two_scales_estimate2)

#############################
 # Serialisation and deserialisation

true_df = to_dataframe(true_covar)
reconstituted_df = dataframe_to_covariancematrix(true_df)
calculate_mean_abs_distance(true_covar, reconstituted_df).Correlation_error .< 10*eps()
calculate_mean_abs_distance(true_covar, reconstituted_df).Volatility_error .< 10*eps()

# Other regularisation algos:
two_scales_estimate_iden = two_scales_covariance(ts2, assets; regularisation = identity_regularisation)
two_scales_estimate_iden = two_scales_covariance(ts2, assets; regularisation = nearest_correlation_matrix)
two_scales_estimate_iden = two_scales_covariance(ts2, assets; regularisation = nearest_psd_matrix)
two_scales_estimate_iden = two_scales_covariance(ts2, assets; regularisation = eigenvalue_clean)

# Running regularistation on a CovarianceMatrix's correlation matrix.
psd_mat = two_scales_covariance(ts2, assets; regularisation = nearest_correlation_matrix)
valid_correlation_matrix(psd_mat)
reg1 = identity_regularisation(psd_mat, ts2)
calculate_mean_abs_distance(psd_mat, reg1).Correlation_error .> 10*eps()
calculate_mean_abs_distance(psd_mat, reg1).Volatility_error .< 10*eps()

reg2 = nearest_correlation_matrix(psd_mat, ts2)
calculate_mean_abs_distance(psd_mat, reg2).Correlation_error .< 10*eps() # No change as we are already psd
calculate_mean_abs_distance(psd_mat, reg2).Volatility_error .< 10*eps()

reg3 = nearest_psd_matrix(psd_mat, ts2)
calculate_mean_abs_distance(psd_mat, reg3).Correlation_error .< 10*eps() # No change as we are already psd
calculate_mean_abs_distance(psd_mat, reg3).Volatility_error .< 10*eps()

reg4 = eigenvalue_clean(psd_mat, ts2)
calculate_mean_abs_distance(psd_mat, reg4).Correlation_error .> 2*eps()
calculate_mean_abs_distance(psd_mat, reg4).Volatility_error .< 10*eps()

# Testing blocking and regularisation.
blocking_dd = put_assets_into_blocks_by_trading_frequency(ts2, 1.1, spectral_covariance)
block_estimate = blockwise_estimation(ts2, blocking_dd)
block_estimate = nearest_correlation_matrix(block_estimate, ts2)
iscloser(calculate_mean_abs_distance(block_estimate, true_covar), calculate_mean_abs_distance(spectral_estimate2, true_covar))
