module HighFrequencyCovariance

using DataFrames
using Dates
using Distributions
using LinearAlgebra
using Random
using Statistics
using StochasticIntegrals # Used for Monte Carlo Functions
using UnivariateFunctions # Used for Monte Carlo Functions

# Preliminary functions
include("00_Structs.jl")
export SortedDataFrame, CovarianceMatrix, +, -
export get_correlation, get_volatility
export make_nan_covariance_matrix, duration
export subset_to_tick, subset_to_time, calculate_mean_abs_distance
export calculate_mean_abs_distance_covar
export to_dataframe, valid_correlation_matrix, is_psd_matrix
export ticks_per_asset, get_assets
export show, relabel, plot, combine
include("01_helpers.jl")
export simple_differencing
include("02_subsample_ticks.jl")
export cov2cor, cor2cov, cov2cor_and_vol, covariance, construct_matrix_from_eigen, get_returns
export combine_covariance_matrices, rearrange
export squared_frobenius, squared_frobenius_distance
export get_all_refresh_times, latest_value, time_between_refreshes, random_value_in_interval
include("03_MonteCarlo.jl")
export make_random_psd_matrix_from_wishart, generate_random_path, ItoSet
export get_draws
include("04_Serialisation.jl")
export to_dataframe, dataframe_to_covariancematrix

# Volatility estimation techniques
include("10_volatility_simple.jl")
export simple_volatility
include("11_volatility_two_scales.jl")
export two_scales_volatility, default_num_grids

# Covariance Estimation techniques
include("20_covariance_simple.jl")
export simple_covariance, simple_covariance_given_returns, default_spacing, get_timegrid
include("21_covariance_bnhls.jl")
export HFC_Kernel, parzen, quadratic_spectral, fejer, tukey_hanning, bnhls_2008
export bnhls_covariance, preaveraging_end_returns
include("22_covariance_spectral.jl")
export spectral_covariance
include("23_covariance_preaveragingHY.jl")
export preaveraged_covariance
include("24_covariance_two_scales.jl")
export two_scales_covariance

# Regularisation Techniques
include("30_Regularisation_identity.jl")
export identity_regularisation
include("31_Regularisation_NearestCorrelationMatrix.jl")
export project_to_U, project_to_S, iterate_higham, nearest_correlation_matrix, nearest_psd_matrix
include("32_Regularisation_RMT.jl")
export eigenvalue_clean

# Blocking and regularisation.
include("40_Blocking.jl")
export blockwise_estimation, put_assets_into_blocks_by_trading_frequency

# Convenience Wrappers
include("50_ConvenienceFunctions.jl")
export estimate_volatility, estimate_covariance, regularise
export estimate_microstructure_noise

end
