"""
    syncronised_simple_eigen(ts::SortedDataFrame; assets::Vector{Symbol} = unique(ts.df[:,ts.grouping]), return_type = :simple)
This uses refresh time sampling to get returns. Then naive covariance estimation. Then eigenvalue cleaning.
### Takes
* ts - A dataframe with data ticks (prices for each asset).
* assets - What assets to estimate.
* return_type - What type of returns (:simple or :log)
### Returns
* A CovarianceMatrix.
"""
function syncronised_simple_eigen(ts::SortedDataFrame; assets::Vector{Symbol} = unique(ts.df[:,ts.grouping]), return_type = :simple)
  # Creating Syncronised data.
  at_times = get_all_refresh_times(ts; assets = assets)
  dd_compiled = latest_value(ts, at_times; assets = assets)
  dd = get_returns(dd_compiled; returns = return_type)
  # Making naive covariance matrix.
  covar = simple_covariance(dd)
  # Eigenvalue cleaning it.
  obs = nrow(dd)
  covar.correlation = eigenvalue_clean(covar.correlation; obs = obs)
  return rearrange(covar, assets)
end




"""
    syncronised_kernel_eigen(ts::DataFrame; returns_interval::Real = 1.0, return_type = :simple)
This uses refresh time sampling to get returns. Then BNHLS kernel covariance estimation. Then eigenvalue cleaning.
### Takes
* ts - A dataframe with data ticks (prices for each asset).
* assets - What assets to estimate.
* returns_interval - What spacing should covariance magnitudes reflect?
* return_type - What type of returns (:simple or :log)
### Returns
* A Hermitian covariance matrix and a vector of labels.
"""
function syncronised_bnhls_eigen(ts::SortedDataFrame; assets::Vector{Symbol} = unique(ts.df[:,ts.grouping]), return_type::Symbol = :simple, covariance_options::Dict = Dict())
  # Creating Syncronised data.
  at_times = get_all_refresh_times(ts; assets = assets)
  dd_compiled = latest_value(ts, at_times; assets = assets)
  dd = get_returns(dd_compiled; returns = return_type)
  # Making bnhls covariance matrix.
  covar = bnhls_covariance_estimate(dd; assets = assets, covariance_options...)
  # Eigenvalue cleaning it.
  obs = nrow(dd)
  covar.correlation = eigenvalue_clean(covar.correlation; obs = obs)
  return rearrange(covar, assets)
end

function estimate_covariance(ts::SortedDataFrame; assets::Vector{Symbol} = unique(ts.df[:,ts.grouping]), return_type::Symbol = :simple, covariance_options::Dict = Dict())


end
