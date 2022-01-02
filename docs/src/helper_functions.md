
```@meta
CurrentModule = HighFrequencyCovariance
```

# Helper Functions

```@index
Pages = ["helper_functions.md"]
```

## Working with SortedDataFrame structs

```@docs
    get_assets
    ticks_per_asset
    duration
    subset_to_tick
    subset_to_time
```

## Working with CovarianceMatrix structs

```@docs
    covariance
    get_correlation
    get_volatility
    make_nan_covariance_matrix
    combine_covariance_matrices
    rearrange
    cov2cor
    cor2cov
    cov2cor_and_vol
    construct_matrix_from_eigen
    get_returns
    valid_correlation_matrix
    is_psd_matrix
```

## Blocking and Regularisation Functions

```@docs
    put_assets_into_blocks_by_trading_frequency
    blockwise_estimation
```

## Monte Carlo

```@docs
    generate_random_path
    ItoSet
    get_draws
```

## For getting a DataFrame version of a CovarianceMatrix and vice versa.

```@docs
    HighFrequencyCovariance.to_dataframe
    dataframe_to_covariancematrix
```
